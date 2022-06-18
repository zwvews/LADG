import torch
from algorithms.group_algorithm import GroupAlgorithm
from scheduler import initialize_scheduler
from optimizer import initialize_optimizer
from torch.nn.utils import clip_grad_norm_
from utils import move_to
from models.initializer import initialize_model
from algorithms.gnn_layer import GatedGCNLayer, DiscGatedGCN
import dgl
import torch.nn.functional as F
from sklearn import preprocessing


class LADG(GroupAlgorithm):

    def __init__(self, config, d_out, grouper, loss, metric, n_train_steps, is_group_in_train):
        self.loss = loss
        logged_metrics = [self.loss, ]
        if metric is not None:
            self.metric = metric
            logged_metrics.append(self.metric)
        else:
            self.metric = None
        self.device = config.device
        (featurizer, classifier) = initialize_model(config, d_out, is_featurizer=True)
        featurizer = featurizer.to(self.device)
        classifier = classifier.to(self.device)

        params = config
        self.params = config
        dis_latentDim = params.latentDimDisc

        disc = DiscGatedGCN(featurizer.d_out, dis_latentDim)

        disc = disc.to(self.device)
        self.opt_disc = initialize_optimizer(config, disc)
        self.opt_fea = initialize_optimizer(config, featurizer)
        self.opt_cls = initialize_optimizer(config, classifier)

        self.max_grad_norm = config.max_grad_norm

        scheduler_fea = initialize_scheduler(config, self.opt_fea, n_train_steps)
        scheduler_cls = initialize_scheduler(config, self.opt_cls, n_train_steps)
        scheduler_disc = initialize_scheduler(config, self.opt_disc, n_train_steps)
        super().__init__(
            device=config.device,
            grouper=grouper,
            logged_metrics=logged_metrics,
            logged_fields=['objective'],
            schedulers=[scheduler_fea, scheduler_cls, scheduler_disc],
            scheduler_metric_names=[config.scheduler_metric_name, ],
            no_group_logging=config.no_group_logging,
        )
        self.num_classes = d_out
        self.num_domains = is_group_in_train.sum().item()
        # membank not enabled
        self.cache_mcr2All_score = torch.Tensor([0.]).to(self.device)
        self.cache_mcr2All_scoreList = []
        self.update_count = torch.tensor([0])
        self.n_train_steps = n_train_steps

        countInDomain = 0
        labelConverter = torch.zeros(is_group_in_train.shape[0], 1).squeeze() - 1
        for i in range(is_group_in_train.shape[0]):
            if is_group_in_train[i]:
                labelConverter[i] = countInDomain
                countInDomain += 1
        self.labelConverter = labelConverter

        self.featurizer = featurizer
        self.classifier = classifier
        self.disc = disc

    def process_batch(self, batch):
        x, y_true, metadata = batch
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)
        g = move_to(self.grouper.metadata_to_group(metadata), self.device)

        emb = self.featurizer(x)
        outputs = self.classifier(emb)

        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'metadata': metadata,
        }
        return results

    def evaluate(self, batch):

        assert not self.is_training
        results = self.process_batch(batch)
        loss = self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)
        results['objective'] = loss.item()
        self.update_log({k: v for k, v in results.items() if k != 'emb'})
        return self.sanitize_dict(results)

    def update(self, batch):

        assert self.is_training
        x, y_true, metadata = batch
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)

        x, y_true, metadata, labelDomain_onehot, domainLabel = self.processDomainLabel(x, y_true, metadata)
        g1 = move_to(self.grouper.metadata_to_group(metadata), self.device)

        indexSample = torch.arange(x.shape[0])
        self.update_count += 1
        steps = self.update_count  # epoch is update count
        results_wilds = {
            'g': g1,
            'y_true': y_true,
            'metadata': metadata,
        }

        results_wilds = self.oursupdate(x, y_true, domainLabel, labelDomain_onehot, indexSample, steps, metadata,
                                        results_wilds)
        self.step_schedulers(
            is_epoch=False,
            metrics=results_wilds,
            log_access=False)
        self.update_log(results_wilds)
        return self.sanitize_dict(results_wilds)

    def processDomainLabel(self, x, y_true, metadata):
        g = move_to(self.grouper.metadata_to_group(metadata), self.device)
        domainLabel = g
        le = preprocessing.LabelEncoder()
        domainLabel = torch.Tensor(le.fit_transform(domainLabel.cpu().numpy())).to(self.device)
        labelDomain_onehot = F.one_hot(domainLabel.long(), int(domainLabel.max().item() + 1))
        return x, y_true, metadata, labelDomain_onehot, domainLabel

    def oursupdate(self, imgs, targetLabel, domainLabel, labelDomain_onehot, indexSample, epoch, metadata,
                   results_wilds):
        if epoch <= self.params.warmupFeaSteps:
            status = 'warmFea'
        elif epoch <= self.params.warmupDiscSteps + self.params.warmupFeaSteps:
            status = 'warmDisc'
        elif epoch >= self.params.warmupDiscSteps + self.params.warmupFeaSteps:
            status = 'train'

        emb = self.featurizer(imgs)
        emb = torch.flatten(emb, 1)
        if True:
            emb_norm_all = F.normalize(emb, dim=-1)
            I = torch.eye(emb_norm_all.shape[0]).to(emb_norm_all.device)
            embAll_sim = emb_norm_all.matmul(emb_norm_all.t())
            SpaceScore = torch.logdet(I + emb_norm_all.shape[0] / emb_norm_all.shape[1] / 0.01 * embAll_sim)

        if status == 'warmFea':
            self.opt_fea.zero_grad()
            self.opt_cls.zero_grad()
            loss, results_wilds = self.forwardCls(emb, targetLabel, indexSample, results_wilds)
            loss.backward()
            self.opt_fea.step()
            self.opt_cls.step()
        elif status == 'warmDisc':
            self.cache_mcr2All_scoreList.append(SpaceScore.detach())

            self.updateDomainBranch_lp(emb.detach(), domainLabel,
                                                                  indexSample, pretrain=True,
                                                                  labelDomain_onehot=labelDomain_onehot)
            _, results_wilds = self.forwardCls(emb, targetLabel, indexSample, results_wilds)
        elif status == 'train':
            lossMCR = self.mcrloss(SpaceScore)
            self.updateDomainBranch_lp(emb.detach(), domainLabel,
                                                           indexSample, pretrain=False,
                                                           labelDomain_onehot=labelDomain_onehot)

            self.opt_fea.zero_grad()
            self.opt_cls.zero_grad()
            lossTarget, results_wilds = self.forwardCls(emb, targetLabel, indexSample, results_wilds)

            embs_indvi = self.discForward(domainLabel, emb)
            domainPred = self.calDomainLP(embs_indvi, domainLabel, indexSample, adv=True,
                                          labelDomain_onehot=labelDomain_onehot)

            labelDomainPred = F.normalize(labelDomain_onehot.sum(dim=0).float(), p=1, dim=0)
            labelDomainPred = labelDomainPred.unsqueeze(0)

            lossAdv = -labelDomainPred.detach() * torch.log(domainPred + 1e-7)

            lossAdv = (lossAdv.sum(dim=-1).mean())
            loss = lossTarget + self.params.weightAdv * (lossAdv) + self.params.weightMCR * lossMCR
            loss.backward()
            self.opt_fea.step()
            self.opt_cls.step()

        return results_wilds

    def mcrloss(self, score):
        if torch.sum(self.cache_mcr2All_score == 0) == 1:
            self.cache_mcr2All_score = torch.Tensor(self.cache_mcr2All_scoreList).mean()
        else:
            beta = self.params.coefMCRUpdate
            self.cache_mcr2All_score = beta * self.cache_mcr2All_score + (1 - beta) * score.detach()
        lossMCR = torch.log(
            torch.cosh(self.params.tolFactor * (score - self.cache_mcr2All_score))) / self.params.tolFactor
        return lossMCR.mean()

    def forwardCls(self, emb, label, indexSample, results_wilds):
        predTarget = self.classifier(emb)
        results_wilds['y_pred'] = predTarget.detach()
        loss = self.loss.compute(predTarget, label, return_dict=False)
        results_wilds['objective'] = loss.item()
        return loss, results_wilds

    def updateDomainBranch_lp(self, emb, domainLabel, indexSample, pretrain=False, labelDomain_onehot=None):
        label_onehot = labelDomain_onehot
        numIter = 1 if pretrain else self.params.numIterDisc
        for iterDisc in range(numIter):
            self.opt_disc.zero_grad()
            embs_indvi = self.discForward(domainLabel, emb)
            domainPred = self.calDomainLP(embs_indvi, domainLabel, indexSample, pretrain, False, labelDomain_onehot)
            lossDisc = - label_onehot * torch.log(domainPred + 1e-7)
            lossDisc = lossDisc.sum(dim=-1).mean()
            lossDisc.backward()
            self.opt_disc.step()


    def calDomainLP(self, embs_indvi, domainLabel, indexSample, pretrain=False, adv=False, labelDomain_onehot=None):
        bs = embs_indvi.shape[0]
        labelDomain_onehot = labelDomain_onehot
        embs_indvi_norm = F.normalize(embs_indvi, dim=-1)
        simDisc_0 = embs_indvi_norm.matmul(embs_indvi_norm.t())
        simDisc_0.fill_diagonal_(-1e7)
        if adv:
            if self.params.topk > 0:
                topk = self.params.topk
                topk_value, topk_indice = torch.topk(simDisc_0, topk, dim=-1)
                simDisc = (-1e7 * torch.ones_like(simDisc_0)).scatter(1, topk_indice,
                                                                      topk_value)
            else:
                simDisc = simDisc_0
        else:
            if self.params.topk > 0:
                topk = int(1.5 * labelDomain_onehot.sum(dim=0)[0])
                topk = topk if topk > self.params.topk else self.params.topk
                topk_value, topk_indice = torch.topk(simDisc_0, topk, dim=-1)
                simDisc = (-1e7 * torch.ones_like(simDisc_0)).scatter(1, topk_indice,
                                                                      topk_value)
            else:
                simDisc = simDisc_0

        simDisc = torch.exp(self.params.tmpScale * simDisc) - torch.exp(
            torch.Tensor([-1 * self.params.tmpScale])).cuda()
        simDisc = torch.relu(simDisc)
        D = torch.sum(simDisc, dim=-1)
        D_sqrt = 1 / torch.sqrt(D + 1e-7)
        simDisc_normalized = (torch.diag(D_sqrt).matmul(simDisc)).matmul(torch.diag(D_sqrt))
        predDomain = torch.linalg.solve(
            (torch.eye(D.shape[0]).to(D.device) - self.params.lprestartrate * simDisc_normalized),
            labelDomain_onehot.float())
        predDomain = torch.relu(predDomain)
        domainPred = F.softmax(predDomain, dim=-1)
        return domainPred

    def discForward(self, domainLabel, emb):
        device = emb.device
        numNodes = emb.shape[0]
        edge_src = torch.repeat_interleave(torch.arange(numNodes), numNodes).to(device)
        edge_dst = torch.repeat_interleave(torch.arange(numNodes), numNodes).reshape(numNodes,
                                                                                     numNodes).t().flatten().to(device)
        g = dgl.graph((edge_src, edge_dst), num_nodes=numNodes, device=device)
        g.ndata['feat'] = emb
        g.edata['feat'] = torch.ones(g.number_of_edges(), emb.shape[-1]).to(device)
        h = self.disc(g, emb, g.edata['feat'])
        return h


