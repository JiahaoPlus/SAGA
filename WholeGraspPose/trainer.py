import os
import shutil
import sys

sys.path.append('.')
sys.path.append('..')
import json
import pickle
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from ignite.contrib.metrics import ROC_AUC
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from utils.train_helper import EarlyStopping, point2point_signed
from utils.utils import makelogger, makepath, to_cpu

from WholeGraspPose.data.dataloader import LoadData
from WholeGraspPose.models.models import FullBodyGraspNet


class Trainer:

    def __init__(self,cfg, inference=False, logger=None):


        """continue to train from latest checkpoint"""
        if cfg.continue_train:
            checkpoint = torch.load(os.path.join(cfg.work_dir, 'checkpoint.pt'))
            cfg = checkpoint['cfg']
            cfg.continue_train = True
            cfg.best_net = None

        self.dtype = torch.float32
        torch.manual_seed(cfg.seed)

        starttime = datetime.now().replace(microsecond=0)
        makepath(cfg.work_dir, isfile=False)

        if logger:
            self.logger = logger
        else:
            self.logger = makelogger(makepath(os.path.join(cfg.work_dir, '%s.log' % (cfg.exp_name)), isfile=True)).info

        if not inference:
            summary_logdir = os.path.join(cfg.work_dir, 'summaries')
            self.swriter = SummaryWriter(log_dir=summary_logdir)
            self.logger('[%s] - Started training GrabNet, experiment code %s' % (cfg.exp_name, starttime))
            self.logger('tensorboard --logdir=%s' % summary_logdir)
            self.logger('Torch Version: %s\n' % torch.__version__)
            self.logger('Base dataset_dir is %s' % cfg.dataset_dir)

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.empty_cache()
        self.device = torch.device("cuda:%d" % cfg.cuda_id if torch.cuda.is_available() else "cpu")

        gpu_brand = torch.cuda.get_device_name(cfg.cuda_id) if use_cuda else None
        gpu_count = torch.cuda.device_count() if cfg.use_multigpu else 1
        if use_cuda:
            self.logger('Using %d CUDA cores [%s] for training!' % (gpu_count, gpu_brand))

        self.logger(cfg)

        self.load_data(cfg, inference)

        self.full_grasp_net = FullBodyGraspNet(cfg).to(self.device)

        if cfg.use_multigpu:
            self.full_grasp_net = nn.DataParallel(self.full_grasp_net)
            self.logger("Training on Multiple GPU's")

        vars_net = [var[1] for var in self.full_grasp_net.named_parameters()]

        net_n_params = sum(p.numel() for p in vars_net if p.requires_grad)
        self.logger('Total Trainable Parameters for ContactNet is %2.2f M.' % ((net_n_params) * 1e-6))

        self.optimizer_net = optim.Adam(vars_net, lr=cfg.base_lr, weight_decay=cfg.reg_coef)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer_net, milestones=[20,40,60], gamma=0.5)
        self.best_loss_net = np.inf

        self.try_num = cfg.try_num
        self.start_epoch = 0
        self.cfg = cfg
        self.full_grasp_net.cfg = cfg

        if cfg.best_net is not None:
            self._get_net_model().load_state_dict(torch.load(cfg.best_net, map_location=self.device), strict=False)
            self.logger('Restored ContactNet model from %s' % cfg.best_net)
        
        if cfg.continue_train:
            self.full_grasp_net, self.optimizer_net, self.start_epoch = self.load_ckp(checkpoint, self._get_net_model(), self.optimizer_net)
            self.logger('Resume Training from %s' % cfg.work_dir)

        self.epoch_completed = self.start_epoch

        self.LossL1 = torch.nn.L1Loss(reduction='none')
        self.LossL2 = torch.nn.MSELoss(reduction='none')

        self.ROC_AUC_object = ROC_AUC()
        self.ROC_AUC_marker = ROC_AUC()


    def load_data(self,cfg, inference):

        kwargs = {'num_workers': cfg.n_workers,
                  'batch_size':cfg.batch_size,
                  'shuffle':True,
                  'drop_last':True
                  }

        if not inference:
            ds_name = 'train'
            ds_train = LoadData(dataset_dir=cfg.dataset_dir, ds_name=ds_name, gender=cfg.gender, motion_intent=cfg.motion_intent, object_class=cfg.object_class)
            self.ds_train = DataLoader(ds_train, **kwargs)

            ds_name = 'val'
            ds_val = LoadData(dataset_dir=cfg.dataset_dir, ds_name=ds_name, gender=cfg.gender)
            self.ds_val = DataLoader(ds_val, **kwargs)

            self.logger('Dataset Train, Valid size respectively: %.2f M, %.2f K' %
                   (len(self.ds_train.dataset) * 1e-6, len(self.ds_val.dataset) * 1e-3))
            self.n_obj_verts = ds_train[0]['verts_object'].shape[0]

        else:
            ds_name = 'test'
            ds_test = LoadData(dataset_dir=cfg.dataset_dir, ds_name=ds_name, gender=cfg.gender)
            self.ds_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

            # self.bps = ds_test.bps
            self.logger('Dataset Test size: %.2f K' %
                   (len(self.ds_test.dataset) * 1e-3))
            self.n_obj_verts = ds_test[0]['verts_object'].shape[0]


    def _get_net_model(self):
        return self.full_grasp_net.module if isinstance(self.full_grasp_net, torch.nn.DataParallel) else self.full_grasp_net

    def save_net(self):
        torch.save(self.full_grasp_net.module.state_dict()
                   if isinstance(self.full_grasp_net, torch.nn.DataParallel)
                   else self.full_grasp_net.state_dict(), self.cfg.best_net)
    def save_ckp(self, state, checkpoint_dir):
        f_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
        torch.save(state, f_path)

    def load_ckp(self, checkpoint, model, optimizer):
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer, checkpoint['epoch']

    def train(self):

        self.full_grasp_net.train()

        save_every_it = len(self.ds_train) // self.cfg.log_every_epoch

        train_loss_dict_net = {}
        torch.autograd.set_detect_anomaly(True)

        for it, dorig in enumerate(self.ds_train):
            dorig = {k: dorig[k].to(self.device) for k in dorig.keys() if k!='smplxparams'}

            self.optimizer_net.zero_grad()

            if self.fit_net:
                dorig['verts_object'] = dorig['verts_object'].permute(0,2,1)
                dorig['feat_object'] = dorig['feat_object'].permute(0,2,1)
                dorig['contacts_object'] = dorig['contacts_object'].view(dorig['contacts_object'].shape[0], 1, -1)
                dorig['contacts_markers'] = dorig['contacts_markers'].view(dorig['contacts_markers'].shape[0], -1, 1)
                drec_net = self.full_grasp_net(**dorig)
                loss_total_net, cur_loss_dict_net = self.loss_net(dorig, drec_net)

                loss_total_net.backward()
                self.optimizer_net.step()

                train_loss_dict_net = {k: train_loss_dict_net.get(k, 0.0) + v.item() for k, v in cur_loss_dict_net.items()}
                if it % (save_every_it + 1) == 0:
                    cur_train_loss_dict_net = {k: v / (it + 1) for k, v in train_loss_dict_net.items()}
                    train_msg = self.create_loss_message(cur_train_loss_dict_net,
                                                        # expr_ID=self.cfg.expr_ID,
                                                        epoch_num=self.epoch_completed,
                                                        model_name='MarkerNet',
                                                        it=it,
                                                        try_num=self.try_num,
                                                        mode='train')

                    self.logger(train_msg)

                self.ROC_AUC_object.update((drec_net['contacts_object'].view(-1, 1).detach().cpu(), dorig['contacts_object'].squeeze().view(-1, 1).detach().cpu()))
                self.ROC_AUC_marker.update((drec_net['contacts_markers'].view(-1, 1).detach().cpu(), dorig['contacts_markers'].squeeze().view(-1, 1).detach().cpu()))



        train_loss_dict_net = {k: v / len(self.ds_train) for k, v in train_loss_dict_net.items()}

        return train_loss_dict_net#, train_loss_dict_rnet

    def evaluate(self, ds_name='val'):
        self.full_grasp_net.eval()
        self.ROC_AUC_object.reset()
        self.ROC_AUC_marker.reset()

        eval_loss_dict_net = {}

        data = self.ds_val if ds_name == 'val' else self.ds_test

        with torch.no_grad():
            for dorig in data:
                dorig = {k: dorig[k].to(self.device) for k in dorig.keys() if k!='smplxparams'}
                dorig['verts_object'] = dorig['verts_object'].permute(0,2,1)
                dorig['feat_object'] = dorig['feat_object'].permute(0,2,1)
                dorig['contacts_object'] = dorig['contacts_object'].view(dorig['contacts_object'].shape[0], 1, -1)
                dorig['contacts_markers'] = dorig['contacts_markers'].view(dorig['contacts_markers'].shape[0], -1, 1)

                if self.fit_net:
                    drec_net = self.full_grasp_net(**dorig)
                    loss_total_net, cur_loss_dict_net = self.loss_net(dorig, drec_net)

                    eval_loss_dict_net = {k: eval_loss_dict_net.get(k, 0.0) + v.item() for k, v in cur_loss_dict_net.items()}
                
                self.ROC_AUC_object.update((drec_net['contacts_object'].view(-1, 1).detach().cpu(), dorig['contacts_object'].squeeze().view(-1, 1).detach().cpu()))
                self.ROC_AUC_marker.update((drec_net['contacts_markers'].view(-1, 1).detach().cpu(), dorig['contacts_markers'].squeeze().view(-1, 1).detach().cpu()))


            eval_loss_dict_net = {k: v / len(data) for k, v in eval_loss_dict_net.items()}

        return eval_loss_dict_net


    def loss_net(self, dorig, drec, ds_name='train'):
        device = dorig['verts_object'].device
        dtype = dorig['verts_object'].dtype

        ################################ loss weight
        if self.cfg.kl_annealing:
            weight_kl = self.cfg.kl_coef
            weight_rec = 0.5
        else:
            weight_kl = self.cfg.kl_coef
            weight_rec = 1-self.cfg.kl_coef
        marker_weight = self.cfg.marker_weight

        ################################# KL loss
        q_z = torch.distributions.normal.Normal(drec['mean'], drec['std'])
        p_z = torch.distributions.normal.Normal(
            loc=torch.tensor(np.zeros([self.cfg.batch_size, self.cfg.latentD]), requires_grad=False).to(device).type(dtype),
            scale=torch.tensor(np.ones([self.cfg.batch_size, self.cfg.latentD]), requires_grad=False).to(device).type(dtype))
        loss_kl = torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1]))
        if self.cfg.robustkl:
            loss_kl = torch.sqrt(1 + loss_kl**2)-1
        loss_kl *= weight_kl

        ################################# object contactnet loss
        target = dorig['contacts_object'].to(device).squeeze()
        # object contact weighted BCE loss
        weight = torch.ones(target.shape[0], target.shape[1]).to(device)
        weight[torch.where(target==1)] = 3
        loss_object_contact_rec = F.binary_cross_entropy(input=drec['contacts_object'].squeeze().float(), target=target.float(), weight=weight, reduction='none')
        loss_object_contact_rec = weight_rec * torch.mean(torch.mean(loss_object_contact_rec, dim=1))

        ################################# markers contact loss
        target = dorig['contacts_markers'].to(device).squeeze()
        # marker contact weighted BCE loss
        weight = torch.ones(target.shape[0], target.shape[1]).to(device)
        weight[torch.where(target==1)] = 5
        loss_markers_contact_rec = F.binary_cross_entropy(input=drec['contacts_markers'].squeeze().float(), target=target.float(), weight=weight, reduction='none')
        loss_markers_contact_rec = weight_rec * torch.mean(torch.mean(loss_markers_contact_rec, dim=1))

        ################################# markernet loss
        markers = drec['markers']
        markers_gt = dorig['markers']

        markers_rhand = torch.cat([markers[:, 64:79, :], markers[:, -22:, :]], dim=1)
        markers_rhand_gt = torch.cat([markers_gt[:, 64:79, :], markers_gt[:, -22:, :]], dim=1)
        
        o2h, h2o_signed, o2h_idx, _ = point2point_signed(markers, dorig['verts_object'].permute(0,2,1), y_normals=dorig['normal_object'])
        o2h_gt, h2o_signed_gt, o2h_gt_idx, _ = point2point_signed(markers_gt, dorig['verts_object'].permute(0,2,1), y_normals=dorig['normal_object'])
        

        ################################# markers xyz rec loss
        loss_marker_rec = self.LossL1(markers.view(markers.size(0), -1), markers_gt.view(markers.size(0), -1))
        
        hand_mask = torch.ones((143*3)).cuda()
        hand_mask[64*3:79*3] = 1
        hand_mask[-22*3:] = 1
        loss_marker_rec = torch.einsum('ij,j->ij', loss_marker_rec, hand_mask)

        loss_marker_rec = weight_rec * marker_weight * torch.mean(torch.sum(loss_marker_rec, dim=1))

        loss_consistency_h2o = self.cfg.consistency_weight * weight_rec * torch.mean(torch.einsum('ij,ij->ij', torch.abs(h2o_signed - h2o_signed_gt), drec['contacts_markers'].squeeze()))
        loss_consistency_o2h = self.cfg.consistency_weight * weight_rec * torch.mean(torch.einsum('ij,ij->ij', torch.abs(o2h - o2h_gt), drec['contacts_object'].squeeze()))


        ################################## loss dict
        loss_dict = {'loss_kl': loss_kl,
                     'loss_object_contact_rec': loss_object_contact_rec,
                     'loss_markers_contact_rec': loss_markers_contact_rec,
                     'loss_marker_rec': loss_marker_rec,
                     'loss_consistency_o2h': loss_consistency_o2h,
                     'loss_consistency_h2o': loss_consistency_h2o,
                     }

        loss_total = torch.stack(list(loss_dict.values())).sum()
        loss_dict['loss_total'] = loss_total

        return loss_total, loss_dict

    def fit(self, n_epochs=None, message=None):

        starttime = datetime.now().replace(microsecond=0)
        if n_epochs is None:
            n_epochs = self.cfg.n_epochs

        self.logger('Started Training at %s for %d epochs' % (datetime.strftime(starttime, '%Y-%m-%d_%H:%M:%S'), n_epochs))
        if message is not None:
            self.logger(message)

        prev_lr_net = np.inf

        self.fit_net = True

        early_stopping_net = EarlyStopping(patience=8, trace_func=self.logger)

        for epoch_num in range(self.start_epoch, n_epochs + 1):
            self.logger('--- starting Epoch # %03d' % epoch_num)
            self.ROC_AUC_object.reset()
            self.ROC_AUC_marker.reset()

            # KL weight linear annealing
            if self.cfg.kl_annealing:
                self.cfg.kl_coef = min(0.5 * (epoch_num+1) / self.cfg.kl_annealing_epoch, 0.5)
                
            train_loss_dict_net = self.train()
            train_roc_auc_object = self.ROC_AUC_object.compute()
            train_roc_auc_markers = self.ROC_AUC_marker.compute()

            eval_loss_dict_net  = self.evaluate()
            eval_roc_auc_object = self.ROC_AUC_object.compute()
            eval_roc_auc_markers = self.ROC_AUC_marker.compute()
            eval_loss_dict_net  = train_loss_dict_net

            if self.cfg.kl_annealing:
                self.cfg.kl_coef = min(0.5 * (epoch_num+1) / self.cfg.kl_annealing_epoch, 0.5)
                print(self.cfg.kl_coef)


            if self.fit_net:

                self.lr_scheduler.step()
                cur_lr_net = self.optimizer_net.param_groups[0]['lr']

                if cur_lr_net != prev_lr_net:
                    self.logger('--- MarkerNet learning rate changed from %.2e to %.2e ---' % (prev_lr_net, cur_lr_net))
                    prev_lr_net = cur_lr_net

                with torch.no_grad():
                    eval_msg = Trainer.create_loss_message(eval_loss_dict_net, 
                                                            # expr_ID=self.cfg.expr_ID,
                                                            epoch_num=self.epoch_completed, it=len(self.ds_val),
                                                            model_name='MarkerNet',
                                                            try_num=self.try_num, mode='evald')

                    self.cfg.best_net = makepath(os.path.join(self.cfg.work_dir, 'snapshots', 'TR%02d_E%03d_net.pt' % (self.try_num, self.epoch_completed)), isfile=True)
                    if not epoch_num % 5:
                        self.save_net()
                    self.logger(eval_msg + ' ** ')
                    self.best_loss_net = eval_loss_dict_net['loss_total']

                    self.swriter.add_scalars('loss_net/kl_loss',
                                             {
                                             'train_loss_kl': train_loss_dict_net['loss_kl'],
                                             'evald_loss_kl': eval_loss_dict_net['loss_kl'],
                                            #  'train_loss_kl_normalized': train_loss_dict_net['loss_kl_normalized'],
                                            #  'evald_loss_kl_normalized': eval_loss_dict_net['loss_kl_normalized'],
                                             },
                                             self.epoch_completed)

                    self.swriter.add_scalars('loss_net/total_rec_loss',
                                             {
                                             'train_loss_total': train_loss_dict_net['loss_total'],
                                             'evald_loss_total': eval_loss_dict_net['loss_total'], 
                                             },
                                             self.epoch_completed)

                    self.swriter.add_scalars('loss_net/object_contact_rec_loss',
                                             {
                                             'train_loss_object_contact_rec': train_loss_dict_net['loss_object_contact_rec'],
                                             'evald_loss_object_contact_rec': eval_loss_dict_net['loss_object_contact_rec'],
                                             },
                                             self.epoch_completed)

                    self.swriter.add_scalars('loss_net/markers_contact_rec_loss',
                                             {
                                             'train_loss_object_markers_rec': train_loss_dict_net['loss_markers_contact_rec'],
                                             'evald_loss_object_markers_rec': eval_loss_dict_net['loss_markers_contact_rec'],
                                             },
                                             self.epoch_completed)

                    self.swriter.add_scalars('loss_net/marker_rec_loss',
                                             {
                                             'train_loss_marker_rec': train_loss_dict_net['loss_marker_rec'],
                                             'evald_loss_marker_rec': eval_loss_dict_net['loss_marker_rec'],
                                             },
                                             self.epoch_completed)
                    self.swriter.add_scalars('loss_net/consistency_o2h_loss',
                                             {
                                             'train_loss_consistency_o2h': train_loss_dict_net['loss_consistency_o2h'],
                                             'evalid_loss_consistency_o2h': eval_loss_dict_net['loss_consistency_o2h'],
                                             },
                                             self.epoch_completed)
                    self.swriter.add_scalars('loss_net/consistency_h2o_loss',
                                             {
                                             'train_loss_consistency_h2o': train_loss_dict_net['loss_consistency_h2o'],
                                             'evalid_loss_consistency_h2o': eval_loss_dict_net['loss_consistency_h2o'],
                                             },
                                             self.epoch_completed)
                    self.swriter.add_scalars('loss_net/AUC_object',
                                             {
                                             'train_roc_auc_object': train_roc_auc_object,
                                             'eval_roc_auc_object': eval_roc_auc_object,
                                             },
                                             self.epoch_completed)
                    self.swriter.add_scalars('loss_net/AUC_markers',
                                             {
                                             'train_roc_auc_markers': train_roc_auc_markers,
                                             'eval_roc_auc_markers': eval_roc_auc_markers,
                                             },
                                             self.epoch_completed)

                    self.logger('object train_auc: %f, object eval_auc: %f' % (train_roc_auc_object, eval_roc_auc_object))
                    self.logger('markers train_auc: %f, markers eval_auc: %f' % (train_roc_auc_markers, eval_roc_auc_markers))

                # if early_stopping_net(eval_loss_dict_net['loss_total']):
                #     self.fit_net = False
                #     self.logger('Early stopping MarkerNet training!')

            self.epoch_completed += 1

            checkpoint = {
                          'epoch': epoch_num + 1,
                          'state_dict': self.full_grasp_net.module.state_dict() if isinstance(self.full_grasp_net, torch.nn.DataParallel) else self.full_grasp_net.state_dict(),
                          'optimizer': self.optimizer_net.state_dict(),
                          'cfg': self.cfg,
                          }
            self.save_ckp(checkpoint, self.cfg.work_dir)

            if not self.fit_net:
                self.logger('Stopping the training!')
                break
                

        endtime = datetime.now().replace(microsecond=0)

        self.logger('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        self.logger(
            'Training done in %s!\n' % (endtime - starttime))
        self.logger('Best MarkerNet val total loss achieved: %.2e\n' % (self.best_loss_net))
        self.logger('Best MarkerNet model path: %s\n' % self.cfg.best_net)


    @staticmethod
    def create_loss_message(loss_dict, epoch_num=0,model_name='ContactNet', it=0, try_num=0, mode='evald'):
        ext_msg = ' | '.join(['%s = %.2e' % (k, v) for k, v in loss_dict.items() if k != 'loss_total'])
        return 'TR%02d_E%03d - It %05d - %s - %s: [T:%.2e] - [%s]' % (
            try_num, epoch_num, it,model_name, mode, loss_dict['loss_total'], ext_msg)
