import sys

sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from WholeGraspPose.models.pointnet import (PointNetFeaturePropagation,
                                            PointNetSetAbstraction)


class ResBlock(nn.Module):
    def __init__(self,
                 Fin,
                 Fout,
                 n_neurons=256):

        super(ResBlock, self).__init__()
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)

        self.fc2 = nn.Linear(n_neurons, Fout)
        self.bn2 = nn.BatchNorm1d(Fout)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        Xout = self.fc1(x)  # n_neurons
        Xout = self.bn1(Xout)
        Xout = self.ll(Xout)

        Xout = self.fc2(Xout)
        Xout = self.bn2(Xout)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)
        return Xout


class PointNetEncoder(nn.Module):

    def __init__(self,
                 hc,
                 in_feature):

        super(PointNetEncoder, self).__init__()
        self.hc = hc
        self.in_feature = in_feature

        self.enc_sa1 = PointNetSetAbstraction(npoint=256, radius=0.2, nsample=32, in_channel=self.in_feature, mlp=[self.hc, self.hc*2], group_all=False)
        self.enc_sa2 = PointNetSetAbstraction(npoint=128, radius=0.25, nsample=64, in_channel=self.hc*2 + 3, mlp=[self.hc*2, self.hc*4], group_all=False)
        self.enc_sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=self.hc*4 + 3, mlp=[self.hc*4, self.hc*8], group_all=True)

    def forward(self, l0_xyz, l0_points):

        l1_xyz, l1_points = self.enc_sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.enc_sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.enc_sa3(l2_xyz, l2_points)
        x = l3_points.view(-1, self.hc*8)
        
        return l1_xyz, l1_points, l2_xyz, l2_points, l3_xyz, x

class MarkerNet(nn.Module):
    def __init__(self, cfg, n_neurons=1024, in_cond=1024, latentD=16, in_feature=143*3, **kwargs):
        super(MarkerNet, self).__init__()

        self.cfg = cfg
        ## condition features
        self.obj_cond_feature = 1 if cfg.cond_object_height else 0

        self.enc_bn1 = nn.BatchNorm1d(in_feature + self.obj_cond_feature)
        self.enc_rb1 = ResBlock(in_cond + in_feature + int(in_feature/3) + self.obj_cond_feature, n_neurons)
        self.enc_rb2 = ResBlock(n_neurons + in_cond + in_feature + int(in_feature/3) + self.obj_cond_feature, n_neurons)

        self.dec_rb1 = ResBlock(latentD + in_cond, n_neurons)
        self.dec_rb2_xyz = ResBlock(n_neurons + latentD + in_cond + self.obj_cond_feature, n_neurons)
        self.dec_rb2_p = ResBlock(n_neurons + latentD + in_cond, n_neurons)

        self.dec_output_xyz = nn.Linear(n_neurons, 143*3)
        self.dec_output_p = nn.Linear(n_neurons, 143)
        self.p_output = nn.Sigmoid()

    def enc(self, cond_object, markers, contacts_markers, transf_transl):
        _, _, _, _, _, object_cond = cond_object

        X = markers.view(markers.shape[0], -1)

        if self.obj_cond_feature == 1:
            X = torch.cat([X, transf_transl[:, -1, None]], dim=-1).float()

        X0 = self.enc_bn1(X)

        X0 = torch.cat([X0, contacts_markers.view(-1, 143), object_cond], dim=-1)


        X = self.enc_rb1(X0, True)
        X = self.enc_rb2(torch.cat([X0, X], dim=1), True)

        return X

    def dec(self, Z, cond_object, transf_transl):

        _, _, _, _, _, object_cond = cond_object
        X0 = torch.cat([Z, object_cond], dim=1).float()

        X = self.dec_rb1(X0, True)
        X_xyz = self.dec_rb2_xyz(torch.cat([X0, X, transf_transl[:, -1, None]], dim=1).float(), True)
        X_p = self.dec_rb2_p(torch.cat([X0, X], dim=1).float(), True)

        xyz_pred = self.dec_output_xyz(X_xyz)
        p_pred = self.p_output(self.dec_output_p(X_p))

        return xyz_pred, p_pred

class ContactNet(nn.Module):
    def __init__(self, cfg, latentD=16, hc=64, object_feature=6, **kwargs):
        super(ContactNet, self).__init__()
        self.latentD = latentD
        self.hc = hc
        self.object_feature  = object_feature

        self.enc_pointnet = PointNetEncoder(self.hc, self.object_feature+1)

        self.dec_fc1 = nn.Linear(self.latentD, self.hc*2)
        self.dec_bn1 = nn.BatchNorm1d(self.hc*2)
        self.dec_drop1 = nn.Dropout(0.1)
        self.dec_fc2 = nn.Linear(self.hc*2, self.hc*4)
        self.dec_bn2 = nn.BatchNorm1d(self.hc*4)
        self.dec_drop2 = nn.Dropout(0.1)
        self.dec_fc3 = nn.Linear(self.hc*4, self.hc*8)
        self.dec_bn3 = nn.BatchNorm1d(self.hc*8)
        self.dec_drop3 = nn.Dropout(0.1)

        self.dec_fc4 = nn.Linear(self.hc*8+self.latentD, self.hc*8)
        self.dec_bn4 = nn.BatchNorm1d(self.hc*8)
        self.dec_drop4 = nn.Dropout(0.1)

        self.dec_fp3 = PointNetFeaturePropagation(in_channel=self.hc*8+self.hc*4, mlp=[self.hc*8, self.hc*4])
        self.dec_fp2 = PointNetFeaturePropagation(in_channel=self.hc*4+self.hc*2, mlp=[self.hc*4, self.hc*2])
        self.dec_fp1 = PointNetFeaturePropagation(in_channel=self.hc*2+self.object_feature, mlp=[self.hc*2, self.hc*2])

        self.dec_conv1 = nn.Conv1d(self.hc*2, self.hc*2, 1)
        self.dec_conv_bn1 = nn.BatchNorm1d(self.hc*2)
        self.dec_conv_drop1 = nn.Dropout(0.1)
        self.dec_conv2 = nn.Conv1d(self.hc*2, 1, 1)

        self.dec_output = nn.Sigmoid()

    def enc(self, contacts_object, verts_object, feat_object):
        l0_xyz = verts_object[:, :3, :]
        l0_points = torch.cat([feat_object, contacts_object], 1) if feat_object is not None else contacts_object
        _, _, _, _, _, x = self.enc_pointnet(l0_xyz, l0_points)

        return x  

    def dec(self, z, cond_object, verts_object, feat_object):
        l0_xyz = verts_object[:, :3, :]
        l0_points = feat_object

        l1_xyz, l1_points, l2_xyz, l2_points, l3_xyz, l3_points = cond_object

        l3_points = torch.cat([l3_points, z], 1)
        l3_points = self.dec_drop4(F.relu(self.dec_bn4(self.dec_fc4(l3_points)), inplace=True))
        l3_points = l3_points.view(l3_points.size()[0], l3_points.size()[1], 1)

        l2_points = self.dec_fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.dec_fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        if l0_points is None:
            l0_points = self.dec_fp1(l0_xyz, l1_xyz, l0_xyz, l1_points)
        else:
            l0_points = self.dec_fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz,l0_points],1), l1_points)
        feat =  F.relu(self.dec_conv_bn1(self.dec_conv1(l0_points)), inplace=True)
        x = self.dec_conv_drop1(feat)
        x = self.dec_conv2(x)
        x = self.dec_output(x)

        return x


class FullBodyGraspNet(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(FullBodyGraspNet, self).__init__()

        self.cfg = cfg
        self.latentD = cfg.latentD
        self.in_feature_list = {}
        self.in_feature_list['joints'] = 127*3
        self.in_feature_list['markers_143'] = 143*3
        self.in_feature_list['markers_214'] = 214*3
        self.in_feature_list['markers_593'] = 593*3

        self.in_feature = self.in_feature_list[cfg.data_representation]

        self.marker_net = MarkerNet(cfg, n_neurons=cfg.n_markers, in_cond=cfg.pointnet_hc*8, latentD=cfg.latentD, in_feature=self.in_feature)
        self.contact_net = ContactNet(cfg, latentD=cfg.latentD, hc=cfg.pointnet_hc, object_feature=cfg.obj_feature)

        self.pointnet = PointNetEncoder(hc=cfg.pointnet_hc, in_feature=cfg.obj_feature)
        # encoder fusion
        self.enc_fusion = ResBlock(cfg.n_markers+self.cfg.pointnet_hc*8, cfg.n_neurons)

        self.enc_mu = nn.Linear(cfg.n_neurons, cfg.latentD)
        self.enc_var = nn.Linear(cfg.n_neurons, cfg.latentD)


    def encode(self, object_cond, verts_object, feat_object, contacts_object, markers, contacts_markers, transf_transl):
        # marker branch
        marker_feat = self.marker_net.enc(object_cond, markers, contacts_markers, transf_transl)   # [B, n_neurons=1024]

        # contact branch
        contact_feat = self.contact_net.enc(contacts_object, verts_object, feat_object)  # [B, hc*8]

        # fusion
        X = torch.cat([marker_feat, contact_feat], dim=-1)
        X = self.enc_fusion(X, True)

        return torch.distributions.normal.Normal(self.enc_mu(X), F.softplus(self.enc_var(X)))


    def decode(self, Z, object_cond, verts_object, feat_object, transf_transl):

        bs = Z.shape[0]
        # marker_branch
        markers_xyz_pred, markers_p_pred = self.marker_net.dec(Z, object_cond, transf_transl)

        # contact branch
        contact_pred = self.contact_net.dec(Z, object_cond, verts_object, feat_object)

        return markers_xyz_pred.view(bs, -1, 3), markers_p_pred, contact_pred

    def forward(self, verts_object, feat_object, contacts_object, markers, contacts_markers, transf_transl, **kwargs):
        object_cond = self.pointnet(l0_xyz=verts_object, l0_points=feat_object)
        z = self.encode(object_cond, verts_object, feat_object, contacts_object, markers, contacts_markers, transf_transl)
        z_s = z.rsample()

        markers_xyz_pred, markers_p_pred, object_p_pred = self.decode(z_s, object_cond, verts_object, feat_object, transf_transl)

        results = {'markers': markers_xyz_pred, 'contacts_markers': markers_p_pred, 'contacts_object': object_p_pred, 'object_code': object_cond[-1], 'mean': z.mean, 'std': z.scale}

        return results


    def sample(self, verts_object, feat_object, transf_transl, seed=None):
        bs = verts_object.shape[0]
        if seed is not None:
            np.random.seed(seed)
        dtype = verts_object.dtype
        device = verts_object.device
        self.eval()
        with torch.no_grad():
            Zgen = np.random.normal(0., 1., size=(bs, self.latentD))
            Zgen = torch.tensor(Zgen,dtype=dtype).to(device)

        object_cond = self.pointnet(l0_xyz=verts_object, l0_points=feat_object)

        return self.decode(Zgen, object_cond, verts_object, feat_object, transf_transl)
