import copy
import json
import os
import pdb
import pickle
import sys

import numpy as np
import open3d as o3d
import smplx
import torch
import torch.nn.functional as F
import torch.optim as optim
from human_body_prior.tools.model_loader import load_vposer
from torch.autograd import Variable
from tqdm import tqdm
from utils.como.como_smooth import Enc
from utils.como.como_utils import (convert_to_3D_rot, convert_to_6D_all,
                                   gen_body_joints_v1)
from utils.train_helper import point2point_signed
from utils.utils_body import (gen_body_mesh_v1, get_body_model,
                              get_markers_ids_indiv)


class FittingOP:
    def __init__(self, fittingconfig):

        for key, val in fittingconfig.items():
            setattr(self, key, val)

        self.smplx_model = get_body_model('smplx', self.gender, 1, self.device)
        self.smplx_model_batch = get_body_model('smplx', self.gender, self.T, self.device)

        self.vposer_model, _ = load_vposer('./body_utils/body_models/vposer_v1_0', vp_model='snapshot')
        self.vposer_model.to(self.device)
        self.vposer_model.eval()

        self.init_param_t()

    def init_param_t(self):
        self.transl_opt_t = torch.zeros(1, 3).to(self.device)
        self.transl_opt_t[:, 1] = 0.4
        self.transl_opt_t[:, 2] = 1.0

        rot_opt_t = torch.zeros(1, 3).to(self.device)
        rot_opt_t[:, 1] = 1.6
        rot_opt_t[:, 2] = 3.14
        self.rot_6d_opt_t = convert_to_6D_all(rot_opt_t)

        self.other_params_opt_t = torch.zeros(1, 56+24).to(self.device) # other params except transl/rot/shape
        self.shape_t = torch.tensor(self.smplx_beta).to(self.device).view(1, -1)  # [1, 10]

        self.transl_opt_t = Variable(self.transl_opt_t, requires_grad=True)
        self.rot_6d_opt_t = Variable(self.rot_6d_opt_t, requires_grad=True)
        self.other_params_opt_t = Variable(self.other_params_opt_t, requires_grad=True)

    def init_param_T(self, transl_opt_T, rot_6d_opt_T, other_params_opt_T, shape_T):
        self.transl_opt_T = Variable(transl_opt_T.to(self.device), requires_grad=True)
        self.rot_6d_opt_T = Variable(rot_6d_opt_T.to(self.device), requires_grad=True)
        self.other_params_opt_T = Variable(other_params_opt_T.to(self.device), requires_grad=True)
        self.shape_T = torch.tensor(shape_T).to(self.device)  # [1, 10]


    def markers_fitting_loss(self, markers_opt, markers_rec, mask=None):
        if mask is not None:
            return F.l1_loss(markers_opt*mask, markers_rec*mask)
        else:
            return F.l1_loss(markers_opt, markers_rec)

    def body_smooth_loss(self, body_verts_opt_T, body_params_opt_T_72):
        with open('./utils/como/my_SSM2.json') as f:  # todo
            smooth_marker_ids = list(
                json.load(f)['markersets'][0]['indices'].values())

        preprocess_stats_smooth_xmean = np.load(
        './utils/como/preprocess_stats_global_markers/Xmean.npy')
        preprocess_stats_smooth_xstd = np.load(
            './utils/como/preprocess_stats_global_markers/Xstd.npy')
        Xmean_global_markers = torch.from_numpy(
            preprocess_stats_smooth_xmean).float().to(self.device)
        Xstd_global_markers = torch.from_numpy(
            preprocess_stats_smooth_xstd).float().to(self.device)

        smooth_encoder = Enc(downsample=False, z_channel=64).to(self.device)
        smooth_weights = torch.load(
            './utils/como/como_smooth_model.pkl', map_location=lambda storage, loc: storage)
        smooth_encoder.load_state_dict(smooth_weights)
        smooth_encoder.eval()
        for param in smooth_encoder.parameters():
            param.requires_grad = False

        # smooth prios, global markers
        joints_3d = gen_body_joints_v1(body_params=body_params_opt_T_72, smplx_model=self.smplx_model_batch,
                                       vposer_model=self.vposer_model)  # [T, 25, 3]
        # [T, 67+..., 3]
        markers_smooth = body_verts_opt_T[:, smooth_marker_ids, :]
        # transfrom to pelvis at origin, face y axis
        # [25, 3], joints of first frame
        joints_frame0 = joints_3d[0].detach()
        x_axis = joints_frame0[2, :] - joints_frame0[1, :]  # [3]
        x_axis[-1] = 0
        x_axis = x_axis / torch.norm(x_axis)
        z_axis = torch.tensor([0, 0, 1]).float().to(self.device)
        y_axis = torch.cross(z_axis, x_axis)
        y_axis = y_axis / torch.norm(y_axis)
        transf_rotmat = torch.stack(
            [x_axis, y_axis, z_axis], dim=1)  # [3, 3]
        markers_frame0 = markers_smooth[0].detach()
        global_markers_smooth_opt = torch.matmul(
            markers_smooth - markers_frame0[0], transf_rotmat)  # [T(/bs), 67, 3]

        clip_img_smooth = global_markers_smooth_opt.reshape(
            global_markers_smooth_opt.shape[0], -1).unsqueeze(0)  # [1, T, d]
        clip_img_smooth = (clip_img_smooth -
                           Xmean_global_markers) / Xstd_global_markers
        clip_img_smooth = clip_img_smooth.permute(
            0, 2, 1).unsqueeze(1)  # [1, 1, d, T]

        # input res, encoder forward
        clip_img_smooth_v = clip_img_smooth[:, :,
                                            :, 1:] - clip_img_smooth[:, :, :, 0:-1]
        # input padding
        p2d = (8, 8, 1, 1)
        clip_img_smooth_v = F.pad(clip_img_smooth_v, p2d, 'reflect')
        # forward
        motion_z, _, _, _, _, _ = smooth_encoder(clip_img_smooth_v)
        motion_z_v = motion_z[:, :, :, 1:] - motion_z[:, :, :, 0:-1]
        loss_smooth = torch.mean(motion_z_v ** 2)

        return loss_smooth

    def contact_loss(self, start_t, body_verts_opt_T, body_markers_opt_T, rh_markers_opt_T, contact_object, contact_markers, normal_object, verts_object, rhand_verts):
        markers_rh_palm = get_markers_ids_indiv('palm')[22:]
        rh_normals = []
        rh_normals_full_verts = []
        verts_full_new = body_verts_opt_T.detach().cpu().numpy()
        for i in range(start_t, len(verts_full_new)):
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts_full_new[i])
            mesh.triangles = o3d.utility.Vector3iVector(self.smplx_model_batch.faces)
            mesh.compute_vertex_normals()
            normals = np.asarray(mesh.vertex_normals)
            rh_normals_full_verts.append(normals[rhand_verts, :])
            rh_normals.append(normals[markers_rh_palm, :])

        rh_normals_full_verts = torch.from_numpy(np.array(rh_normals_full_verts)).to(torch.float32).view(-1, len(rhand_verts), 3).cuda()  # [T, 22, 3]
        self.rh_normals = torch.from_numpy(np.array(rh_normals)).to(torch.float32).view(-1, 22, 3).cuda()  # [T, 22, 3]
        self.rh_normals_norm = torch.norm(self.rh_normals, dim=-1)
        # o2h: [T-start_t, 2048]
        # h2o_signed: [T-start_t, 778]
        # h2o_signed_marker: [T-start_t, 143]
        # o2h_idx: [T-start_t, 2048]
        # h2o_idx: [T-start_t, 778]
        o2h_marker, self.h2o_signed_marker, o2h_idx_marker, h2o_idx_marker, o2h_marker_vector, self.h2o_marker_vector = point2point_signed(
            body_markers_opt_T, verts_object.float(), None, normal_object.float(), return_vector=True)
        self.o2h_signed, self.h2o_signed, o2h_idx, h2o_idx = point2point_signed(
            rh_markers_opt_T, verts_object.float(), rh_normals_full_verts, normal_object.float())
        h2o_signed_min = torch.min(self.h2o_signed, dim=-1)[0]

        attract_start_t = -5
        attraction_weight_only_last_five_frames = torch.zeros((61-start_t, 1)).cuda()
        attraction_weight_only_last_five_frames[attract_start_t:] = 1

        loss_attraction_object_BT = torch.abs(self.o2h_signed) * contact_object * attraction_weight_only_last_five_frames
        loss_attraction_markers_BT = torch.abs(
            self.h2o_signed_marker) * contact_markers * attraction_weight_only_last_five_frames

        loss_contact_object = torch.mean(loss_attraction_object_BT)
        loss_contact_markers = torch.mean(loss_attraction_markers_BT)

        loss_contact = loss_contact_object + loss_contact_markers

        return loss_contact

    def collision_loss(self, ):
        mask_pulse_h2o = torch.zeros_like(self.h2o_signed).cuda()
        h2o_dist_neg = self.h2o_signed < -0.001
        mask_pulse_h2o[h2o_dist_neg] = 10 # more weight for penetration

        mask_pulse_o2h = torch.zeros_like(self.o2h_signed).cuda()
        o2h_dist_neg = self.o2h_signed < -0.003
        mask_pulse_o2h[o2h_dist_neg] = 20 # more weight for penetration

        loss_collision_o2h = torch.abs(self.o2h_signed) * mask_pulse_o2h
        loss_collision = 5 * torch.mean(torch.mean(loss_collision_o2h, dim=-1))
        
        return loss_collision 

    def skating_loss(self, body_verts_opt_T, contact_lbl_rec):
        body_segments_dir = './body_utils/body_segments'
        with open(os.path.join(body_segments_dir, 'L_Leg.json'), 'r') as f:
            data = json.load(f)
            left_foot_verts_id = np.asarray(list(set(data["verts_ind"])))
        left_heel_verts_id = np.load(
            './body_utils/left_heel_verts_id.npy')
        left_toe_verts_id = np.load(
            './body_utils/left_toe_verts_id.npy')
        left_heel_verts_id = left_foot_verts_id[left_heel_verts_id]
        left_toe_verts_id = left_foot_verts_id[left_toe_verts_id]
        left_whole_foot_verts_id = np.load(
            './body_utils/left_whole_foot_verts_id.npy')
        right_whole_foot_verts_id = np.load(
            './body_utils/right_whole_foot_verts_id.npy')

        with open(os.path.join(body_segments_dir, 'R_Leg.json'), 'r') as f:
            data = json.load(f)
            right_foot_verts_id = np.asarray(list(set(data["verts_ind"])))
        right_heel_verts_id = np.load(
            './body_utils/right_heel_verts_id.npy')
        right_toe_verts_id = np.load(
            './body_utils/right_toe_verts_id.npy')
        right_heel_verts_id = right_foot_verts_id[right_heel_verts_id]
        right_toe_verts_id = right_foot_verts_id[right_toe_verts_id]

        left_heel_contact = contact_lbl_rec[:, 0]  # [T]
        right_heel_contact = contact_lbl_rec[:, 1]
        left_toe_contact = contact_lbl_rec[:, 2]
        right_toe_contact = contact_lbl_rec[:, 3]

        body_verts_opt_vel = (
            body_verts_opt_T[1:] - body_verts_opt_T[0:-1]) * 30  # [T-1, 10475, 3]
        left_heel_verts_vel = body_verts_opt_vel[:,
                                                    left_heel_verts_id, :][left_heel_contact[0:-1] == 1]
        left_heel_verts_vel = torch.norm(
            left_heel_verts_vel, dim=-1)  # [t, n]

        right_heel_verts_vel = body_verts_opt_vel[:,
                                                    right_heel_verts_id, :][right_heel_contact[0:-1] == 1]
        right_heel_verts_vel = torch.norm(right_heel_verts_vel, dim=-1)

        left_toe_verts_vel = body_verts_opt_vel[:,
                                                left_toe_verts_id, :][left_toe_contact[0:-1] == 1]
        left_toe_verts_vel = torch.norm(left_toe_verts_vel, dim=-1)

        right_toe_verts_vel = body_verts_opt_vel[:,
                                                    right_toe_verts_id, :][right_toe_contact[0:-1] == 1]
        right_toe_verts_vel = torch.norm(right_toe_verts_vel, dim=-1)

        vel_thres = 0.1  # todo: to set
        loss_contact_vel_left_heel = torch.tensor(0.0).to(self.device)
        if (left_heel_verts_vel - vel_thres).gt(0).sum().item() >= 1:
            loss_contact_vel_left_heel = left_heel_verts_vel[left_heel_verts_vel > vel_thres].abs(
            ).mean()

        loss_contact_vel_right_heel = torch.tensor(0.0).to(self.device)
        if (right_heel_verts_vel - vel_thres).gt(0).sum().item() >= 1:
            loss_contact_vel_right_heel = right_heel_verts_vel[right_heel_verts_vel > vel_thres].abs(
            ).mean()

        loss_contact_vel_left_toe = torch.tensor(0.0).to(self.device)
        if (left_toe_verts_vel - vel_thres).gt(0).sum().item() >= 1:
            loss_contact_vel_left_toe = left_toe_verts_vel[left_toe_verts_vel > vel_thres].abs(
            ).mean()

        loss_contact_vel_right_toe = torch.tensor(0.0).to(self.device)
        if (right_toe_verts_vel - vel_thres).gt(0).sum().item() >= 1:
            loss_contact_vel_right_toe = right_toe_verts_vel[right_toe_verts_vel > vel_thres].abs(
            ).mean()

        loss_contact_vel = loss_contact_vel_left_heel + loss_contact_vel_right_heel + \
            loss_contact_vel_left_toe + loss_contact_vel_right_toe

        return loss_contact_vel

    def angle_loss(self, start_t):
        rh_h2o_marker_vector = -self.h2o_marker_vector[:, -22:]   # [T, N, 3] the vector from object to markers  (for each marker)
        rh_h2o_marker_vector_norm = torch.norm(rh_h2o_marker_vector, dim=-1)

        cosine_theta = torch.bmm(self.rh_normals.reshape(-1, 3).unsqueeze(1), rh_h2o_marker_vector.reshape(-1, 3).unsqueeze(2)).view(self.T_opt_colli, -1)/(self.rh_normals_norm * rh_h2o_marker_vector_norm)
        sign_cosine_theta = cosine_theta.sign()

        ### angle loss
        angle_weight = (1 - torch.true_divide(torch.arange(
            61-start_t)+1, 61-start_t).view(-1, 1) ** 2).cuda()

        mask_angle = (self.h2o_signed_marker < 0.01).int().cuda()

        loss_rhand_angle_BT = mask_angle[:, -22:] * ((cosine_theta - 1)**2) * angle_weight   # [T, 22] * [1, 22]
        loss_rhand_angle = torch.mean(loss_rhand_angle_BT)
        return loss_rhand_angle


    def fitting_stage1(self, body_markers_rec):  # body_markers_rec: tensor
        print('Stage 1 optimization: minimize E_fit...')
        self.optimizer_s1 = optim.Adam([self.transl_opt_t, self.rot_6d_opt_t, self.other_params_opt_t], lr=self.init_lr_stage1)

        transl_opt_T = []
        rot_6d_opt_T = []
        other_params_opt_T = []
        shape_T = []
        markers_rec_T = []
        for t in tqdm(range(self.T)):
            markers_rec_t = body_markers_rec[t:t+1].to(self.device).float()

            # learning rate scheduling
            if t > 0:
                for param_group in self.optimizer_s1.param_groups:
                    param_group['lr'] = 0.005 * 1
            for step in range(self.num_iter[0]):
                if step > 60:
                    for param_group in self.optimizer_s1.param_groups:
                        param_group['lr'] = 0.01
                if step > 80:
                    for param_group in self.optimizer_s1.param_groups:
                        param_group['lr'] = 0.003
                self.optimizer_s1.zero_grad()

                body_params_opt_t = torch.cat([self.transl_opt_t, self.rot_6d_opt_t, self.shape_t, self.other_params_opt_t], dim=-1)  # [1, 75]
                body_params_opt_t_72 = convert_to_3D_rot(body_params_opt_t)  # tensor, [bs=1, 72]
                body_verts_opt_t, _ = gen_body_mesh_v1(body_params=body_params_opt_t_72, smplx_model=self.smplx_model,
                                        vposer_model=self.vposer_model)  # tensor [1, 10475, 3]
                markers_opt_t = body_verts_opt_t[:, self.markers_ids, :]  # [1, 67, 3]

                loss_marker = self.markers_fitting_loss(markers_opt_t, markers_rec_t)
                loss_vposer = torch.mean(body_params_opt_t_72[:, 16:48] ** 2)
                loss_shape = torch.mean(body_params_opt_t_72[:, 6:16] ** 2)
                loss_hand = torch.mean(body_params_opt_t_72[:, 48:] ** 2)

                loss = self.stage1_weight_loss_rec_markers * loss_marker + \
                self.stage1_weight_loss_vposer * loss_vposer + \
                self.stage1_weight_loss_shape * loss_shape + self.stage1_weight_loss_hand * loss_hand


                loss.backward(retain_graph=True)

                self.optimizer_s1.step()

            transl_opt_T.append(self.transl_opt_t.clone().detach())
            rot_6d_opt_T.append(self.rot_6d_opt_t.clone().detach())
            shape_T.append(self.shape_t.clone().detach())
            other_params_opt_T.append(self.other_params_opt_t.clone().detach())
            markers_rec_T.append(markers_rec_t.clone().detach())

        transl_opt_T = torch.stack(transl_opt_T).squeeze(1).detach()
        rot_6d_opt_T = torch.stack(rot_6d_opt_T).squeeze(1).detach()
        shape_T = torch.stack(shape_T).squeeze(1).detach()
        other_params_opt_T = torch.stack(other_params_opt_T).squeeze(1).detach()
        markers_rec_T = torch.stack(markers_rec_T).squeeze(1).detach()

        return transl_opt_T, rot_6d_opt_T, shape_T, other_params_opt_T, markers_rec_T

    def fitting_stage2(self, markers_rec_T, markers_end_new, rhand_verts, contact_lbl_rec,
                        contact_object, contact_markers, normal_object, verts_object):
        print('Stage 2 optimization...')
        
        start_t = 30   

        self.optimizer_s2 = optim.Adam([self.transl_opt_T, self.rot_6d_opt_T, self.other_params_opt_T], lr=self.init_lr_stage2)
        
        for step in tqdm(range(self.num_iter[1])):
            if step > 150:
                for param_group in self.optimizer_s2.param_groups:
                    param_group['lr'] = 0.005
            self.optimizer_s2.zero_grad()

            body_params_opt_T = torch.cat(
                [self.transl_opt_T, self.rot_6d_opt_T, self.shape_T, self.other_params_opt_T], dim=-1)  # [T, 75]
            body_params_opt_T_72 = convert_to_3D_rot(
                body_params_opt_T)  # tensor, [T, 72]
            body_verts_opt_T, _ = gen_body_mesh_v1(body_params=body_params_opt_T_72, smplx_model=self.smplx_model_batch,
                                                vposer_model=self.vposer_model)  # tensor [T, 10475, 3]
            markers_opt_T = body_verts_opt_T[:, self.markers_ids, :]  # [T, 67, 3]

            rh_markers_opt_T = body_verts_opt_T[start_t:, rhand_verts, :]
            # [T, N, 3]
            body_markers_opt_T = body_verts_opt_T[start_t:, self.markers_ids_143, :]
            self.T_opt_colli = len(body_markers_opt_T)

            # loss_smooth = self.smooth_loss(body_verts_opt_T, body_params_opt_T_72)
            
            markers_mask = torch.ones(1, markers_opt_T.shape[1], 1).to(self.device)
            markers_mask[:, 49:49+6] = 0    # hand
            markers_mask[:, 33:33+6] = 0    # forearm
            markers_mask[:, 69:] = 0        # finger
            loss_marker = self.markers_fitting_loss(markers_opt_T, markers_rec_T, markers_mask)

            loss_last_frame_fit = self.markers_fitting_loss(markers_opt_T[-1], markers_end_new)

            loss_skating = self.skating_loss(body_verts_opt_T, contact_lbl_rec)
            loss_contact = self.contact_loss(start_t, body_verts_opt_T, body_markers_opt_T, rh_markers_opt_T, contact_object, contact_markers, normal_object, verts_object, rhand_verts)

            loss_body_smooth = self.body_smooth_loss(body_verts_opt_T, body_params_opt_T_72)

            rh_markers_v = rh_markers_opt_T[1:] - rh_markers_opt_T[:-1]
            loss_rh_smooth = torch.mean(torch.norm(rh_markers_v, dim=-1))


            loss_collision = self.collision_loss()

            loss_hand_angle = self.angle_loss(start_t)

            vposer_pose = body_params_opt_T_72[:, 16:48]
            loss_vposer = torch.mean(vposer_pose ** 2)
            hand_params = body_params_opt_T_72[:, 48:]
            loss_hand = torch.mean(hand_params ** 2)

            if not (step+1) % 20:
                print(loss_skating)

            loss2 = self.stage2_weight_loss_rec_markers * loss_marker + \
                    self.stage2_weight_loss_end_markers_fit * loss_last_frame_fit + \
                    self.stage2_weight_loss_vposer * loss_vposer + \
                    self.stage2_weight_loss_hand * loss_hand + \
                    self.stage2_weight_loss_skating * loss_skating + \
                    self.stage2_weight_loss_smooth * loss_body_smooth + \
                    self.stage2_weight_loss_hand_smooth * loss_rh_smooth + \
                    self.stage2_weight_loss_collision * loss_collision + \
                    self.stage2_weight_loss_contact * loss_contact + \
                    self.stage2_weight_loss_hand_angle * loss_hand_angle

            loss2.backward(retain_graph=True)
            self.optimizer_s2.step()

        return self.transl_opt_T, self.rot_6d_opt_T, self.shape_T, self.other_params_opt_T


    def fitting(self, body_markers_rec, markers_end_new, rhand_verts, contact_lbl_rec,
                        contact_object, contact_markers, normal_object, verts_object):
        transl_opt_T, rot_6d_opt_T, shape_T, other_params_opt_T, markers_rec_T = self.fitting_stage1(body_markers_rec)
        print('stage1 done!')

        self.init_param_T(transl_opt_T.detach(), rot_6d_opt_T.detach(), other_params_opt_T.detach(), shape_T.detach())

        transl_opt_T_final, rot_6d_opt_T_final, shape_T_final, other_params_opt_T_final = self.fitting_stage2(markers_rec_T, markers_end_new, rhand_verts, contact_lbl_rec,
                        contact_object, contact_markers, normal_object, verts_object)
        print('stage2 done!')

        return transl_opt_T, rot_6d_opt_T, shape_T, other_params_opt_T, transl_opt_T_final, rot_6d_opt_T_final, shape_T_final, other_params_opt_T_final




        

