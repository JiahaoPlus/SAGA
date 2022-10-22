import argparse
import os
import sys

import numpy as np
import scipy.ndimage.filters as filters
import torch
import torch.nn.functional as F
import torch.optim as optim
from human_body_prior.tools.model_loader import load_vposer
from smplx.lbs import batch_rodrigues
from tqdm import tqdm

from MotionFill.models.fittingop import FittingOP
from MotionFill.models.LocalMotionFill import Motion_CNN_CVAE
from MotionFill.models.TrajFill import Traj_MLP_CVAE
from utils.como.como_utils import *  # ## to integrate utils file
from utils.utils_body import (gen_body_mesh_v1, get_body_mesh, get_body_model,
                              get_global_pose, get_markers_ids,
                              get_object_mesh)

"""
	FullGraspMotion Pipeline:
		- Generate ending pose (either load from saved results or load saved model and implement the optimization)
		- Set initial frames (in markers)
		- Generate trajectories (in position) -> maybe we need some post-optimization here
		- Feed into the Motion-CVAE network
"""

def load_ending_pose(args, grasppose_result_path):
    end_data = np.load(grasppose_result_path, allow_pickle=True)

    sample_index = np.arange(0, len(end_data['markers']))   # ignore

    end_body_full = end_data['body'][()]
    end_body = {}
    for k in end_body_full:
        end_body[k] = end_body_full[k][sample_index]

    # object data
    object_transl_0 = torch.tensor(end_data['object'][()]['transl'])[sample_index].to(device)
    object_global_orient_0 = torch.tensor(end_data['object'][()]['global_orient'])[sample_index].to(device)
    object_global_orient_0 = batch_rodrigues(object_global_orient_0.view(-1, 3)).view([len(object_global_orient_0), 3, 3])#.detach().cpu().numpy()

    # get ending body (optional) mesh and markers/joints
    smplx_beta = end_body['betas']
    start_idx = 0
    bs = len(smplx_beta)
    end_body_mesh, end_smplx_results = get_body_mesh(end_body, args.gender, start_idx, bs)
    marker_end = end_smplx_results.vertices.detach().cpu().numpy()[:, markers_ids, :]
    joint_end = end_smplx_results.joints.detach().cpu().numpy()

    return end_data, end_body, marker_end, joint_end, object_transl_0, object_global_orient_0

def set_initial_pose(args, end_smplx, markers_ids):
    betas = end_smplx['betas']
    n = betas.shape[0]

    ### can be customized
    initial_orient = np.array([[1.5421, -0.00219, -0.0171]])   # Set initial body pose orientation / None (same orientation as the ending pose)
    initial_pose = np.array([-0.10901122,  0.0461413,   0.02993835,  0.11612727, -0.06200547,  0.08139142,
                                -0.02208922,  0.06683847, -0.02794579,  0.45293584, -0.16446967, -0.06646398,
                                0.07430738,  0.16469607,  0.05346995,  0.23588121, -0.09054547,  0.06633219,
                                -0.08885075,  0.25389493, -0.04105648, -0.1263972,  -0.2095012,  -0.01349497,
                                -0.1308483,   0.00866051, -0.00762679, -0.20351738, -0.0055567,   0.09453899,
                                0.09627768,  0.10411494,  0.03997851,  0.07713828, -0.01521101, -0.04545524,
                                0.10470242, -0.09646956, -0.40639114,  0.11441539,  0.09596836,  0.3891292,
                                0.1657324,   0.12639643,  0.01392403,  0.0669774,  -0.25228527, -0.69750136,
                                -0.01904383,  0.1466294,   0.6928179,   0.00282627,  0.00742727, -0.11434615,
                                -0.08387394, -0.05599072,  0.0974379,   0.00966642, -0.03484239,  0.10031673,
                                0.04399946,  0.04642308, -0.10101389]).reshape(-1, 63)
    rand_x = np.random.rand(n).reshape(-1, 1) * 0.04 - 0.02
    rand_y = np.random.rand(n).reshape(-1, 1) + 0.05

    end_transl = end_smplx['transl']
    end_global_orient = end_smplx['global_orient']

    rand_z = np.zeros(n).reshape(-1, 1)
    rand_displacement = np.concatenate([rand_x, rand_y, rand_z], axis=-1).reshape(n, -1)

    start_smplx = {}
    start_smplx['betas'] = betas
    start_smplx['transl'] = end_transl + rand_displacement
    start_smplx['global_orient'] = end_global_orient if initial_orient is None else initial_orient.repeat(n, axis=0).astype(np.float32)

    if initial_pose is not None:
        start_smplx['body_pose'] = initial_pose.repeat(n, axis=0).astype(np.float32)

    start_body_mesh, start_smplx_results = get_body_mesh(start_smplx, args.gender, 0, betas.shape[0])
    marker_start = start_smplx_results.vertices.detach().cpu().numpy()[:, markers_ids, :]
    joint_start = start_smplx_results.joints.detach().cpu().numpy()

	
    return marker_start.astype(np.float32), joint_start.astype(np.float32)

def get_forward_joint(joint_start):
	""" Joint_start: [B, N, 3] in xyz """
	x_axis = joint_start[:, 2, :] - joint_start[:, 1, :]
	x_axis[:, -1] = 0
	x_axis = x_axis / torch.norm(x_axis, dim=-1).unsqueeze(1)
	z_axis = torch.tensor([0, 0, 1]).float().unsqueeze(0).repeat(len(x_axis), 1).to(device)
	y_axis = torch.cross(z_axis, x_axis)
	y_axis = y_axis / torch.norm(y_axis, dim=-1).unsqueeze(1)
	transf_rotmat = torch.stack([x_axis, y_axis, z_axis], dim=1)
	return y_axis, transf_rotmat

def prepare_traj_input(joint_start, joint_end):
	""" Joints: [B, N, 3] in xyz """
	B, N, _ = joint_start.shape
	T = 62
	joint_sr_input = torch.ones(B, 4, T)  # [B, xyr, T]
	y_axis, transf_rotmat = get_forward_joint(joint_start)
	joint_start_new = joint_start.clone()
	joint_end_new = joint_end.clone()  # to check whether original joints change or not
	joint_start_new = torch.matmul(joint_start - joint_start[:, 0:1], transf_rotmat)
	joint_end_new = torch.matmul(joint_end - joint_start[:, 0:1], transf_rotmat)

	# start_forward, _ = get_forward_joint(joint_start_new)
	start_forward = torch.tensor([0, 1, 0]).unsqueeze(0)
	end_forward, _ = get_forward_joint(joint_end_new)

	joint_sr_input[:, :2, 0] = joint_start_new[:, 0, :2]  # xy
	joint_sr_input[:, :2, -1] = joint_end_new[:, 0, :2]   # xy
	joint_sr_input[:, 2:, 0] = start_forward[:, :2]  # r
	joint_sr_input[:, 2:, -1] = end_forward[:, :2]  # r


	# normalize
	traj_mean = torch.tensor(traj_stats['traj_Xmean']).unsqueeze(0).unsqueeze(2)
	traj_std = torch.tensor(traj_stats['traj_Xstd']).unsqueeze(0).unsqueeze(2)

	joint_sr_input_normed = (joint_sr_input - traj_mean) / traj_std
	for t in range(joint_sr_input_normed.size(-1)):
		joint_sr_input_normed[:, :, t] = joint_sr_input_normed[:, :, 0] + (joint_sr_input_normed[:, :, -1] - joint_sr_input_normed[:, :, 0])*t/(joint_sr_input_normed.size(-1)-1)
		joint_sr_input_normed[:, -2:, t] = joint_sr_input_normed[:, -2:, t] / torch.norm(joint_sr_input_normed[:, -2:, t], dim=1).unsqueeze(1)

	for t in range(joint_sr_input.size(-1)):
		joint_sr_input[:, :, t] = joint_sr_input[:, :, 0] + (joint_sr_input[:, :, -1] - joint_sr_input[:, :, 0])*t/(joint_sr_input.size(-1)-1)
		joint_sr_input[:, -2:, t] = joint_sr_input[:, -2:, t] / torch.norm(joint_sr_input[:, -2:, t], dim=1).unsqueeze(1)

	# linear interpolation

	return joint_sr_input_normed.float().to(device), joint_sr_input.float().to(device), transf_rotmat, joint_start_new, joint_end_new

def prepare_clip_img_input(marker_start, marker_end, joint_start, joint_end, joint_start_new, joint_end_new, transf_rotmat, traj_samples_unnormed_best, traj_sr_unnormed, end_body_smplx, object_transl_0, object_global_orient_0, traj_smoothed=True):
	B, n_markers, _ = marker_start.shape
	_, n_joints, _ = joint_start.shape
	markers = torch.rand(B, 61, n_markers, 3)  # [B, T, N ,3]
	joints = torch.rand(B, 61, n_joints, 3)  # [B, T, N ,3]

	marker_start_new = torch.matmul(marker_start - joint_start[:, 0:1], transf_rotmat)
	marker_end_new = torch.matmul(marker_end - joint_start[:, 0:1], transf_rotmat)  

	z_transl_to_floor_start = torch.min(marker_start_new[:, :, -1], dim=-1)[0]# - 0.03
	z_transl_to_floor_end = torch.min(marker_end_new[:, :, -1], dim=-1)[0]# - 0.03

	marker_start_new[:, :, -1] -= z_transl_to_floor_start.unsqueeze(1)
	marker_end_new[:, :, -1] -= z_transl_to_floor_end.unsqueeze(1)
	joint_start_new[:, :, -1] -= z_transl_to_floor_start.unsqueeze(1)
	joint_end_new[:, :, -1] -= z_transl_to_floor_end.unsqueeze(1)

	markers[:, 0] = marker_start_new
	markers[:, -1] = marker_end_new
	joints[:, 0] = joint_start_new
	joints[:, -1] = joint_end_new

	cur_body = torch.cat([joints[:, :, 0:1], markers], dim=2)

	cur_body[:, :, :, [1, 2]] = cur_body[:, :, :, [2, 1]]  # => xyz -> xzy

	reference = cur_body[:, :, 0] * torch.tensor([1, 0, 1])  # => the xy of pelvis joint?
	cur_body = torch.cat([reference.unsqueeze(2), cur_body], dim=2)   # [B, T, 1(reference)+1(pelvis)+N, 3]

	# position to local frame
	cur_body[:, :, :, 0] = cur_body[:, :, :, 0] - cur_body[:, :, 0:1, 0]
	cur_body[:, :, :, -1] = cur_body[:, :, :, -1] - cur_body[:, :, 0:1, -1]

	forward = np.zeros((B, 62, 3))
	forward[:, :, :2] = traj_samples_unnormed_best[:, 2:].transpose(0, 2, 1)
	forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]
	forward[:, :, [1, 2]] = forward[:, :, [2, 1]]

	if traj_smoothed:
		direction_filterwidth = 20
		forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=1, mode='nearest')
		traj_samples_unnormed_best[:, 2] = forward[:, :, 0]
		traj_samples_unnormed_best[:, 3] = forward[:, :, -1]
    
	target = np.array([[0, 0, 1]])#.repeat(len(forward), axis=0)

	rotation = Quaternions.between(forward, target)[:, :, np.newaxis]  # [B, T, 1, 4]

	cur_body = rotation[:, :-1] * cur_body.detach().cpu().numpy()  # [B, T, 1+1+N, xzy]
	cur_body[:, 1:-1] = 0
	cur_body[:, :, :, [1, 2]] = cur_body[:, :, :, [2, 1]]  # xzy => xyz
	cur_body = cur_body[:, :, 1:, :]
	cur_body = cur_body.reshape(cur_body.shape[0], cur_body.shape[1], -1)  # [B, T, N*3]

	velocity = np.zeros((B, 3, 61))
	velocity[:, 0, :] = traj_samples_unnormed_best[:, 0, 1:] - traj_samples_unnormed_best[:, 0, 0:-1]  # [B, 2, 60] on Joint frame
	velocity[:, -1, :] = traj_samples_unnormed_best[:, 1, 1:] - traj_samples_unnormed_best[:, 1, 0:-1]  # [B, 2, 60] on Joint frame


	velocity = rotation[:, 1:] * velocity.transpose(0, 2, 1).reshape(B, 61, 1, 3)
	rvelocity = Pivots.from_quaternions(rotation[:, 1:] * -rotation[:, :-1]).ps   # [B, T-1, 1]
	rot_0_pivot = Pivots.from_quaternions(rotation[:, 0]).ps


	global_x = velocity[:, :, 0, 0]
	global_y = velocity[:, :, 0, 2]
	contact_lbls = np.zeros((B, 61, 4))

	channel_local = np.concatenate([cur_body, contact_lbls], axis=-1)[:, np.newaxis, :, :]  # [B, 1, T-1, d=N*3+4]
	T, d = channel_local.shape[-2], channel_local.shape[-1]
	channel_global_x = np.repeat(global_x, d).reshape(-1, 1, T, d)  # [B, 1, T-1, d]
	channel_global_y = np.repeat(global_y, d).reshape(-1, 1, T, d)  # [B, 1, T-1, d]
	channel_global_r = np.repeat(rvelocity, d).reshape(-1, 1, T, d)  # [B, 1, T-1, d]

	cur_body = np.concatenate([channel_local, channel_global_x, channel_global_y, channel_global_r], axis=1)  # [B, 4, T-1, d]

	cur_body[:, 0] = (cur_body[:, 0] - markers_stats['Xmean_local']) / markers_stats['Xstd_local']
	cur_body[:, 1:3] = (cur_body[:, 1:3] - markers_stats['Xmean_global_xy']) / markers_stats['Xstd_global_xy']
	cur_body[:, 3] = (cur_body[:, 3] - markers_stats['Xmean_global_r']) / markers_stats['Xstd_global_r']


	# mask cur_body
	cur_body = cur_body.transpose(0, 1, 3, 2)  # [B, 4, D, T-1]
	mask_t_1 = [0, 60]
	mask_t_0 = list(set(range(60+1)) - set(mask_t_1))
	cur_body[:, 0, 2:, mask_t_0] = 0.
	cur_body[:, 0, -4:, :] = 0.
	# print('Mask the markers in the following frames: ', mask_t_0)

	object_glocal_orient_new = torch.matmul(object_global_orient_0, transf_rotmat)
	object_transl_new = torch.matmul((object_transl_0.reshape(B, 1, 3) - joint_start[:, 0:1]).float(), transf_rotmat).squeeze().view(B, -1)# + np.array([0, 0, -z_transl])
	object_transl_new[:, -1] -= z_transl_to_floor_end


	end_body_smplx['global_orient'] = R.from_rotvec(end_body_smplx['global_orient']).as_matrix()

	for key in end_body_smplx.keys():
		end_body_smplx[key] = torch.tensor(end_body_smplx[key]).to(joint_start.device)
	
	end_body_smplx['global_orient'] = torch.matmul(end_body_smplx['global_orient'].float(), transf_rotmat)
	end_body_smplx['transl'] = torch.matmul((end_body_smplx['transl'].reshape(B, 1, 3) - joint_start[:, 0:1]).float(), transf_rotmat).squeeze()# + np.array([0, 0, -z_transl])

	for key in end_body_smplx.keys():
		end_body_smplx[key] = end_body_smplx[key].detach().cpu().numpy()
	end_body_smplx['global_orient'] = R.from_matrix(end_body_smplx['global_orient']).as_rotvec().astype(np.float32)

	# for key in end_body_smplx.keys():
	# 	print('processing:', key, end_body_smplx[key].shape)

	return cur_body, rot_0_pivot, object_transl_new, object_glocal_orient_new, end_body_smplx, marker_start_new, marker_end_new, traj_samples_unnormed_best

def motion_infilling_inference(model, clip_img_input_new):
	with torch.no_grad():
		z_rand = torch.randn((clip_img_input_new.size(0), 512)).cuda()
		clip_img_rec, _, _ = model(input=clip_img_input_new, is_train=False, z=z_rand)

	contact_lbl_rec = F.sigmoid(clip_img_rec[:, 0, -4:, :].permute(0, 2, 1))  # [B, T, 4]
	contact_lbl_rec[contact_lbl_rec > 0.5] = 1.0
	contact_lbl_rec[contact_lbl_rec <= 0.5] = 0.0

	return clip_img_rec, contact_lbl_rec

	
def opt_markers_fit_smplx(body_markers_rec, smplx_beta, vposer_model, markers_ids, smplx_model):
    transl_opt_T = []
    rot_6d_opt_T = []
    shape_T = []
    other_params_opt_T = []
    markers_rec_T = []
    T = body_markers_rec.shape[0]
    for t in range(T):
        print('Optimize sample {}...'.format(t))
        markers_rec_t = torch.from_numpy(body_markers_rec[t:t + 1, :]).float().cuda()  # np, [1, 67, 3]
        shape_t = torch.tensor(smplx_beta).cuda().view(1, -1)  # fixed shape  [1, 10]
        ############### init opt params ##################
        if t == 0:
            transl_opt_t = torch.zeros(1, 3).cuda()
            rot_opt_t = torch.zeros(1, 3).cuda()
            # initialize todo: how to make it face y-axis
            transl_opt_t[:, 1] = 0.4
            transl_opt_t[:, 2] = 1.0
            # rot_opt_t[:, 0] = 0.
            rot_opt_t[:, 1] = 1.6
            rot_opt_t[:, 2] = 3.14

            rot_6d_opt_t = convert_to_6D_all(rot_opt_t)
            other_params_opt_t = torch.zeros(1, 56+24).cuda()  # other params except transl/rot/shape

            transl_opt_t.requires_grad = True
            rot_6d_opt_t.requires_grad = True
            other_params_opt_t.requires_grad = True

        final_params = [transl_opt_t, rot_6d_opt_t, other_params_opt_t]
        if t == 0:
            init_lr = 0.1
        else:
            init_lr = 0.01
        optimizer = optim.Adam(final_params, lr=init_lr)  # todo: set lr

        # fitting iteration

        total_steps = 100
        for step in range(total_steps):  # todo: set total step
            if step > 60:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.01
            if step > 80:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 0.003
            optimizer.zero_grad()
            body_params_opt_t = torch.cat([transl_opt_t, rot_6d_opt_t, shape_t, other_params_opt_t], dim=-1)  # [1, 75]
            body_params_opt_t_72 = convert_to_3D_rot(body_params_opt_t)  # tensor, [bs=1, 72]
            body_verts_opt_t, _ = gen_body_mesh_v1(body_params=body_params_opt_t_72, smplx_model=smplx_model,
                                                vposer_model=vposer_model)  # tensor [1, 10475, 3]
            markers_opt_t = body_verts_opt_t[:, markers_ids, :]  # [1, 67, 3]

            ### marker rec loss
            loss_marker = F.l1_loss(markers_opt_t, markers_rec_t)
            ### vposer loss
            vposer_pose = body_params_opt_t_72[:, 16:48]
            loss_vposer = torch.mean(vposer_pose ** 2)
            ### shape prior loss
            shape_params = body_params_opt_t_72[:, 6:16]
            loss_shape = torch.mean(shape_params ** 2)
            ### hand pose prior loss
            hand_params = body_params_opt_t_72[:, 48:]
            loss_hand = torch.mean(hand_params ** 2)

            ### todo: contact label loss
            # loss_contact_vel = torch.tensor(0.0).cuda()

            loss = args.weight_loss_rec_markers * loss_marker + \
            args.weight_loss_vposer * loss_vposer + \
            args.weight_loss_shape * loss_shape + args.weight_loss_hand * loss_hand


            loss.backward(retain_graph=True)

            optimizer.step()

        transl_opt_T.append(transl_opt_t.clone().detach())
        rot_6d_opt_T.append(rot_6d_opt_t.clone().detach())
        shape_T.append(shape_t.clone().detach())
        other_params_opt_T.append(other_params_opt_t.clone().detach())
        markers_rec_T.append(markers_rec_t.clone().detach())


    transl_opt_T = torch.stack(transl_opt_T).squeeze(1).detach()
    rot_6d_opt_T = torch.stack(rot_6d_opt_T).squeeze(1).detach()
    shape_T = torch.stack(shape_T).squeeze(1).detach()
    other_params_opt_T = torch.stack(other_params_opt_T).squeeze(1).detach()
    markers_rec_T = torch.stack(markers_rec_T).squeeze(1).detach()

    transl_opt_T.requires_grad = True
    rot_6d_opt_T.requires_grad = True
    other_params_opt_T.requires_grad = True

    return transl_opt_T, rot_6d_opt_T, shape_T, other_params_opt_T, markers_rec_T

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GraspMotion-Opt')

    """Config for GraspMotion"""
    parser.add_argument('--GraspPose_exp_name', default=None, type=str, help='Loaded GraspPose training experiment name')
    parser.add_argument('--object', default=None, type=str, help='object name')
    parser.add_argument('--gender', default=None, type=str, help='gender')
    parser.add_argument('--traj_ckpt_path', default='./pretrained_model/TrajFill_model.pkl', type=str, help='traj_infilling checkpoint path')
    parser.add_argument('--motion_ckpt_path', default='./pretrained_model/LocalMotionFill_model.pkl', type=str, help='traj_infilling checkpoint path')
    parser.add_argument('--traj_stats_path', default='./pretrained_model/prestats_GRAB_traj.npz', type=str, help='traj statistics')
    parser.add_argument('--markers_stats_dir', default='./pretrained_model/prestats_GRAB_contact_given_global_withHand_local_markers_3dv_4chan.npz', type=str, help='markers statistics')

    parser.add_argument('--stage1_weight_loss_rec_markers', type=float, default=1.0)
    parser.add_argument('--stage1_weight_loss_vposer', type=float, default=0.02)
    parser.add_argument('--stage1_weight_loss_shape', type=float, default=0.01)
    parser.add_argument('--stage1_weight_loss_hand', type=float, default=0.01)

    parser.add_argument('--stage2_weight_loss_rec_markers', type=float, default=0.1)
    parser.add_argument('--stage2_weight_loss_vposer', type=float, default=0.02)
    parser.add_argument('--stage2_weight_loss_shape', type=float, default=0.02)
    parser.add_argument('--stage2_weight_loss_hand', type=float, default=0.02)
    parser.add_argument('--stage2_weight_loss_skating', type=float, default=0.05)
    parser.add_argument('--stage2_weight_loss_smooth', type=float, default=2e7)  # 2e7
    parser.add_argument('--stage2_weight_loss_hand_smooth',
                        type=float, default=1)  # 1
    parser.add_argument('--stage2_weight_loss_hand_angle',
                        type=float, default=1)  # 1
    parser.add_argument('--stage2_weight_loss_contact',
                        type=float, default=60)  # 60
    parser.add_argument('--stage2_weight_loss_collision', type=float, default=10)  # 10
    parser.add_argument('--stage2_weight_loss_end_markers_fit',
                        type=float, default=10)  # 0.1

    args = parser.parse_args()
    cwd = os.getcwd()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mano_fname = './body_utils/smplx_mano_flame_correspondences/MANO_SMPLX_vertex_ids.pkl'
    with open(mano_fname, 'rb') as f:
        idxs_data = pickle.load(f)
        rhand_verts = idxs_data['right_hand']
        lhand_verts = idxs_data['left_hand']
    # markers setup
    markers_ids = get_markers_ids('f0_p5')  # different from grasppose training where we have dense markers on the hand, for motion infilling, we only use 5 markers on each palm.
    markers_ids_143 = get_markers_ids('f15_p22')

    # print(len(markers_ids))
    # print(len(markers_ids_143))

    """ 1. Load generated ending pose from the first stage """
    grasppose_result_path = cwd + '/results/{}/GraspPose/{}/fitting_results.npz'.format(args.GraspPose_exp_name, args.object)
    end_data, end_smplx, marker_end, joint_end, object_transl_0, object_global_orient_0 = load_ending_pose(args, grasppose_result_path)
    bs = len(marker_end)

    """ 2. Set initial pose (can be customized in set_initial_pose()) """
    # set initial pose -> can be customized
    marker_start, joint_start = set_initial_pose(args, end_smplx, markers_ids)
    marker_start = torch.tensor(marker_start).to(device)
    joint_start = torch.tensor(joint_start).to(device)
    marker_end = torch.tensor(marker_end).to(device)
    joint_end = torch.tensor(joint_end).to(device)

    """ 3. Generate in-between trajectories and local motions """
	### prepare models
    traj_model = Traj_MLP_CVAE(nz=512, feature_dim=4, T=62, residual='True', load_path=args.traj_ckpt_path).to(device)
    motion_model = Motion_CNN_CVAE(nz=512, downsample='True', in_channel=4, kernel=3, clip_seconds=2).to(device)
    ## todo: integrate checkpoint loading into motion model
    motion_model.load_state_dict(torch.load(args.motion_ckpt_path)['model_dict'])
    traj_model.eval()
    motion_model.eval()
    vposer_model_path = './body_utils/body_models/vposer_v1_0'
    vposer_model, _ = load_vposer(vposer_model_path, vp_model='snapshot')
    vposer_model = vposer_model.cuda()

    # prepare statistics
    traj_stats = np.load(args.traj_stats_path)
    markers_stats = np.load(args.markers_stats_dir)

	# generate in-between trajectories
    traj_sr_input, traj_sr_unnormed, transf_rotmat, joint_start_new, joint_end_new = prepare_traj_input(joint_start, joint_end)  # Note: this is the joint forward
    traj_samples = traj_model.sample(traj_sr_input.view(bs, -1))
    traj_mean = torch.tensor(traj_stats['traj_Xmean']).unsqueeze(0).unsqueeze(2).to(device)
    traj_std = torch.tensor(traj_stats['traj_Xstd']).unsqueeze(0).unsqueeze(2).to(device)
    traj_samples_unnormed = (traj_samples * traj_std + traj_mean).detach().cpu().numpy()

	# generate in-between local motions
    clip_img_input, rot_0_pivot, object_transl, object_global_orient, end_body_new, marker_start_new, marker_end_new, traj_input = prepare_clip_img_input(marker_start, marker_end, joint_start, joint_end, joint_start_new, joint_end_new, transf_rotmat, traj_samples_unnormed, traj_sr_unnormed, end_smplx, object_transl_0, object_global_orient_0)
    clip_img_input_new = torch.tensor(clip_img_input).to(device).float()  # [B, 4, D, T]
    clip_img_rec, contact_lbl_rec = motion_infilling_inference(motion_model, clip_img_input_new)


    """ 4. Optimization """
    contacts_object = end_data['contact'][()]['object']
    contacts_markers = end_data['contact'][()]['body']

    object_mesh = get_object_mesh(
        args.object, 'GRAB', object_transl.detach().cpu().numpy(), object_global_orient.detach().cpu().numpy(), bs, rotmat=True)
    object_vertices_shape = np.asarray(object_mesh[0].vertices).shape[0]
    object_index = np.linspace(
        0, object_vertices_shape, num=2048, endpoint=False, retstep=True, dtype=int)[0]

    saved_smplx_s1 = {}
    saved_smplx_final = {}

    for sample_index in tqdm(range(bs)):
        fittingconfig = {'T': 61,
        'gender': args.gender,
        'smplx_beta': end_smplx['betas'][sample_index],
        'init_lr_stage1': 0.1,
        'init_lr_stage2': 0.01,
        'num_iter': [100, 300],
        'device': 'cuda',
        'markers_ids': markers_ids,
        'markers_ids_143': markers_ids_143,
        ## loss weight for stage 1 optimization
        'stage1_weight_loss_rec_markers': args.stage1_weight_loss_rec_markers,
        'stage1_weight_loss_vposer': args.stage1_weight_loss_vposer,
        'stage1_weight_loss_shape': args.stage1_weight_loss_shape,
        'stage1_weight_loss_hand': args.stage1_weight_loss_hand,
        ## loss weight for stage 2 optimization
        'stage2_weight_loss_rec_markers': args.stage2_weight_loss_rec_markers,
        'stage2_weight_loss_end_markers_fit': args.stage2_weight_loss_end_markers_fit,
        'stage2_weight_loss_vposer': args.stage2_weight_loss_vposer,
        'stage2_weight_loss_hand': args.stage2_weight_loss_hand,
        'stage2_weight_loss_skating': args.stage2_weight_loss_skating,
        'stage2_weight_loss_smooth': args.stage2_weight_loss_smooth,
        'stage2_weight_loss_collision': args.stage2_weight_loss_collision,
        'stage2_weight_loss_contact': args.stage2_weight_loss_contact,
        'stage2_weight_loss_hand_smooth': args.stage2_weight_loss_hand_smooth,
        'stage2_weight_loss_hand_angle': args.stage2_weight_loss_hand_angle,
        }
        fittingop = FittingOP(fittingconfig)

        _, body_markers_rec = get_global_pose(clip_img_input_new[sample_index], clip_img_rec[sample_index], rot_0_pivot[sample_index], markers_stats)   #  [T, 79, 3]


        start_t = 30 ## todo
        verts_object = np.repeat(np.asarray(object_mesh[sample_index].vertices)[
                                 object_index].reshape(1, -1, 3), 61-start_t, axis=0)  # .repeat((61, 1, 1))
        normal_object = np.repeat(np.asarray(object_mesh[sample_index].vertex_normals)[
                                  object_index].reshape(1, -1, 3), 61-start_t, axis=0)

        verts_object = torch.from_numpy(verts_object).cuda()
        normal_object = torch.from_numpy(normal_object).cuda()
        contact_object = torch.from_numpy(contacts_object[sample_index:sample_index+1]).cuda()
        contact_markers = torch.from_numpy(contacts_markers[sample_index:sample_index+1]).cuda()

        transl_opt_T_s1, rot_6d_opt_T_s1, shape_T_s1, other_params_opt_T_s1, transl_opt_T_final, rot_6d_opt_T_final, shape_T_final, other_params_opt_T_final = fittingop.fitting(torch.tensor(body_markers_rec), marker_end_new[sample_index], rhand_verts, contact_lbl_rec[sample_index],
                        contact_object, contact_markers, normal_object, verts_object)

        body_params_opt_T_s1 = torch.cat([transl_opt_T_s1, rot_6d_opt_T_s1, shape_T_s1, other_params_opt_T_s1], dim=-1)  # [T, 75]
        body_params_opt_T_72_s1 = convert_to_3D_rot(body_params_opt_T_s1)  # tensor, [T, 72]
        body_verts_opt_T_s1, body_smplx_param_opt_T_s1 = gen_body_mesh_v1(body_params=body_params_opt_T_72_s1, smplx_model=fittingop.smplx_model_batch,
		                                        vposer_model=fittingop.vposer_model)

        body_params_opt_T_final = torch.cat([transl_opt_T_final, rot_6d_opt_T_final, shape_T_final, other_params_opt_T_final], dim=-1)  # [T, 75]
        body_params_opt_T_72_final = convert_to_3D_rot(body_params_opt_T_final)  # tensor, [T, 72]
        body_verts_opt_T_final, body_smplx_param_opt_T_final = gen_body_mesh_v1(body_params=body_params_opt_T_72_final, smplx_model=fittingop.smplx_model_batch,
		                                        vposer_model=fittingop.vposer_model)

        for key in body_smplx_param_opt_T_s1.keys():
            if key in saved_smplx_s1:
                saved_smplx_s1[key].append(
                    body_smplx_param_opt_T_s1[key].detach().cpu().numpy())
            else:
                saved_smplx_s1[key] = [
                    body_smplx_param_opt_T_s1[key].detach().cpu().numpy()]

        for key in body_smplx_param_opt_T_final.keys():
            if key in saved_smplx_final:
                saved_smplx_final[key].append(
                    body_smplx_param_opt_T_final[key].detach().cpu().numpy())
            else:
                saved_smplx_final[key] = [
                    body_smplx_param_opt_T_final[key].detach().cpu().numpy()]

    for key in saved_smplx_s1.keys():
        saved_smplx_s1[key] = np.asarray(saved_smplx_s1[key])
        saved_smplx_s1[key] = np.asarray(saved_smplx_s1[key])
    
    for key in saved_smplx_final.keys():
        saved_smplx_final[key] = np.asarray(saved_smplx_final[key])
        saved_smplx_final[key] = np.asarray(saved_smplx_final[key])

    saved_results = {}
    saved_results['body_orig'] = saved_smplx_s1
    saved_results['body_opt'] = saved_smplx_final

    saved_results['object_name'] = str(args.object)
    saved_results['object'] = {}
    saved_results['object']['transl'] = object_transl.detach().cpu().numpy()
    saved_results['object']['global_orient'] = object_global_orient.detach().cpu().numpy()

    result_save_path = './results/{}/GraspMotion/{}'.format(args.GraspPose_exp_name, args.object)
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)
    np.save(os.path.join(result_save_path, 'fitting_results'), saved_results)


