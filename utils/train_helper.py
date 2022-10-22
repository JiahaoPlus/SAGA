import os
import sys

import chamfer_distance as chd
import numpy as np
import scipy.ndimage.filters as filters
import torch

from utils.Pivots import Pivots
from utils.Pivots_torch import Pivots_torch
from utils.Quaternions import Quaternions
from utils.Quaternions_torch import Quaternions_torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def point2point_signed(
        x,
        y,
        x_normals=None,
        y_normals=None,
        return_vector=False,
):
    """
    signed distance between two pointclouds

    Args:
        x: FloatTensor of shape (N, P1, D) representing a batch of point clouds
            with P1 points in each batch element, batch size N and feature
            dimension D.
        y: FloatTensor of shape (N, P2, D) representing a batch of point clouds
            with P2 points in each batch element, batch size N and feature
            dimension D.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).

    Returns:

        - y2x_signed: Torch.Tensor
            the sign distance from y to x
        - y2x_signed: Torch.Tensor
            the sign distance from y to x
        - yidx_near: Torch.tensor
            the indices of x vertices closest to y

    """


    N, P1, D = x.shape
    P2 = y.shape[1]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")

    # ch_dist = chd.ChamferDistance()

    x_near, y_near, xidx_near, yidx_near = chd.ChamferDistance(x,y)

    xidx_near_expanded = xidx_near.view(N, P1, 1).expand(N, P1, D).to(torch.long)
    x_near = y.gather(1, xidx_near_expanded)

    yidx_near_expanded = yidx_near.view(N, P2, 1).expand(N, P2, D).to(torch.long)
    y_near = x.gather(1, yidx_near_expanded)

    x2y = x - x_near  # y point to x
    y2x = y - y_near  # x point to y

    if x_normals is not None:
        y_nn = x_normals.gather(1, yidx_near_expanded)
        in_out = torch.bmm(y_nn.view(-1, 1, 3), y2x.view(-1, 3, 1)).view(N, -1).sign()
        y2x_signed = y2x.norm(dim=2) * in_out

    else:
        y2x_signed = y2x.norm(dim=2)

    if y_normals is not None:
        x_nn = y_normals.gather(1, xidx_near_expanded)
        in_out_x = torch.bmm(x_nn.view(-1, 1, 3), x2y.view(-1, 3, 1)).view(N, -1).sign()
        x2y_signed = x2y.norm(dim=2) * in_out_x
    else:
        x2y_signed = x2y.norm(dim=2)

    if not return_vector:
        return y2x_signed, x2y_signed, yidx_near, xidx_near
    else:
        return y2x_signed, x2y_signed, yidx_near, xidx_near, y2x, x2y




class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.trace_func is not None:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop

def save_ckp(state, checkpoint_dir):
    f_path = os.path.join(checkpoint_dir, 'checkpoint.pt')
    torch.save(state, f_path)

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']


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

def prepare_traj_input(joint_start, joint_end, traj_Xmean, traj_Xstd):
    """ Joints: [B, N, 3] in xyz """
    B, N, _ = joint_start.shape
    T = 62
    joint_sr_input_unnormed = torch.ones(B, 4, T)  # [B, xyr, T]
    y_axis, transf_rotmat = get_forward_joint(joint_start)
    joint_start_new = joint_start.clone()
    joint_end_new = joint_end.clone()  # to check whether original joints change or not
    joint_start_new = torch.matmul(joint_start - joint_start[:, 0:1], transf_rotmat)
    joint_end_new = torch.matmul(joint_end - joint_start[:, 0:1], transf_rotmat)

    # start_forward, _ = get_forward_joint(joint_start_new)
    start_forward = torch.tensor([0, 1, 0]).unsqueeze(0)
    end_forward, _ = get_forward_joint(joint_end_new)

    joint_sr_input_unnormed[:, :2, 0] = joint_start_new[:, 0, :2]  # xy
    joint_sr_input_unnormed[:, :2, -2] = joint_end_new[:, 0, :2]   # xy
    joint_sr_input_unnormed[:, 2:, 0] = start_forward[:, :2]  # r
    joint_sr_input_unnormed[:, 2:, -2] = end_forward[:, :2]  # r

    # normalize
    traj_mean = traj_Xmean.unsqueeze(2).cpu()
    traj_std = traj_Xstd.unsqueeze(2).cpu()

    # linear interpolation
    joint_sr_input_normed = (joint_sr_input_unnormed - traj_mean) / traj_std
    for t in range(joint_sr_input_normed.size(-1)):
        joint_sr_input_normed[:, :, t] = joint_sr_input_normed[:, :, 0] + (joint_sr_input_normed[:, :, -2] - joint_sr_input_normed[:, :, 0])*t/(joint_sr_input_normed.size(-1)-2)
        joint_sr_input_normed[:, -2:, t] = joint_sr_input_normed[:, -2:, t] / torch.norm(joint_sr_input_normed[:, -2:, t], dim=1).unsqueeze(1)

    for t in range(joint_sr_input_unnormed.size(-1)):
        joint_sr_input_unnormed[:, :, t] = joint_sr_input_unnormed[:, :, 0] + (joint_sr_input_unnormed[:, :, -2] - joint_sr_input_unnormed[:, :, 0])*t/(joint_sr_input_unnormed.size(-1)-2)
        joint_sr_input_unnormed[:, -2:, t] = joint_sr_input_unnormed[:, -2:, t] / torch.norm(joint_sr_input_unnormed[:, -2:, t], dim=1).unsqueeze(1)

    return joint_sr_input_normed.float().to(device), joint_sr_input_unnormed.float().to(device), transf_rotmat, joint_start_new, joint_end_new

def prepare_clip_img_input(clip_img, marker_start, marker_end, joint_start, joint_end, joint_start_new, joint_end_new, transf_rotmat, traj_pred_unnormed, traj_sr_input_unnormed, traj_smoothed, markers_stats):
    traj_pred_unnormed = traj_pred_unnormed.detach().cpu().numpy()

    traj_pred_unnormed[:, :, 0] = traj_sr_input_unnormed[:, :, 0].detach().cpu().numpy()
    traj_pred_unnormed[:, :, -2] = traj_sr_input_unnormed[:, :, -2].detach().cpu().numpy()

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
    forward[:, :, :2] = traj_pred_unnormed[:, 2:].transpose(0, 2, 1)
    forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]
    forward[:, :, [1, 2]] = forward[:, :, [2, 1]]

    if traj_smoothed:
        forward_saved = forward.copy()
        direction_filterwidth = 20
        forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=1, mode='nearest')
        traj_pred_unnormed[:, 2] = forward[:, :, 0]
        traj_pred_unnormed[:, 3] = forward[:, :, -1]
    
    target = np.array([[0, 0, 1]])
    rotation = Quaternions.between(forward, target)[:, :, np.newaxis]  # [B, T, 1, 4]

    cur_body = rotation[:, :-1] * cur_body.detach().cpu().numpy()  # [B, T, 1+1+N, xzy]
    cur_body[:, 1:-1] = 0
    cur_body[:, :, :, [1, 2]] = cur_body[:, :, :, [2, 1]]  # xzy => xyz
    cur_body = cur_body[:, :, 1:, :]
    cur_body = cur_body.reshape(cur_body.shape[0], cur_body.shape[1], -1)  # [B, T, N*3]

    velocity = np.zeros((B, 3, 61))
    velocity[:, 0, :] = traj_pred_unnormed[:, 0, 1:] - traj_pred_unnormed[:, 0, 0:-1]  # [B, 2, 61] on Joint frame
    velocity[:, -1, :] = traj_pred_unnormed[:, 1, 1:] - traj_pred_unnormed[:, 1, 0:-1]  # [B, 2, 61] on Joint frame

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

    cur_body = np.concatenate([clip_img[:, 0:1].detach().permute(0, 1, 3, 2).cpu().numpy(), channel_global_x, channel_global_y, channel_global_r], axis=1)  # [B, 4, T-1, d]

    # cur_body[:, 0] = (cur_body[:, 0] - markers_stats['Xmean_local']) / markers_stats['Xstd_local']
    cur_body[:, 1:3] = (cur_body[:, 1:3] - markers_stats['Xmean_global_xy']) / markers_stats['Xstd_global_xy']
    cur_body[:, 3] = (cur_body[:, 3] - markers_stats['Xmean_global_r']) / markers_stats['Xstd_global_r']

    # mask cur_body
    cur_body = cur_body.transpose(0, 1, 3, 2)  # [B, 4, D, T-1]
    mask_t_1 = [0, 60]
    mask_t_0 = list(set(range(60+1)) - set(mask_t_1))
    cur_body[:, 0, 2:, mask_t_0] = 0.
    cur_body[:, 0, -4:, :] = 0.
    # print('Mask the markers in the following frames: ', mask_t_0)

    return torch.from_numpy(cur_body).float().to(device), rot_0_pivot, marker_start_new, marker_end_new, traj_pred_unnormed


def prepare_clip_img_input_torch(clip_img, marker_start, marker_end, joint_start, joint_end, 
                                 joint_start_new, joint_end_new, transf_rotmat, 
                                 traj_pred_unnormed, traj_sr_input_unnormed, 
                                 traj_smoothed, markers_stats):

    traj_pred_unnormed[:, :, 0] = traj_sr_input_unnormed[:, :, 0]#.detach().cpu().numpy()
    traj_pred_unnormed[:, :, -2] = traj_sr_input_unnormed[:, :, -2]#.detach().cpu().numpy()

    B, n_markers, _ = marker_start.shape
    _, n_joints, _ = joint_start.shape
    markers = torch.rand(B, 61, n_markers, 3).to(device)  # [B, T, N ,3]
    joints = torch.rand(B, 61, n_joints, 3).to(device)  # [B, T, N ,3]

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
    reference = cur_body[:, :, 0] * torch.tensor([1, 0, 1]).to(device)  # => the xy of pelvis joint?
    cur_body = torch.cat([reference.unsqueeze(2), cur_body], dim=2)   # [B, T, 1(reference)+1(pelvis)+N, 3]

    # position to local frame
    cur_body[:, :, :, 0] = cur_body[:, :, :, 0] - cur_body[:, :, 0:1, 0]
    cur_body[:, :, :, -1] = cur_body[:, :, :, -1] - cur_body[:, :, 0:1, -1]

    forward = torch.zeros((B, 62, 3)).to(device)
    forward[:, :, :2] = traj_pred_unnormed[:, 2:].permute(0, 2, 1)
    forward = forward / torch.sqrt((forward ** 2).sum(dim=-1)).unsqueeze(-1)
    forward[:, :, [1, 2]] = forward[:, :, [2, 1]]

    if traj_smoothed:  
        # forward_saved = forward.copy()
        direction_filterwidth = 20
        forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=1, mode='nearest')
        traj_pred_unnormed[:, 2] = forward[:, :, 0]
        traj_pred_unnormed[:, 3] = forward[:, :, -1]
    
    target = torch.tensor([[[0, 0, 1]]]).float().to(device).repeat(forward.size(0), forward.size(1), 1) #.repeat(len(forward), axis=0)
    # rotation = Quaternions.between(forward, target)[:, :, np.newaxis]  # [B, T, 1, 4]
    rotation = Quaternions_torch.between(forward, target).unsqueeze(2)

    cur_body = rotation[:, :-1] * cur_body#.detach().cpu().numpy()  # [B, T, 1+1+N, xzy]
    cur_body[:, 1:-1] = 0
    cur_body[:, :, :, [1, 2]] = cur_body[:, :, :, [2, 1]]  # xzy => xyz
    cur_body = cur_body[:, :, 1:, :]
    cur_body = cur_body.reshape(cur_body.shape[0], cur_body.shape[1], -1)  # [B, T, N*3]

    velocity = torch.zeros((B, 3, 61)).to(device)
    velocity[:, 0, :] = traj_pred_unnormed[:, 0, 1:] - traj_pred_unnormed[:, 0, 0:-1]  # [B, 2, 61] on Joint frame
    velocity[:, -1, :] = traj_pred_unnormed[:, 1, 1:] - traj_pred_unnormed[:, 1, 0:-1]  # [B, 2, 61] on Joint frame

    velocity = rotation[:, 1:] * velocity.permute(0, 2, 1).reshape(B, 61, 1, 3)
    rvelocity = Pivots_torch.from_quaternions(rotation[:, 1:] * -rotation[:, :-1]).ps   # [B, T-1, 1]
    rot_0_pivot = Pivots_torch.from_quaternions(rotation[:, 0]).ps

    global_x = velocity[:, :, 0, 0]
    global_y = velocity[:, :, 0, 2]

    T, d = clip_img.shape[-1], clip_img.shape[-2]
    channel_global_x = torch.repeat_interleave(global_x, d).reshape(-1, 1, T, d)  # [B, 1, T-1, d]
    channel_global_y = torch.repeat_interleave(global_y, d).reshape(-1, 1, T, d)  # [B, 1, T-1, d]
    channel_global_r = torch.repeat_interleave(rvelocity, d).reshape(-1, 1, T, d)  # [B, 1, T-1, d]

    cur_body = torch.cat([clip_img[:, 0:1].permute(0, 1, 3, 2), channel_global_x, channel_global_y, channel_global_r], dim=1)  # [B, 4, T-1, d]

    # cur_body[:, 0] = (cur_body[:, 0] - markers_stats['Xmean_local']) / markers_stats['Xstd_local']
    cur_body[:, 1:3] = (cur_body[:, 1:3] - torch.from_numpy(markers_stats['Xmean_global_xy']).float().to(device)) / torch.from_numpy(markers_stats['Xstd_global_xy']).float().to(device)
    cur_body[:, 3] = (cur_body[:, 3] - torch.from_numpy(markers_stats['Xmean_global_r']).float().to(device)) / torch.from_numpy(markers_stats['Xstd_global_r']).float().to(device)

    # mask cur_body
    cur_body = cur_body.permute(0, 1, 3, 2)  # [B, 4, D, T-1]
    mask_t_1 = [0, 60]
    mask_t_0 = list(set(range(60+1)) - set(mask_t_1))
    cur_body[:, 0, 2:, mask_t_0] = 0.
    cur_body[:, 0, -4:, :] = 0.
    # print('Mask the markers in the following frames: ', mask_t_0)
    return cur_body, rot_0_pivot, marker_start_new, marker_end_new, traj_pred_unnormed
