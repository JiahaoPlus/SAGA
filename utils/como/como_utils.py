import copy
import datetime
import json
import logging
import os
import pickle
import random
import sys
from importlib import import_module

import numpy as np
import scipy.ndimage.filters as filters
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry as tgm
from scipy.spatial.transform import Rotation as R
from torch.optim import lr_scheduler
from utils.Pivots import Pivots
from utils.Quaternions import Quaternions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rotation_matrix_to_angle_axis(rotation_matrix):
    """Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    # todo add check that matrix is a valid rotation matrix
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    return quaternion_to_angle_axis(quaternion)


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * (~ mask_d0_d1)
    mask_c2 = (~ mask_d2) * mask_d0_nd1
    mask_c3 = (~ mask_d2) * (~ mask_d0_nd1)
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


class ContinousRotReprDecoder(nn.Module):
    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    # def forward(self, module_input):
    #     reshaped_input = module_input.view(-1, 3, 2)
    #     b1 = F.normalize(reshaped_input[:, :, 0], dim=1)
    #     dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
    #     b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
    #     b3 = torch.cross(b1, b2, dim=1)
    #
    #     return torch.stack([b1, b2, b3], dim=-1)


    @staticmethod
    def decode(module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)


    @staticmethod
    def matrot2aa(pose_matrot):  # input: [bs, 3, 3]
        '''
        :param pose_matrot: Nx1xnum_jointsx9
        :return: Nx1xnum_jointsx3
        '''

        homogen_matrot = F.pad(pose_matrot.view(-1, 3, 3), [0,1])  # [bs, 3, 4], float
        # pose = tgm.rotation_matrix_to_angle_axis(homogen_matrot).view(-1, 3).contiguous()
        # original library has error (in pytorch > 1.1.0, cannot "- bool". use "~" instead of "1-").
        pose = rotation_matrix_to_angle_axis(homogen_matrot).view(-1, 3).contiguous()
        

        return pose

    @staticmethod
    def aa2matrot(pose):
        '''
        :param Nx1xnum_jointsx3
        :return: pose_matrot: Nx1xnum_jointsx9
        '''
        pose_body_matrot = tgm.angle_axis_to_rotation_matrix(pose.reshape(-1, 3))[:, :3, :3].contiguous()
        return pose_body_matrot


def get_logger(logdir):
    logger = logging.getLogger('emotion')
    logger.setLevel(logging.INFO)
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ", "_")
    ts = ts.replace(":", "_").replace("-","_")
    file_path = os.path.join(logdir, 'run_{}.log'.format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    return logger

def save_config(logdir, config):
    param_path = os.path.join(logdir, "params.json")
    print("[*] PARAM path: %s" % param_path)
    with open(param_path, 'w') as fp:
        json.dump(config.__dict__, fp, indent=4, sort_keys=True)


def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    image_paths = []
    for looproot, _, filenames in os.walk(rootdir):
        for filename in filenames:
            if filename.endswith(suffix):
                image_paths.append(os.path.join(looproot, filename))
    return image_paths


def get_scheduler(optimizer, policy, nepoch_fix=None, nepoch=None, decay_step=None, gamma=None):
    if policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - nepoch_fix) / float(nepoch - nepoch_fix + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=decay_step, gamma=0.1)
    elif policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', policy)
    return scheduler


def convert_to_6D_rot(x_batch):
    '''
    input: [transl, rotation, local params]
    convert global rotation from Eular angle to 6D continuous representation
    '''

    xt = x_batch[:,:3]
    xr = x_batch[:,3:6]
    xb = x_batch[:, 6:]

    xr_mat = ContinousRotReprDecoder.aa2matrot(xr) # return [:,3,3]
    xr_repr =  xr_mat[:,:,:-1].reshape([-1,6])

    return torch.cat([xt, xr_repr, xb], dim=-1)



def convert_to_3D_rot(x_batch):
    '''
    input: [transl, 6d rotation, local params]
    convert global rotation from 6D continuous representation to Eular angle
    '''
    xt = x_batch[:,:3]   # (reconstructed) normalized global translation
    xr = x_batch[:,3:9]  # (reconstructed) 6D rotation vector
    xb = x_batch[:,9:]   # pose $ shape parameters

    xr_mat = ContinousRotReprDecoder.decode(xr)  # [bs,3,3]
    xr_aa = ContinousRotReprDecoder.matrot2aa(xr_mat) # return [:,3]

    return torch.cat([xt, xr_aa, xb], dim=-1)



def convert_to_6D_all(x_batch):
    xr_mat = ContinousRotReprDecoder.aa2matrot(x_batch)  # return [:,3,3]
    xr_repr = xr_mat[:, :, :-1].reshape([-1, 6])
    return xr_repr


def convert_to_3D_all(x_batch):
    # x_batch: [bs, 6]
    xr_mat = ContinousRotReprDecoder.decode(x_batch)  # [bs,3,3]
    xr_aa = ContinousRotReprDecoder.matrot2aa(xr_mat)  # return [:,3]
    return xr_aa


def gen_body_mesh(body_params, pose_mode, with_hand, smplx_model, vposer_model):
    # body_params: [T, 3+6+10+32/126 (+180:hands)]
    bs = body_params.shape[0]  # T=120 frames
    body_params_dict = {}
    body_params_dict['transl'] = body_params[:, 0:3]  # [T, 3]
    body_params_dict['global_orient'] = convert_to_3D_all(body_params[:, 3:9])  # [T, 3]
    body_params_dict['betas'] = body_params[:, 9:19]

    if pose_mode == 'vposer':
        body_params_dict['body_pose'] = vposer_model.decode(body_params[:, 19:51], output_type='aa').view(bs, -1)
    elif pose_mode == '6d_rot':
        body_params_dict['body_pose'] = convert_to_3D_all(body_params[:, 19:(19 + 126)].reshape(-1, 6)).reshape(bs, -1)

    if not with_hand:
        body_params_dict['left_hand_pose'] = torch.zeros([bs, 45]).to(body_params.device)  # todo: for pca?
        body_params_dict['right_hand_pose'] = torch.zeros([bs, 45]).to(body_params.device)
    else:
        body_params_dict['left_hand_pose'] = body_params[:, 48:60]
        body_params_dict['right_hand_pose'] = body_params[:, 60:]

    smplx_output = smplx_model(return_verts=True, **body_params_dict)  # generated human body mesh
    body_verts = smplx_output.vertices  # [bs, n_body_vert, 3]
    return body_verts

def gen_body_mesh_v1(body_params, smplx_model, vposer_model):
    # body_params: [T, 3+6+10+32/126 (+180:hands)]
    bs = body_params.shape[0]
    body_params_dict = {}
    body_params_dict['transl'] = body_params[:, 0:3]  # [T, 3]
    body_params_dict['global_orient'] = body_params[:, 3:6]  # [T, 3]
    body_params_dict['betas'] = body_params[:, 6:16]
    body_params_dict['body_pose'] = vposer_model.decode(body_params[:, 16:48], output_type='aa').view(bs, -1)
    body_params_dict['left_hand_pose'] = body_params[:, 48:60+12]
    body_params_dict['right_hand_pose'] = body_params[:, 60+12:]

    smplx_output = smplx_model(return_verts=True, **body_params_dict)  # generated human body mesh
    body_verts = smplx_output.vertices  # [bs, n_body_vert, 3]
    return body_verts, body_params_dict

def gen_body_joints_v1(body_params, smplx_model, vposer_model):
    # body_params: [T, 3+6+10+32/126 (+180:hands)]
    bs = body_params.shape[0]  # T=120 frames
    body_params_dict = {}
    body_params_dict['transl'] = body_params[:, 0:3]  # [T, 3]
    body_params_dict['global_orient'] = body_params[:, 3:6]  # [T, 3]
    body_params_dict['betas'] = body_params[:, 6:16]
    body_params_dict['body_pose'] = vposer_model.decode(body_params[:, 16:48], output_type='aa').view(bs, -1)
    body_params_dict['left_hand_pose'] = body_params[:, 48:60+12]
    body_params_dict['right_hand_pose'] = body_params[:, 60+12:]

    smplx_output = smplx_model(return_verts=True, **body_params_dict)  # generated human body mesh
    body_joints = smplx_output.joints  # [bs, n_body_vert, 3]
    return body_joints


def gen_body_mesh_v1_amass(body_params, smplx_model, with_hand=True):
    # body_params: [T, 3+6+10+32/126 (+180:hands)]
    bs = body_params.shape[0]
    body_params_dict = {}
    body_params_dict['transl'] = body_params[:, 0:3]  # [T, 3]
    body_params_dict['global_orient'] = body_params[:, 3:6]  # [T, 3]
    body_params_dict['betas'] = body_params[:, 6:16]
    body_params_dict['body_pose'] = body_params[:, 16:79]
    if with_hand:
        body_params_dict['left_hand_pose'] = body_params[:, 79:124]
        body_params_dict['right_hand_pose'] = body_params[:, 124:]
    smplx_output = smplx_model(return_verts=True, **body_params_dict)  # generated human body mesh
    body_verts = smplx_output.vertices  # [bs, n_body_vert, 3]
    return body_verts

def gen_body_joints_v1_amass(body_params, smplx_model):
    # body_params: [T, 3+6+10+32/126 (+180:hands)]
    bs = body_params.shape[0]
    body_params_dict = {}
    body_params_dict['transl'] = body_params[:, 0:3]  # [T, 3]
    body_params_dict['global_orient'] = body_params[:, 3:6]  # [T, 3]
    body_params_dict['betas'] = body_params[:, 6:16]
    body_params_dict['body_pose'] = body_params[:, 16:79]
    body_params_dict['left_hand_pose'] = body_params[:, 79:124]
    body_params_dict['right_hand_pose'] = body_params[:, 124:]

    smplx_output = smplx_model(return_verts=True, **body_params_dict)  # generated human body mesh
    body_joints = smplx_output.joints  # [bs, n_body_vert, 3]
    return body_joints



JOINT_NAMES = [
    'pelvis',
    'left_hip',
    'right_hip',
    'spine1',
    'left_knee',
    'right_knee',
    'spine2',
    'left_ankle',
    'right_ankle',
    'spine3',
    'left_foot',
    'right_foot',
    'neck',
    'left_collar',
    'right_collar',
    'head',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'jaw',
    'left_eye_smplhf',
    'right_eye_smplhf'] # first 25 joints in smplx

LIMBS_BODY = [(23, 15),
         (24, 15),
         (15, 22),
         (22, 12),
         # left arm
         (12, 13),
         (13, 16),
         (16, 18),
         (18, 20),
         # right arm
         (12, 14),
         (14, 17),
         (17, 19),
         (19, 21),
         # spline
         (12, 9),
         (9, 6),
         (6, 3),
         (3, 0),
         # left leg
         (0, 1),
         (1, 4),
         (4, 7),
         (7, 10),
         # right leg
         (0, 2),
         (2, 5),
         (5, 8),
         (8, 11)]


LIMBS_BODY_SMPL = [(15, 12),
         # left arm
         (12, 13),
         (13, 16),
         (16, 18),
         (18, 20),
         (20, 22),
         # right arm
         (12, 14),
         (14, 17),
         (17, 19),
         (19, 21),
         (21, 23),
         # spline
         (12, 9),
         (9, 6),
         (6, 3),
         (3, 0),
         # left leg
         (0, 1),
         (1, 4),
         (4, 7),
         (7, 10),
         # right leg
         (0, 2),
         (2, 5),
         (5, 8),
         (8, 11),]



LIMBS_HAND = [(20, 25),
              (25, 26),
              (26, 27),
              (20, 28),
              (28, 29),
              (29, 30),
              (20, 31),
              (31, 32),
              (32, 33),
              (20, 34),
              (34, 35),
              (35, 36),
              (20, 37),
              (37, 38),
              (38, 39),
              # right hand
              (21, 40),
              (40, 41),
              (41, 42),
              (21, 43),
              (43, 44),
              (44, 45),
              (21, 46),
              (46, 47),
              (47, 48),
              (21, 49),
              (49, 50),
              (50, 51),
              (21, 52),
              (52, 53),
              (53, 54)]




def softmax(x, **kw):
    softness = kw.pop('softness', 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))

def softmin(x, **kw):
    return -softmax(-x, **kw)


def reconstruct_global_joints(body_joints_input):
    root_traj = body_joints_input[:, -1]  # [T-1, 3]
    root_r, root_x, root_z = root_traj[:, 2], root_traj[:, 0], root_traj[:, 1]  # [T-1]
    body_joints_input = body_joints_input[:, 0:-1]  # [T-1, 25+1, 3]
    body_joints_input[:, :, [1, 2]] = body_joints_input[:, :, [2, 1]]
    rotation = Quaternions.id(1)
    offsets = []
    translation = np.array([[0, 0, 0]])
    for i in range(len(body_joints_input)):
        # if i == 0:
        #     rotation = Quaternions.from_angle_axis(-rot_0_pivot, np.array([0, 1, 0])) * rotation   # t=0
        body_joints_input[i, :, :] = rotation * body_joints_input[i]
        body_joints_input[i, :, 0] = body_joints_input[i, :, 0] + translation[0, 0]
        body_joints_input[i, :, 2] = body_joints_input[i, :, 2] + translation[0, 2]
        rotation = Quaternions.from_angle_axis(-root_r[i], np.array([0, 1, 0])) * rotation
        offsets.append(rotation * np.array([0, 0, 1]))
        translation = translation + rotation * np.array([root_x[i], 0, root_z[i]])
    body_joints_input[:, :, [1, 2]] = body_joints_input[:, :, [2, 1]]
    body_joints_input = body_joints_input[:, 1:, :]
    return body_joints_input


def reconstruct_global_joints_new(body_joints_input, rot_0_pivot):
    root_traj = body_joints_input[:, -1]  # [T, 3]
    root_r, root_x, root_z = root_traj[:, 2], root_traj[:, 0], root_traj[:, 1]  # [T]
    body_joints_input = body_joints_input[:, 0:-1]  # [T, 25+1, 3]
    body_joints_input[:, :, [1, 2]] = body_joints_input[:, :, [2, 1]]
    rotation = Quaternions.id(1)
    # offsets = []
    translation = np.array([[0, 0, 0]])
    for i in range(len(body_joints_input)):
        if i == 0:
            rotation = Quaternions.from_angle_axis(-rot_0_pivot, np.array([0, 1, 0])) * rotation   # t=0
        body_joints_input[i, :, :] = rotation * body_joints_input[i]
        body_joints_input[i, :, 0] = body_joints_input[i, :, 0] + translation[0, 0]
        body_joints_input[i, :, 2] = body_joints_input[i, :, 2] + translation[0, 2]
        rotation = Quaternions.from_angle_axis(-root_r[i], np.array([0, 1, 0])) * rotation
        translation = translation + rotation * np.array([root_x[i], 0, root_z[i]])

    body_joints_input[:, :, [1, 2]] = body_joints_input[:, :, [2, 1]]
    body_joints_input = body_joints_input[:, 1:, :]
    return body_joints_input

def reconstruct_global_joints_v1(body_joints_input):
    root_traj = body_joints_input[:, -1]  # [T-1, 3]
    root_r, root_x, root_z = root_traj[:, 2], root_traj[:, 0], root_traj[:, 1]  # [T-1]
    body_joints_input = body_joints_input[:, 0:-1]  # [T-1, 25+1, 3]?
    body_joints_input[:, :, [1, 2]] = body_joints_input[:, :, [2, 1]]  # switch y,z axis
    for i in range(len(body_joints_input)):
        rotation = Quaternions.from_angle_axis(-root_r[i], np.array([0, 1, 0]))
        body_joints_input[i, :, :] = rotation * body_joints_input[i]
        body_joints_input[i, :, 0] = body_joints_input[i, :, 0] + root_x[i]
        body_joints_input[i, :, 2] = body_joints_input[i, :, 2] + root_z[i]

    body_joints_input[:, :, [1, 2]] = body_joints_input[:, :, [2, 1]]
    body_joints_input = body_joints_input[:, 1:, :]
    return body_joints_input


def get_local_joint_3dv_new(cur_body):
    # cur_body: numpy, [T, 25, 3], in (x,y,z)
    cur_body[:, :, [1, 2]] = cur_body[:, :, [2, 1]]  # swap y/z axis  --> in (x,z,y)

    """ Put on Floor """
    cur_body[:, :, 1] = cur_body[:, :, 1] - cur_body[:, :, 1].min()

    """ Add Reference Joint """
    trajectory_filterwidth = 3
    reference = cur_body[:, 0] * np.array([1, 0, 1])  # [T, 3], (x,y,0)
    # reference = filters.gaussian_filter1d(reference, trajectory_filterwidth, axis=0, mode='nearest')
    cur_body = np.concatenate([reference[:, np.newaxis], cur_body], axis=1)  # [T, 1+25, 3]

    """ Get Root Velocity in floor plane """
    velocity = (cur_body[1:, 0:1] - cur_body[0:-1, 0:1]).copy()  # [T-1, 3] ([:, 1]==0)
    velocity = np.concatenate([np.array([[[0, 0, 0]]]), velocity], axis=0)  # [T, 3], t=0: [0,0,0]

    """ To local coordinates """
    cur_body[:, :, 0] = cur_body[:, :, 0] - cur_body[:, 0:1, 0]  # [T, 1+25, 3]
    cur_body[:, :, 2] = cur_body[:, :, 2] - cur_body[:, 0:1, 2]

    """ Get Forward Direction """
    sdr_l, sdr_r, hip_l, hip_r = 16, 17, 1, 2
    across1 = cur_body[:, hip_l] - cur_body[:, hip_r]
    across0 = cur_body[:, sdr_l] - cur_body[:, sdr_r]
    across = across0 + across1
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    direction_filterwidth = 20
    forward = np.cross(across, np.array([[0, 1, 0]]))
    forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')
    forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

    """ Remove Y Rotation """
    target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
    rotation = Quaternions.between(forward, target)[:, np.newaxis]
    cur_body = rotation * cur_body  # [T, 1+25, 3]

    """ Get Root Rotation """
    velocity = rotation * velocity  # [T, 1, 3]
    rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps  # [T-1, 1]
    rvelocity_0 = Pivots.from_quaternions(rotation[0]).ps[..., np.newaxis]
    rvelocity = np.concatenate([rvelocity_0, rvelocity], axis=0)

    cur_body[:, :, [1, 2]] = cur_body[:, :, [2, 1]]
    # cur_body = cur_body[:-1]  # [T-1, 1+25, 3]
    cur_body = cur_body.reshape(len(cur_body), -1)
    cur_body = np.concatenate([cur_body, velocity[:, :, 0]], axis=-1)
    cur_body = np.concatenate([cur_body, velocity[:, :, 2]], axis=-1)
    cur_body = np.concatenate([cur_body, rvelocity], axis=-1)  # numpy, [T-1, d=81]
    return cur_body


def get_local_joint_3dv(cur_body):
    # cur_body: numpy, [T, 25, 3], in (x,y,z)
    cur_body[:, :, [1, 2]] = cur_body[:, :, [2, 1]]  # swap y/z axis  --> in (x,z,y)

    """ Put on Floor """
    cur_body[:, :, 1] = cur_body[:, :, 1] - cur_body[:, :, 1].min()

    """ Add Reference Joint """
    # trajectory_filterwidth = 3
    reference = cur_body[:, 0] * np.array([1, 0, 1])  # [T, 3], (x,y,0)
    # reference = filters.gaussian_filter1d(reference, trajectory_filterwidth, axis=0, mode='nearest')
    cur_body = np.concatenate([reference[:, np.newaxis], cur_body], axis=1)  # [T, 1+25, 3]

    """ Get Root Velocity in floor plane """
    velocity = (cur_body[1:, 0:1] - cur_body[0:-1, 0:1]).copy()  # [T-1, 3] ([:, 1]==0)

    """ To local coordinates """
    cur_body[:, :, 0] = cur_body[:, :, 0] - cur_body[:, 0:1, 0]  # [T, 1+25, 3]
    cur_body[:, :, 2] = cur_body[:, :, 2] - cur_body[:, 0:1, 2]

    """ Get Forward Direction """
    sdr_l, sdr_r, hip_l, hip_r = 16, 17, 1, 2
    across1 = cur_body[:, hip_l] - cur_body[:, hip_r]
    across0 = cur_body[:, sdr_l] - cur_body[:, sdr_r]
    across = across0 + across1
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    direction_filterwidth = 20
    forward = np.cross(across, np.array([[0, 1, 0]]))
    forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')
    forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

    """ Remove Y Rotation """
    target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
    rotation = Quaternions.between(forward, target)[:, np.newaxis]
    cur_body = rotation * cur_body  # [T, 1+25, 3]

    """ Get Root Rotation """
    velocity = rotation[1:] * velocity  # [T-1, 1, 3]
    rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps  # [T-1, 1]

    cur_body[:, :, [1, 2]] = cur_body[:, :, [2, 1]]
    cur_body = cur_body[:-1]  # [T-1, 1+25, 3]
    cur_body = cur_body.reshape(len(cur_body), -1)
    cur_body = np.concatenate([cur_body, velocity[:, :, 0]], axis=-1)
    cur_body = np.concatenate([cur_body, velocity[:, :, 2]], axis=-1)
    cur_body = np.concatenate([cur_body, rvelocity], axis=-1)  # numpy, [T-1, d=81]
    return cur_body


def get_local_joint_3dv_v1(cur_body):
    cur_body[:, :, [1, 2]] = cur_body[:, :, [2, 1]]  # swap y/z axis  --> in (x,z,y)

    """ Put on Floor """
    cur_body[:, :, 1] = cur_body[:, :, 1] - cur_body[:, :, 1].min()

    """ Add Reference Joint """
    reference = cur_body[:, 0] * np.array([1, 0, 1])  # [T, 3], (x,y,0)
    cur_body = np.concatenate([reference[:, np.newaxis], cur_body], axis=1)  # [T, 1+25, 3]

    # """ Get Root Velocity in floor plane """
    # velocity = (cur_body[1:, 0:1] - cur_body[0:-1, 0:1]).copy()  # [T-1, 3] ([:, 1]==0)

    """ To local coordinates """
    cur_body[:, :, 0] = cur_body[:, :, 0] - cur_body[:, 0:1, 0]  # [T, 1+25, 3]
    cur_body[:, :, 2] = cur_body[:, :, 2] - cur_body[:, 0:1, 2]

    """ Get Forward Direction """
    sdr_l, sdr_r, hip_l, hip_r = 16, 17, 1, 2
    across1 = cur_body[:, hip_l] - cur_body[:, hip_r]
    across0 = cur_body[:, sdr_l] - cur_body[:, sdr_r]
    across = across0 + across1
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    direction_filterwidth = 20
    forward = np.cross(across, np.array([[0, 1, 0]]))
    forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')
    forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

    """ Remove Y Rotation """
    target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
    rotation = Quaternions.between(forward, target)[:, np.newaxis]
    cur_body = rotation * cur_body  # [T, 1+25, 3]

    """ Get Root Rotation """
    rotation = Pivots.from_quaternions(rotation).ps  # [T, 1]

    cur_body[:, :, [1, 2]] = cur_body[:, :, [2, 1]]
    cur_body = cur_body.reshape(len(cur_body), -1)
    cur_body = np.concatenate([cur_body, reference[:, 0:1]], axis=-1)
    cur_body = np.concatenate([cur_body, reference[:, 2:3]], axis=-1)
    cur_body = np.concatenate([cur_body, rotation], axis=-1)  # [T-1, d=81]
    return cur_body


def get_local_markers_3dv_4chan(cur_body, contact_lbls):
    # cur_body: numpy, [T, 1+67, 3]
    # contact_lbls: numpy, [T, 4]
    cur_body[:, :, [1, 2]] = cur_body[:, :, [2, 1]]  # swap y/z axis  --> in (x,z,y)

    """ Put on Floor """
    cur_body[:, :, 1] = cur_body[:, :, 1] - cur_body[:, :, 1].min()

    """ Add Reference Joint """
    reference = cur_body[:, 0] * np.array([1, 0, 1])  # [T, 3], (x,y,0)
    cur_body = np.concatenate([reference[:, np.newaxis], cur_body], axis=1)  # [T, 1+25, 3]

    """ Get Root Velocity in floor plane """
    velocity = (cur_body[1:, 0:1] - cur_body[0:-1, 0:1]).copy()  # [T-1, 3] ([:, 1]==0)

    """ To local coordinates """
    cur_body[:, :, 0] = cur_body[:, :, 0] - cur_body[:, 0:1, 0]  # [T, 1+25, 3]
    cur_body[:, :, 2] = cur_body[:, :, 2] - cur_body[:, 0:1, 2]

    """ Get Forward Direction """
    sdr_l, sdr_r, hip_l, hip_r = 26 + 1 + 1, 56 + 1 + 1, 27 + 1 + 1, 57 + 1 + 1  # +1+1: [0]: reference, [1]: pelvis
    across1 = cur_body[:, hip_r] - cur_body[:, hip_l]
    across0 = cur_body[:, sdr_r] - cur_body[:, sdr_l]
    across = across0 + across1
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    direction_filterwidth = 20
    forward = np.cross(across, np.array([[0, 1, 0]]))
    forward = filters.gaussian_filter1d(forward, direction_filterwidth, axis=0, mode='nearest')
    forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

    """ Remove Y Rotation """
    target = np.array([[0, 0, 1]]).repeat(len(forward), axis=0)
    rotation = Quaternions.between(forward, target)[:, np.newaxis]
    cur_body = rotation * cur_body  # [T, 1+25, 3]

    """ Get Root Rotation """
    velocity = rotation[1:] * velocity  # [T-1, 1, 3]
    rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps  # [T-1, 1]

    rot_0_pivot = Pivots.from_quaternions(rotation[0]).ps  # [T-1, 1]

    cur_body[:, :, [1, 2]] = cur_body[:, :, [2, 1]]
    cur_body = cur_body[0:-1, 1:, :]  # [T-1, 25, 3]  exclude reference joint
    cur_body = cur_body.reshape(len(cur_body), -1)  # [T-1, 75]


    channel_local = np.concatenate([cur_body, contact_lbls[0:-1]], axis=-1)[np.newaxis, :, :]  # [1, T-1, d=75+4]
    T, d = channel_local.shape[1], channel_local.shape[-1]
    global_x, global_y = velocity[:, :, 0], velocity[:, :, 2]  # [T-1, 1]
    channel_global_x = np.repeat(global_x, d).reshape(1, T, d)  # [1, T-1, d]
    channel_global_y = np.repeat(global_y, d).reshape(1, T, d)  # [1, T-1, d]
    channel_global_r = np.repeat(rvelocity, d).reshape(1, T, d)  # [1, T-1, d]

    cur_body = np.concatenate([channel_local, channel_global_x, channel_global_y, channel_global_r],
                               axis=0)  # [4, T-1, d]
    return cur_body, rot_0_pivot

openpose2smplx = [(1, 12),
                  (2, 17),
                  (3, 19),
                  (4, 21),
                  (5, 16),
                  (6, 18),
                  (7, 20),
                  (8, 0),
                  (9, 2),
                  (10, 5),
                  (11, 8),
                  (12, 1),
                  (13, 4),
                  (14, 7),
                  (11, 11),
                  (14, 10)]  # (1, 12): openpose joint 1 = smplx joint 12

LIMBS_MARKER_SSM2 = [(65, 63),
                     (65, 39),
                     (63, 9),
                     (39, 9),
                     (63, 64),
                     (65, 66),
                     (39, 56),
                     (9, 26),
                     (56, 1),
                     (26, 1),
                     (1, 61),
                     (61, 38),
                     (61, 8),
                     (38, 52),
                     (8, 22),
                     (52, 33),
                     (22, 3),
                     (33, 31),
                     (3, 31),
                     (33, 57),
                     (3, 27),
                     (57, 45),
                     (27, 14),
                     (45, 48),
                     (14, 18),
                     (48, 59),
                     (18, 29),
                     (59, 32),
                     (29, 2),
                     (32, 51),
                     (2, 21),
                     # arm
                     (56, 40),
                     (40, 43),
                     (43, 53),
                     (53, 42),
                     (26, 5),
                     (5, 10),
                     (10, 13),
                     (13, 23),
                     (23, 12),
                     # # back
                     # (64, 0),
                     # (66, 0),
                     # (0, 4),
                     # (0, 34),
                     # (4, 6),
                     # (34, 36),
                     # (6, 62),
                     # (36, 62),
                     # (62, 24),
                     # (62, 54),
                     # (24, 7),
                     # (54, 37),
                     # (7, 16),
                     # (37, 47)
                     ]


def update_globalRT_for_smplx(body_params, smplx_model, trans_to_target_origin, delta_T=None):
    '''
    input:
        body_params: array, [72], under camera coordinate
        smplx_model: the model to generate smplx mesh, given body_params
        trans_to_target_origin: coordinate transformation [4,4] mat
    Output:
        body_params with new globalR and globalT, which are corresponding to the new coord system
    '''

    ### step (1) compute the shift of pelvis from the origin
    body_params_dict = {}
    body_params_dict['transl'] = np.expand_dims(body_params[:3], axis=0)
    body_params_dict['global_orient'] = np.expand_dims(body_params[3:6], axis=0)
    body_params_dict['betas'] = np.expand_dims(body_params[6:16], axis=0)
    body_params_dict['body_pose'] = np.expand_dims(body_params[16:79], axis=0)
    body_params_dict['left_hand_pose'] = np.expand_dims(body_params[79:124], axis=0)
    body_params_dict['right_hand_pose'] = np.expand_dims(body_params[124:], axis=0)

    body_param_dict_torch = {}
    for key in body_params_dict.keys():
        body_param_dict_torch[key] = torch.FloatTensor(body_params_dict[key]).to(device)

    if delta_T is None:
        body_param_dict_torch['transl'] = torch.zeros([1,3], dtype=torch.float32).to(device)
        body_param_dict_torch['global_orient'] = torch.zeros([1,3], dtype=torch.float32).to(device)
        smplx_out = smplx_model(return_verts=True, **body_param_dict_torch)
        delta_T = smplx_out.joints[0,0,:] # (3,)
        delta_T = delta_T.detach().cpu().numpy()

    ### step (2): calibrate the original R and T in body_params
    body_R_angle = body_params_dict['global_orient'][0]
    body_R_mat = R.from_rotvec(body_R_angle).as_dcm() # to a [3,3] rotation mat
    body_T = body_params_dict['transl'][0]
    body_mat = np.eye(4)
    body_mat[:-1,:-1] = body_R_mat
    body_mat[:-1, -1] = body_T + delta_T

    ### step (3): perform transformation, and decalib the delta shift
    body_params_dict_new = copy.deepcopy(body_params_dict)
    body_mat_new = np.dot(trans_to_target_origin, body_mat)
    body_R_new = R.from_dcm(body_mat_new[:-1,:-1]).as_rotvec()
    body_T_new = body_mat_new[:-1, -1]
    body_params_dict_new['global_orient'] = body_R_new.reshape(1,3)
    body_params_dict_new['transl'] = (body_T_new - delta_T).reshape(1,3)

    body_param_new = np.concatenate([body_params_dict_new['transl'], body_params_dict_new['global_orient'],
                                     body_params_dict_new['betas'], body_params_dict_new['body_pose'],
                                     body_params_dict_new['left_hand_pose'], body_params_dict_new['right_hand_pose']], axis=-1)  # [1, d]
    return body_param_new





def read_prox_pkl(pkl_path):
    body_params_dict = {}
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        # data keys: camera_rotation, camera_translation, (useless)
        # transl, global_orient, betas, body_pose, pose_embedding
        # left_hand_pose, right_hand_pose,
        # jaw_pose, leye_pose, reye_pose, expression
        body_params_dict['transl'] = data['transl'][0]
        body_params_dict['global_orient'] = data['global_orient'][0]
        body_params_dict['betas'] = data['betas'][0]
        body_params_dict['body_pose'] = data['body_pose'][0]  # array, [63,]
        body_params_dict['pose_embedding'] = data['pose_embedding'][0]

        body_params_dict['left_hand_pose'] = data['left_hand_pose'][0]
        body_params_dict['right_hand_pose'] = data['right_hand_pose'][0]
        body_params_dict['jaw_pose'] = data['jaw_pose'][0]
        body_params_dict['leye_pose'] = data['leye_pose'][0]
        body_params_dict['reye_pose'] = data['reye_pose'][0]
        body_params_dict['expression'] = data['expression'][0]
    return body_params_dict

