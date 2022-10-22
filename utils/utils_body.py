import os
import sys

sys.path.append('..')
sys.path.append('../..')
import json

import numpy as np
import open3d as o3d
import smplx
import torch
from WholeGraspPose.models.objectmodel import ObjectModel

from utils.Pivots import Pivots
from utils.Quaternions import Quaternions


def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]], 
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat
    
def caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr/ scale
    # must ensure pVec_Arr is also a unit vec. 
    z_unit_Arr = np.array([0,0,1])
    z_mat = get_cross_prod_mat(z_unit_Arr)

    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)

    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:   
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
                                                    z_c_vec_mat)/(1 + np.dot(z_unit_Arr, pVec_Arr))

    qTrans_Mat *= scale
    return qTrans_Mat

def update_cam(cam_param, trans):
    cam_R = np.transpose(trans[:-1, :-1])
    cam_T = -trans[:-1, -1:]
    cam_T = np.matmul(cam_R, cam_T)  # !!!!!! T is applied in the rotated coord
    cam_aux = np.array([[0, 0, 0, 1]])
    mat = np.concatenate([cam_R, cam_T], axis=-1)
    mat = np.concatenate([mat, cam_aux], axis=0)
    cam_param.extrinsic = mat
    return cam_param

def update_cam_new(cam_param, trans):
    cam_R = np.transpose(trans[:-1, :-1])
    cam_T = trans[:-1, -1:]
    # cam_T = np.matmul(cam_R, cam_T)  # !!!!!! T is applied in the rotated coord
    cam_aux = np.array([[0, 0, 0, 1]])
    mat = np.concatenate([cam_R, cam_T], axis=-1)
    mat = np.concatenate([mat, cam_aux], axis=0)
    cam_param.extrinsic = mat
    return cam_param

def color_hex2rgb(hex):
    h = hex.lstrip('#')
    return np.array(  tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) )/255
    
def create_lineset(x_range, y_range, z_range):
    gp_lines = o3d.geometry.LineSet()
    gp_pcd = o3d.geometry.PointCloud()
    points = np.stack(np.meshgrid(x_range, y_range, z_range), axis=-1)

    lines = []
    for ii in range(x_range.shape[0]-1):
        for jj in range(y_range.shape[0]-1):
            lines.append(np.array([ii*x_range.shape[0]+jj, ii*x_range.shape[0]+jj+1]))
            lines.append(np.array([ii*x_range.shape[0]+jj, ii*x_range.shape[0]+jj+y_range.shape[0]]))

    points = np.reshape(points, [-1,3])
    colors = np.random.rand(len(lines), 3)*0.5+0.5

    gp_lines.points = o3d.utility.Vector3dVector(points)
    gp_lines.colors = o3d.utility.Vector3dVector(colors)
    gp_lines.lines = o3d.utility.Vector2iVector(np.stack(lines,axis=0))
    gp_pcd.points = o3d.utility.Vector3dVector(points)

    return gp_lines, gp_pcd

def gen_body_mesh_v1(body_params, smplx_model, vposer_model, return_normal=False):
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

def get_body_model(type, gender, batch_size,device='cuda', v_template=None):
    '''
    type: smpl, smplx smplh and others. Refer to smplx tutorial
    gender: male, female, neutral
    batch_size: an positive integar
    '''
    body_model_path = './body_utils/body_models'
    body_model = smplx.create(body_model_path, model_type=type,
                                    gender=gender, ext='npz',
                                    num_pca_comps=24,
                                    create_global_orient=True,
                                    create_body_pose=True,
                                    create_betas=True,
                                    create_left_hand_pose=True,
                                    create_right_hand_pose=True,
                                    create_expression=True,
                                    create_jaw_pose=True,
                                    create_leye_pose=True,
                                    create_reye_pose=True,
                                    create_transl=True,
                                    batch_size=batch_size,
                                    v_template=v_template
                                    )
    if device == 'cuda':
        return body_model.cuda()
    else:
        return body_model


def get_body_mesh(smplxparams, gender, start_idx, n_samples, device='cpu', color=None):
    body_mesh_list = []

    for key in smplxparams.keys():
        # print(key, smplxparams[key].shape)
        smplxparams[key] = torch.tensor(smplxparams[key][start_idx:start_idx+n_samples]).to(device)


    bm = get_body_model('smplx', str(gender), n_samples, device=device)
    smplx_results = bm(return_verts=True, **smplxparams)
    verts = smplx_results.vertices.detach().cpu().numpy()
    face = bm.faces

    for i in range(n_samples):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts[i])
        mesh.triangles = o3d.utility.Vector3iVector(face)
        mesh.compute_vertex_normals()
        if color is not None:
            mesh.paint_uniform_color(color_hex2rgb(color))   # orange
        body_mesh_list.append(mesh)

    return body_mesh_list, smplx_results

def get_object_mesh(obj, dataset, transl, global_orient, n_samples, device='cpu', rotmat=False):
    object_mesh_list = []
    global_orient_dim = 9 if rotmat else 3

    global_orient = torch.FloatTensor(global_orient).to(device)
    transl = torch.FloatTensor(transl).to(device)

    if dataset == 'GRAB':
        mesh_base = './dataset/contact_meshes'
        obj_mesh_base = o3d.io.read_triangle_mesh(os.path.join(mesh_base, obj + '.ply'))
    # elif dataset == 'FHB':
    #     mesh_base = '/home/dalco/wuyan/data/FHB/Object_models'
    #     obj_mesh_base = o3d.io.read_triangle_mesh(os.path.join(mesh_base, '{}_model/{}_model.ply'.format(obj, obj)))
    # elif dataset == 'HO3D':
    #     mesh_base = '/home/dalco/wuyan/data/HO3D/YCB_Video_Models/models'
    #     obj_mesh_base = o3d.io.read_triangle_mesh(os.path.join(mesh_base, '{}/textured_simple.obj'.format(obj)))
    # elif dataset == 'ShapeNet':
    #     mesh_base = '/home/dalco/wuyan/data/ShapeNet/ShapeNet_selected'
    #     obj_mesh_base = o3d.io.read_triangle_mesh(os.path.join(mesh_base, '{}.obj'.format(obj)))
    #     obj_mesh_base.scale(0.15, center=np.zeros((3, 1)))
    else:
        raise NotImplementedError

    obj_mesh_base.compute_vertex_normals()
    v_temp = torch.FloatTensor(obj_mesh_base.vertices).to(device).view(1, -1, 3).repeat(n_samples, 1, 1)
    normal_temp = torch.FloatTensor(obj_mesh_base.vertex_normals).to(device).view(1, -1, 3).repeat(n_samples, 1, 1)
    obj_model = ObjectModel(v_temp, normal_temp, n_samples)
#     global_orient_dim = 9 if rotmat else 3
    object_output = obj_model(global_orient.view(n_samples, global_orient_dim), transl.view(n_samples, 3), v_temp.to(device), normal_temp.to(device), rotmat)

    object_verts = object_output[0].detach().squeeze().view(n_samples, -1, 3).cpu().numpy()
    # object_vertex_normal = object_output[1].detach().squeeze().cpu().numpy()

    for i in range(n_samples):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(object_verts[i])
        mesh.triangles = obj_mesh_base.triangles
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color_hex2rgb('#f59002'))   # orange
        object_mesh_list.append(mesh)

    return object_mesh_list


def get_forward_direction(joints):
    bs = len(joints)
    x_axis = joints[:, 2, :] - joints[:, 1, :]  # [bs, 3]
    x_axis[:, -1] = 0
    x_axis = x_axis / torch.norm(x_axis, dim=1).unsqueeze(1)
    z_axis = torch.tensor([0, 0, 1]).float().unsqueeze(0).repeat(bs,1)
    y_axis = torch.cross(z_axis, x_axis)
    y_axis = y_axis / torch.norm(y_axis, dim=1).unsqueeze(1)
    transf_rotmat = torch.stack([x_axis, y_axis, z_axis], dim=2)  # [bs, 3, 3]
    return y_axis, transf_rotmat


def get_forward_direction_markers(markers):
    bs = len(markers)
    sdr_l, sdr_r, hip_l, hip_r = 7+1+1, 23+1+1, 8+1+1, 24+1+1  # +1+1: [0]: reference, [1]: pelvis

    across1 = markers[:, hip_r] - markers[:, hip_l]
    across0 = markers[:, sdr_r] - markers[:, sdr_l]

    across = across0 + across1
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]
    forward = -np.cross(across, np.array([[0, 0, 1]]))
    forward = forward / np.sqrt((forward ** 2).sum(axis=-1))[..., np.newaxis]

    return torch.FloatTensor(forward)


def traj_infill(clip_img_input, marker_start, marker_end, joint_start, joint_end, stats):
    '''
    clip_img_input: [bs, 4, d, T] input to cnn model
    marker_start: [bs, 79, 3] markers on the first frame
    joint_start: [bs, 127, 3] joints on the first frame
    '''
    bs, channel, d, T = clip_img_input.shape

    # transfrom to pelvis at origin, face y axis
    pelvis_start_0 = joint_start[:, 0]  # [bs, 3]
    pelvis_end_0 = joint_end[:, 0]  # [bs, 3]
    forward_start, rotmat_start = get_forward_direction(joint_start)
    forward_end, _ = get_forward_direction(joint_end)

    pelvis_end = torch.matmul((pelvis_end_0 - pelvis_start_0).unsqueeze(1), rotmat_start).squeeze(1)
    pelvis_start = torch.matmul((pelvis_start_0 - pelvis_start_0).unsqueeze(1), rotmat_start).squeeze(1)

    marker_end = torch.matmul((marker_end - pelvis_start_0.unsqueeze(1)), rotmat_start).squeeze(1)
    marker_start = torch.matmul((marker_start - pelvis_start_0.unsqueeze(1)), rotmat_start).squeeze(1)

    # markers to local coor.
    marker_end[0:2] = (marker_end - pelvis_end.unsqueeze(1))[0:2]
    marker_start[0:2] = (marker_start - pelvis_start.unsqueeze(1))[0:2]

    forward_start = get_forward_direction_markers(marker_start)
    forward_end = get_forward_direction_markers(marker_end)

    # # need to test whether this pelvis_end is the same as that in dataloader
    # forward_start = torch.matmul(forward_start.unsqueeze(1), rotmat_start).squeeze(1)
    # forward_end = torch.matmul(forward_end.unsqueeze(1), rotmat_start).squeeze(1)

    # root velocity on floor
    velocity = ((pelvis_end - pelvis_start)/60.0).unsqueeze(1).repeat(1, T, 1)  # [bs, T, 3]

    forward = torch.zeros([bs, T+1, 3])
    for ti in range(T+1):
        forward[:, ti, :] = torch.lerp(forward_start, forward_end, ti/(T-1))

    # make sure unit vecter 
    forward = forward / torch.sqrt((forward ** 2).sum(dim=-1)).unsqueeze(-1)
    forward = forward.detach().cpu().numpy()
    velocity = velocity.detach().cpu().numpy()

    """ Remove Y Rotation """
    # swap y/z axis  --> in (x,z,y)
    velocity[:, :, [1, 2]] = velocity[:, :, [2, 1]]
    forward[:, :, [1, 2]] = forward[:, :, [2, 1]]
    target = np.array([[0, 0, 1]])#.repeat(len(forward), axis=0)
    rotation = Quaternions.between(forward, target)
    rotation = rotation[:,:, np.newaxis]
    # ipdb.set_trace()

    """ Get Root Rotation """
    velocity = rotation[:, 1:, 0] * velocity # [bs, T, 3]
    rvelocity = Pivots.from_quaternions(rotation[:, 1:] * -rotation[:, :-1]).ps   # [bs, T, 1]

    rot_0_pivot = Pivots.from_quaternions(rotation[0]).ps   # [bs, T, 1]

    # print(velocity[:, 0, 0], velocity[:, 0, 2], rvelocity[:, 0])

    global_x, global_y = velocity[:, :, 0][:,:, np.newaxis], velocity[:, :, 2][:,:, np.newaxis]  # [bs, T, 1]
    channel_global_x = np.repeat(global_x, d).reshape(bs, 1, d, T)  
    channel_global_y = np.repeat(global_y, d).reshape(bs, 1, d, T) 
    channel_global_r = np.repeat(rvelocity, d).reshape(bs, 1, d, T)

    cur_body = np.concatenate([clip_img_input[:, 0:1].detach().cpu().numpy(), channel_global_x, channel_global_y, channel_global_r], axis=1)  # [bs, 4, T, d]


    cur_body[:, 1:3] = (cur_body[:, 1:3] - stats['Xmean_global_xy']) / stats['Xstd_global_xy']
    cur_body[:, 3] = (cur_body[:, 3] - stats['Xmean_global_r']) / stats['Xstd_global_r']

    clip_img_input_new = torch.from_numpy(cur_body).float().cuda()

    # TODO: compute the overall transformation matrix that changed the original coordinates
    transf_mat = []

    return clip_img_input_new, rot_0_pivot, transf_mat


def get_markers_ids(markers_type):
        f, p = markers_type.split('_')
        finger_n, palm_n = int(f[1:]), int(p[1:])
        with open('./body_utils/smplx_markerset.json') as f:
            markerset = json.load(f)['markersets']
            markers_ids = []
            for marker in markerset:
                if marker['type'] == 'finger' and finger_n == 0:
                    continue
                elif 'palm' in marker['type']:
                    if palm_n == 5 and marker['type'] == 'palm_5':
                        markers_ids += list(marker['indices'].values())
                    elif palm_n == 22 and marker['type'] == 'palm':
                        markers_ids += list(marker['indices'].values())
                    else:
                        continue
                else:
                    markers_ids += list(marker['indices'].values())
        return markers_ids

def get_markers_ids_indiv(markers_type):
        markers_ids = None
        with open('./body_utils/smplx_markerset.json') as f:
            markerset = json.load(f)['markersets']
            markers_ids = []
            for marker in markerset:
                # print(marker['type'])
                if marker['type'] == markers_type:
                    markers_ids = list(marker['indices'].values())
                else:
                    continue
        return markers_ids

def reconstruct_global_joints(body_joints_input, rot_0_pivot):
    root_traj = body_joints_input[:, -1]  # [T-1, 3]
    root_r, root_x, root_z = root_traj[:, 2], root_traj[:, 0], root_traj[:, 1]  # [T-1]
    body_joints_input = body_joints_input[:, 0:-1]  # [T-1, 25+1, 3]
    body_joints_input[:, :, [1, 2]] = body_joints_input[:, :, [2, 1]]
    rotation = Quaternions.id(1)
    offsets = []
    translation = np.array([[0, 0, 0]])
    for i in range(len(body_joints_input)):
        if i == 0:
            rotation = Quaternions.from_angle_axis(-rot_0_pivot, np.array([0, 1, 0])) * rotation   # t=0
        body_joints_input[i, :, :] = rotation * body_joints_input[i]
        body_joints_input[i, :, 0] = body_joints_input[i, :, 0] + translation[0, 0]
        body_joints_input[i, :, 2] = body_joints_input[i, :, 2] + translation[0, 2]
        rotation = Quaternions.from_angle_axis(-root_r[i], np.array([0, 1, 0])) * rotation
        offsets.append(rotation * np.array([0, 0, 1]))
        translation = translation + rotation * np.array([root_x[i], 0, root_z[i]])
    body_joints_input[:, :, [1, 2]] = body_joints_input[:, :, [2, 1]]
    body_joints_input = body_joints_input[:, 1:, :]
    return body_joints_input

def get_global_pose(clip_img_input_new, clip_img_rec, rot_0_pivot, markers_stats):

	body_markers_gt = clip_img_input_new[0][0:-4, :]  # [75, T]
	body_markers_rec = clip_img_rec[0][0:-4, :]
    # global_traj_x = clip_img[0, 1, 0:1]  # [1, T]

	global_traj = torch.cat([clip_img_input_new[1, 0:1], clip_img_input_new[2, 0:1], clip_img_input_new[3, 0:1]], dim=0)  # TO CHECK: should not be noisy[3, T]
	global_traj_input = torch.cat([clip_img_input_new[1, 0:1], 
	                                clip_img_input_new[2, 0:1], 
	                                clip_img_input_new[3, 0:1]], dim=0)  # [3, T], noisy

	body_markers_gt = torch.cat([global_traj, body_markers_gt], dim=0)  # [78, T], global+local
	body_markers_rec = torch.cat([global_traj_input, body_markers_rec], dim=0)

	body_markers_gt = body_markers_gt.permute(1, 0).reshape(body_markers_gt.shape[1], -1, 3).detach().cpu().numpy()  # [T, 81, 3]
	body_markers_rec = body_markers_rec.permute(1, 0).reshape(body_markers_rec.shape[1], -1, 3).detach().cpu().numpy()  # [T, 81, 3]


	body_markers_gt = np.reshape(body_markers_gt, (body_markers_gt.shape[0], -1))
	body_markers_rec = np.reshape(body_markers_rec, (body_markers_rec.shape[0], -1))

	body_markers_gt[:, 3:] = body_markers_gt[:, 3:] * markers_stats['Xstd_local'][0:-4] + markers_stats['Xmean_local'][0:-4]
	body_markers_gt[:, 0:2] = body_markers_gt[:, 0:2] * markers_stats['Xstd_global_xy'] + markers_stats['Xmean_global_xy']
	body_markers_gt[:, 2] = body_markers_gt[:, 2] * markers_stats['Xstd_global_r'] + markers_stats['Xmean_global_r']
	body_markers_gt = np.reshape(body_markers_gt, (body_markers_gt.shape[0], -1, 3))

	body_markers_rec[:, 3:] = body_markers_rec[:, 3:] * markers_stats['Xstd_local'][0:-4] + markers_stats['Xmean_local'][0:-4]
	body_markers_rec[:, 0:2] = body_markers_rec[:, 0:2] * markers_stats['Xstd_global_xy'] + markers_stats['Xmean_global_xy']
	body_markers_rec[:, 2] = body_markers_rec[:, 2] * markers_stats['Xstd_global_r'] + markers_stats['Xmean_global_r']
	body_markers_rec = np.reshape(body_markers_rec, (body_markers_rec.shape[0], -1, 3))

	# back to old format
	pad_0 = np.zeros([body_markers_rec.shape[0], 1, 3])
	body_markers_gt = np.concatenate([pad_0, body_markers_gt[:, 1:], body_markers_gt[:, 0:1]], axis=1)
	body_markers_rec = np.concatenate([pad_0, body_markers_rec[:, 1:], body_markers_rec[:, 0:1]], axis=1)

	body_markers_gt = reconstruct_global_joints(body_markers_gt, rot_0_pivot)
	body_markers_rec = reconstruct_global_joints(body_markers_rec, rot_0_pivot)

	body_markers_rec = body_markers_rec[:, 1:, :]  # remove first pelvis joint   [T, 79, 3]
	body_markers_gt = body_markers_gt[:, 1:, :]  # remove first pelvis joint   [T, 79, 3]

	return body_markers_gt, body_markers_rec
