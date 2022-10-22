import sys

sys.path.append('..')
import os

import numpy as np
import open3d as o3d
import smplx
import torch
from WholeGraspPose.models.objectmodel import ObjectModel

def update_cam(cam_param, trans):
    cam_R = np.transpose(trans[:-1, :-1])
    cam_T = -trans[:-1, -1:]
    cam_T = np.matmul(cam_R, cam_T)  # !!!!!! T is applied in the rotated coord
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
    for ii in range( x_range.shape[0]-1):
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

def get_body_model(type, gender, batch_size,device='cuda',v_template=None):
    '''
    type: smpl, smplx smplh and others. Refer to smplx tutorial
    gender: male, female, neutral
    batch_size: an positive integar
    '''
    body_model_path = '../body_utils/body_models'
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

def get_body_mesh(smplxparams, gender, n_samples, device='cpu', color=None):
    body_mesh_list = []

    for key in smplxparams.keys():
        # print(key, smplxparams[key].shape)
        smplxparams[key] = torch.tensor(smplxparams[key][:n_samples]).to(device)


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
        mesh_base = '../dataset/contact_meshes'
        # mesh_base = '/home/dalco/wuyan/data/GRAB/tools/object_meshes/contact_meshes'
        obj_mesh_base = o3d.io.read_triangle_mesh(os.path.join(mesh_base, obj + '.ply'))
    elif dataset == 'FHB':
        mesh_base = '/home/dalco/wuyan/data/FHB/Object_models'
        obj_mesh_base = o3d.io.read_triangle_mesh(os.path.join(mesh_base, '{}_model/{}_model.ply'.format(obj, obj)))
    elif dataset == 'HO3D':
        mesh_base = '/home/dalco/wuyan/data/HO3D/YCB_Video_Models/models'
        obj_mesh_base = o3d.io.read_triangle_mesh(os.path.join(mesh_base, '{}/textured_simple.obj'.format(obj)))
    elif dataset == 'ShapeNet':
        mesh_base = '/home/dalco/wuyan/data/ShapeNet/ShapeNet_selected'
        obj_mesh_base = o3d.io.read_triangle_mesh(os.path.join(mesh_base, '{}.obj'.format(obj)))
        obj_mesh_base.scale(0.15, center=np.zeros((3, 1)))
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


# def get_object_mesh(obj, transl, global_orient, n_samples, device='cpu'):
#     object_mesh_list = []

#     global_orient = torch.FloatTensor(global_orient).to(device)
#     transl = torch.FloatTensor(transl).to(device)

#     mesh_base = '../dataset/contact_meshes'
#     obj_mesh_base = o3d.io.read_triangle_mesh(os.path.join(mesh_base, obj + '.ply'))
#     obj_mesh_base.compute_vertex_normals()
#     v_temp = torch.FloatTensor(obj_mesh_base.vertices).to(device).view(1, -1, 3).repeat(n_samples, 1, 1)
#     normal_temp = torch.FloatTensor(obj_mesh_base.vertex_normals).to(device).view(1, -1, 3).repeat(n_samples, 1, 1)
#     obj_model = ObjectModel(v_temp, normal_temp, n_samples)
#     object_output = obj_model(global_orient.view(n_samples, 3), transl.view(n_samples, 3), v_temp.to(device), normal_temp.to(device))

#     object_verts = object_output[0].detach().squeeze().cpu().numpy()
#     # object_vertex_normal = object_output[1].detach().squeeze().cpu().numpy()

#     for i in range(n_samples):
#         mesh = o3d.geometry.TriangleMesh()
#         print('debug:', object_verts[i].shape)
#         mesh.vertices = o3d.utility.Vector3dVector(object_verts[i])
#         mesh.triangles = obj_mesh_base.triangles
#         mesh.compute_vertex_normals()
#         mesh.paint_uniform_color(color_hex2rgb('#f59002'))   # orange
#         object_mesh_list.append(mesh)

#     return object_mesh_list

def get_pcd(points, contact=None):
    print(points.shape)
    pcd_list = []
    n_samples = points.shape[0]
    for i in range(n_samples):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[i])
        if contact is not None:
            colors = np.zeros((points.shape[1], 3))
            colors[:, 0] = contact[i].squeeze()
            pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd_list.append(pcd)
    return pcd_list

