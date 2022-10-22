import os
import sys

sys.path.append(os.getcwd())
import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from tqdm import tqdm

from visualization_utils import (color_hex2rgb, create_lineset, get_body_mesh,
                                 get_object_mesh, update_cam)


def vis_graspmotion_third_view(body_meshes, object_mesh, object_transl, sample_index):
	mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0])

	vis = o3d.visualization.Visualizer()
	vis.create_window()
	# vis.add_geometry(mesh_frame)
	render_opt=vis.get_render_option()
	render_opt.mesh_show_back_face=True
	render_opt.line_width=10
	render_opt.point_size=5
	render_opt.background_color = color_hex2rgb('#1c2434')

	vis.add_geometry(object_mesh)

	x_range = np.arange(-200, 200, 0.75)
	y_range = np.arange(-200, 200, 0.75)
	z_range = np.arange(0, 1, 1)
	gp_lines, gp_pcd = create_lineset(x_range, y_range, z_range)
	gp_lines.paint_uniform_color(color_hex2rgb('#7ea4ab'))
	gp_pcd.paint_uniform_color(color_hex2rgb('#7ea4ab'))
	vis.add_geometry(gp_lines)
	vis.poll_events()
	vis.update_renderer()
	vis.add_geometry(gp_pcd)
	vis.poll_events()
	vis.update_renderer()

	for t in range(len(body_meshes)):
		vis.add_geometry(body_meshes[t])

		### get cam R
		### update render cam
		ctr = vis.get_view_control()
		cam_param = ctr.convert_to_pinhole_camera_parameters()
		trans = np.eye(4)
		trans[:3, :3] = np.array([[0, 0, -1], [-1, 0, 0], [0, -1, 0]])
		trans[:3, -1] = np.array([4, 1, 1])
		cam_param = update_cam(cam_param, trans)
		ctr.convert_from_pinhole_camera_parameters(cam_param)
		vis.poll_events()
		vis.update_renderer()

		vis.capture_screen_image(
		    vis_save_path_third_view+"/clip_%04d_%04d.jpg" % (sample_index, t), True)
		vis.remove_geometry(body_meshes[t])


def vis_graspmotion_first_view(body_meshes, object_mesh, object_transl, sample_index):
	mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0])

	vis = o3d.visualization.Visualizer()
	vis.create_window()
	render_opt=vis.get_render_option()
	render_opt.mesh_show_back_face=True
	render_opt.line_width=10
	render_opt.point_size=5

	render_opt.background_color = color_hex2rgb('#545454')

	vis.add_geometry(object_mesh)

	for t in range(len(body_meshes)):
		vis.add_geometry(body_meshes[t])

		### get cam R
		cam_o = np.array(body_meshes[t].vertices)[8999]
		cam_z =  object_transl - cam_o
		cam_z = cam_z / np.linalg.norm(cam_z)
		cam_x = np.array([cam_z[1], -cam_z[0], 0.0])
		cam_x = cam_x / np.linalg.norm(cam_x)
		cam_y = np.array([cam_z[0], cam_z[1], -(cam_z[0]**2 + cam_z[1]**2)/cam_z[2] ])
		cam_y = cam_y / np.linalg.norm(cam_y)
		cam_r = np.stack([cam_x, -cam_y, cam_z], axis=1)
		### update render cam
		ctr = vis.get_view_control()
		cam_param = ctr.convert_to_pinhole_camera_parameters()
		transf = np.eye(4)
		transf[:3,:3]=cam_r
		transf[:3,-1] = cam_o
		cam_param = update_cam(cam_param, transf)
		ctr.convert_from_pinhole_camera_parameters(cam_param)
		vis.poll_events()
		vis.update_renderer()

		vis.capture_screen_image(
		    vis_save_path_first_view+"/clip_%04d_%04d.jpg" % (sample_index, t), True)
		vis.remove_geometry(body_meshes[t])


def vis_graspmotion_top_view(body_meshes, object_mesh, object_transl, sample_index):
	mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.5, origin=[0, 0, 0])

	vis = o3d.visualization.Visualizer()
	vis.create_window()
	render_opt=vis.get_render_option()
	render_opt.mesh_show_back_face=True
	render_opt.line_width=10
	render_opt.point_size=5
	render_opt.background_color = color_hex2rgb('#1c2434')

	vis.add_geometry(object_mesh)

	x_range = np.arange(-200, 200, 0.75)
	y_range = np.arange(-200, 200, 0.75)
	z_range = np.arange(0, 1, 1)
	gp_lines, gp_pcd = create_lineset(x_range, y_range, z_range)
	gp_lines.paint_uniform_color(color_hex2rgb('#7ea4ab'))
	gp_pcd.paint_uniform_color(color_hex2rgb('#7ea4ab'))
	vis.add_geometry(gp_lines)
	vis.poll_events()
	vis.update_renderer()
	vis.add_geometry(gp_pcd)
	vis.poll_events()
	vis.update_renderer()

	for t in range(len(body_meshes)):
		vis.add_geometry(body_meshes[t])

		ctr = vis.get_view_control()
		cam_param = ctr.convert_to_pinhole_camera_parameters()

		cam_o = np.array([0, 0, 4])
		reference_point = np.zeros(3)
		reference_point[:2] = object_transl[:2]/2
		cam_z =  reference_point - cam_o
		cam_z = cam_z / np.linalg.norm(cam_z)
		cam_x = np.array([cam_z[1], -cam_z[0], 0.0])
		cam_x = cam_x / np.linalg.norm(cam_x)
		cam_y = np.array([cam_z[0], cam_z[1], -(cam_z[0]**2 + cam_z[1]**2)/cam_z[2] ])
		cam_y = cam_y / np.linalg.norm(cam_y)
		cam_r = np.stack([cam_x, -cam_y, cam_z], axis=1)
		### update render cam
		ctr = vis.get_view_control()
		cam_param = ctr.convert_to_pinhole_camera_parameters()
		transf = np.eye(4)
		transf[:3,:3]=cam_r
		transf[:3,-1] = cam_o


		cam_param = update_cam(cam_param, transf)
		ctr.convert_from_pinhole_camera_parameters(cam_param)
		vis.poll_events()
		vis.update_renderer()

		vis.capture_screen_image(
		    vis_save_path_top_view+"/clip_%04d_%04d.jpg" % (sample_index, t), True)
		vis.remove_geometry(body_meshes[t])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='visualization from saved')

    parser.add_argument('--GraspPose_exp_name', type=str, help='exp name')
    parser.add_argument('--dataset', default='GRAB', type=str, help='exp name')
    parser.add_argument('--object', default='camera', type=str, help='object name')
    parser.add_argument('--gender', type=str, help='object name')

    args = parser.parse_args()

    result_path = '../results/{}/GraspMotion/{}'.format(args.GraspPose_exp_name, args.object)
    opt_results = np.load('{}/fitting_results.npy'.format(result_path), allow_pickle=True)[()]

    print('Saving visualization results to {}').format(result_path)

    vis_save_path_first_view = os.path.join(result_path, 'visualization/first_view')
    vis_save_path_third_view = os.path.join(result_path, 'visualization/third_view')
    vis_save_path_top_view = os.path.join(result_path, 'visualization/top_view')
    if not os.path.exists(vis_save_path_first_view):
        os.makedirs(vis_save_path_first_view)
    if not os.path.exists(vis_save_path_third_view):
        os.makedirs(vis_save_path_third_view)
    if not os.path.exists(vis_save_path_top_view):
        os.makedirs(vis_save_path_top_view)

    object_params = opt_results['object']   # Tensor
    object_name = str(opt_results['object_name'])
    body_orig = opt_results['body_orig']  # Numpy
    body_opt = opt_results['body_opt']

    for key in body_orig:
        body_orig[key] = torch.from_numpy(body_orig[key])
        body_opt[key] = torch.from_numpy(body_opt[key])

    B, T, _ = body_orig['transl'].shape

    object_mesh = get_object_mesh(object_name, args.dataset, object_params['transl'][:B], object_params['global_orient'][:B], B, rotmat=True)

    # frame by frame visualization
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)

    x_range = np.arange(-200, 200, 0.75)
    y_range = np.arange(-200, 200, 0.75)
    z_range = np.arange(0, 1, 1)
    gp_lines, gp_pcd = create_lineset(x_range, y_range, z_range)

    collision_eval_list = []

    camera_point_index = 8999

    pos_list, vel_list, acc_list = [], [], []

    for i in tqdm(range(0, B)):
        collision_eval_T = {}
        collision_eval_T['vol'] = []
        collision_eval_T['depth'] = []
        collision_eval_T['contact'] = []
        # get body mesh
        orig_smplxparams = {}
        opt_smplxparams = {}
        for k in body_orig.keys():
            orig_smplxparams[k] = body_orig[k][i]
            opt_smplxparams[k] = body_opt[k][i]
        body_meshes_orig, _ = get_body_mesh(orig_smplxparams, args.gender, n_samples=T, device='cpu', color='D4BEA3')
        body_meshes_opt, _ = get_body_mesh(opt_smplxparams, args.gender, n_samples=T, device='cpu', color='D4BEA3')


        body_meshes = body_meshes_opt
        
        vis_graspmotion_first_view(body_meshes, object_mesh[i], object_params['transl'][i], i)
        vis_graspmotion_top_view(body_meshes, object_mesh[i], object_params['transl'][i], i)
        vis_graspmotion_third_view(body_meshes, object_mesh[i], object_params['transl'][i], i)
