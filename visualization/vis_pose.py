import argparse
import os
import sys

import numpy as np
import open3d as o3d

from visualization_utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='grabpose-Testing')

    parser.add_argument('--exp_name', default = None, type=str,
                        help='experiment name')

    parser.add_argument('--gender', default = None, type=str,
                        help='gender')

    parser.add_argument('--object', default = None, type=str,
                        help='object name')

    parser.add_argument('--object_format', default = 'mesh', type=str,
                        help='pcd or mesh')

    args = parser.parse_args()


    cwd = os.getcwd()

    load_path = '../results/{}/GraspPose/{}/fitting_results.npz'.format(args.exp_name, args.object)

    data = np.load(load_path, allow_pickle=True)
    gender = args.gender
    object_name = args.object

    n_samples = len(data['markers'])

    # Prepare mesh and pcd
    object_pcd = get_pcd(data['object'][()]['verts_object'][:n_samples])
    object_mesh = get_object_mesh(object_name, 'GRAB', data['object'][()]['transl'][:n_samples], data['object'][()]['global_orient'][:n_samples], n_samples)
    body_mesh, _ = get_body_mesh(data['body'][()], gender, n_samples)


    # ground
    x_range = np.arange(-5, 50, 1)
    y_range = np.arange(-5, 50, 1)
    z_range = np.arange(0, 1, 1)
    gp_lines, gp_pcd = create_lineset(x_range, y_range, z_range)
    gp_lines.paint_uniform_color(color_hex2rgb('#bdbfbe'))   # grey
    gp_pcd.paint_uniform_color(color_hex2rgb('#bdbfbe'))     # grey
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25)

    for i in range(n_samples):
        print(body_mesh[i])
        visualization_list = [body_mesh[i], object_mesh[i], coord, gp_lines, gp_pcd]
        o3d.visualization.draw_geometries(visualization_list)

