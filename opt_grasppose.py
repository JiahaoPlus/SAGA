import argparse
import os
import sys
from collections import defaultdict

import numpy as np
# import smplx
import open3d as o3d
import torch
from smplx.lbs import batch_rodrigues
from tqdm import tqdm

from utils.cfg_parser import Config
from utils.utils import makelogger, makepath
from WholeGraspPose.models.fittingop import FittingOP
from WholeGraspPose.models.objectmodel import ObjectModel
from WholeGraspPose.trainer import Trainer


#### inference
def load_object_data_random(object_name, n_samples):
    mesh_base = './dataset/contact_meshes'
    obj_mesh_base = o3d.io.read_triangle_mesh(os.path.join(mesh_base, object_name + '.ply'))
    obj_mesh_base.compute_vertex_normals()
    v_temp = torch.FloatTensor(obj_mesh_base.vertices).to(grabpose.device).view(1, -1, 3).repeat(n_samples, 1, 1)
    normal_temp = torch.FloatTensor(obj_mesh_base.vertex_normals).to(grabpose.device).view(1, -1, 3).repeat(n_samples, 1, 1)
    obj_model = ObjectModel(v_temp, normal_temp, n_samples)

    """Prepare transl/global_orient data"""
    """Example: randomly sample object height and orientation"""
    transf_transl_list = torch.rand(n_samples) + 0.6   #### can be customized
    global_orient_list = (np.pi)*torch.rand(n_samples) - np.pi/2   #### can be customized
    transl = torch.zeros(n_samples, 3)   # for object model which is centered at object
    transf_transl = torch.zeros(n_samples, 3)
    transf_transl[:, -1] = transf_transl_list
    global_orient = torch.zeros(n_samples, 3)
    global_orient[:, -1] = global_orient_list
    global_orient_rotmat = batch_rodrigues(global_orient.view(-1, 3)).to(grabpose.device)   # [N, 3, 3]

    object_output = obj_model(global_orient_rotmat, transl.to(grabpose.device), v_temp.to(grabpose.device), normal_temp.to(grabpose.device), rotmat=True)
    object_verts = object_output[0].detach().squeeze().cpu().numpy() if n_samples != 1 else object_output[0].detach().cpu().numpy()
    object_normal = object_output[1].detach().squeeze().cpu().numpy() if n_samples != 1 else object_output[1].detach().cpu().numpy()
    
    index = np.linspace(0, object_verts.shape[1], num=2048, endpoint=False,retstep=True,dtype=int)[0]
    
    verts_object = object_verts[:, index]
    normal_object = object_normal[:, index]
    global_orient_rotmat_6d = global_orient_rotmat.view(-1, 1, 9)[:, :, :6].detach().cpu().numpy()
    feat_object = np.concatenate([normal_object, global_orient_rotmat_6d.repeat(2048, axis=1)], axis=-1)
    
    verts_object = torch.from_numpy(verts_object).to(grabpose.device)
    feat_object = torch.from_numpy(feat_object).to(grabpose.device)
    transf_transl = transf_transl.to(grabpose.device)
    return {'verts_object':verts_object, 'normal_object': normal_object, 'global_orient':global_orient, 'global_orient_rotmat':global_orient_rotmat, 'feat_object':feat_object, 'transf_transl':transf_transl}


def load_object_data_uniform_sample(object_name, n_samples):
    mesh_base = './dataset/contact_meshes'
    obj_mesh_base = o3d.io.read_triangle_mesh(os.path.join(mesh_base, object_name + '.ply'))
    obj_mesh_base.compute_vertex_normals()
    v_temp = torch.FloatTensor(obj_mesh_base.vertices).to(grabpose.device).view(1, -1, 3).repeat(n_samples, 1, 1)
    normal_temp = torch.FloatTensor(obj_mesh_base.vertex_normals).to(grabpose.device).view(1, -1, 3).repeat(n_samples, 1, 1)
    obj_model = ObjectModel(v_temp, normal_temp, n_samples)

    """Prepare transl/global_orient data"""
    """Example: uniformly sample object height and orientation (can be customized)"""
    transf_transl_list = torch.arange(n_samples)*1.0/(n_samples-1) + 0.5
    global_orient_list = (2*np.pi)*torch.arange(n_samples)/n_samples
    n_samples = transf_transl_list.shape[0] * global_orient_list.shape[0]
    transl = torch.zeros(n_samples, 3)   # for object model which is centered at object
    transf_transl = torch.zeros(n_samples, 3)
    transf_transl[:, -1] = transf_transl_list.repeat_interleave(global_orient_list.shape[0])
    global_orient = torch.zeros(n_samples, 3)
    global_orient[:, -1] = global_orient_list.repeat(transf_transl_list.shape[0])  # [6+6+6.....]
    global_orient_rotmat = batch_rodrigues(global_orient.view(-1, 3)).to(grabpose.device)   # [N, 3, 3]

    object_output = obj_model(global_orient_rotmat, transl.to(grabpose.device), v_temp.to(grabpose.device), normal_temp.to(grabpose.device), rotmat=True)
    object_verts = object_output[0].detach().squeeze().cpu().numpy() if n_samples != 1 else object_output[0].detach().cpu().numpy()
    object_normal = object_output[1].detach().squeeze().cpu().numpy() if n_samples != 1 else object_output[1].detach().cpu().numpy()
    
    index = np.linspace(0, object_verts.shape[1], num=2048, endpoint=False,retstep=True,dtype=int)[0]
    
    verts_object = object_verts[:, index]
    normal_object = object_normal[:, index]
    global_orient_rotmat_6d = global_orient_rotmat.view(-1, 1, 9)[:, :, :6].detach().cpu().numpy()
    feat_object = np.concatenate([normal_object, global_orient_rotmat_6d.repeat(2048, axis=1)], axis=-1)
    
    verts_object = torch.from_numpy(verts_object).to(grabpose.device)
    feat_object = torch.from_numpy(feat_object).to(grabpose.device)
    transf_transl = transf_transl.to(grabpose.device)
    return {'verts_object':verts_object, 'normal_object': normal_object, 'global_orient':global_orient, 'global_orient_rotmat':global_orient_rotmat, 'feat_object':feat_object, 'transf_transl':transf_transl}

def inference(grabpose, obj, n_samples, n_rand_samples, object_type, save_dir):
    """ prepare test object data: [verts_object, feat_object(normal + rotmat), transf_transl] """
    ### object centered
    # for obj in grabpose.cfg.object_class:
    if object_type == 'uniform':
        obj_data = load_object_data_uniform_sample(obj, n_samples)
    elif object_type == 'random':
        obj_data = load_object_data_random(obj, n_samples)
    obj_data['feat_object'] = obj_data['feat_object'].permute(0,2,1)
    obj_data['verts_object'] = obj_data['verts_object'].permute(0,2,1)

    n_samples_total = obj_data['feat_object'].shape[0]

    markers_gen = []
    object_contact_gen = []
    markers_contact_gen = []
    for i in range(n_samples_total):
        sample_results = grabpose.full_grasp_net.sample(obj_data['verts_object'][None, i].repeat(n_rand_samples,1,1), obj_data['feat_object'][None, i].repeat(n_rand_samples,1,1), obj_data['transf_transl'][None, i].repeat(n_rand_samples,1))
        markers_gen.append((sample_results[0]+obj_data['transf_transl'][None, i]))
        markers_contact_gen.append(sample_results[1])
        object_contact_gen.append(sample_results[2])

    markers_gen = torch.cat(markers_gen, dim=0)   # [B, N, 3]
    object_contact_gen = torch.cat(object_contact_gen, dim=0).squeeze()   # [B, 2048]
    markers_contact_gen = torch.cat(markers_contact_gen, dim=0)   # [B, N]

    output = {}
    output['markers_gen'] = markers_gen.detach().cpu().numpy()
    output['markers_contact_gen'] = markers_contact_gen.detach().cpu().numpy()
    output['object_contact_gen'] = object_contact_gen.detach().cpu().numpy()
    output['normal_object'] = obj_data['normal_object']#.repeat(n_rand_samples, axis=0)
    output['transf_transl'] = obj_data['transf_transl'].detach().cpu().numpy()#.repeat(n_rand_samples, axis=0)
    output['global_orient_object'] = obj_data['global_orient'].detach().cpu().numpy()#.repeat(n_rand_samples, axis=0)
    output['global_orient_object_rotmat'] = obj_data['global_orient_rotmat'].detach().cpu().numpy()#.repeat(n_rand_samples, axis=0)
    output['verts_object'] = (obj_data['verts_object']+obj_data['transf_transl'].view(-1,3,1).repeat(1,1,2048)).permute(0, 2, 1).detach().cpu().numpy()#.repeat(n_rand_samples, axis=0)

    save_path = os.path.join(save_dir, 'markers_results.npy')
    np.save(save_path, output)
    print('Saving to {}'.format(save_path))

    return output

def fitting_data_save(save_data,
              markers,
              markers_fit,
              smplxparams,
              gender,
              object_contact, body_contact,
              object_name, verts_object, global_orient_object, transf_transl_object):
    # markers & markers_fit
    save_data['markers'].append(markers)
    save_data['markers_fit'].append(markers_fit)
    # print('markers:', markers.shape)

    # body params
    for key in save_data['body'].keys():
        # print(key, smplxparams[key].shape)
        save_data['body'][key].append(smplxparams[key].detach().cpu().numpy())
    # object name & object params
    save_data['object_name'].append(object_name)
    save_data['gender'].append(gender)
    save_data['object']['transl'].append(transf_transl_object)
    save_data['object']['global_orient'].append(global_orient_object)
    save_data['object']['verts_object'].append(verts_object)

    # contact
    save_data['contact']['body'].append(body_contact)
    save_data['contact']['object'].append(object_contact)

#### fitting

def pose_opt(grabpose, samples_results, n_random_samples, obj, gender, save_dir, logger, device):
    # prepare objects
    n_samples = len(samples_results['verts_object'])
    verts_object = torch.tensor(samples_results['verts_object'])[:n_samples].to(device)  # (n, 2048, 3)
    normals_object = torch.tensor(samples_results['normal_object'])[:n_samples].to(device)  # (n, 2048, 3)
    global_orients_object = torch.tensor(samples_results['global_orient_object'])[:n_samples].to(device)  # (n, 2048, 3)
    transf_transl_object = torch.tensor(samples_results['transf_transl'])[:n_samples].to(device)  # (n, 2048, 3)

    # prepare body markers
    markers_gen = torch.tensor(samples_results['markers_gen']).to(device)  # (n*k, 143, 3)
    object_contacts_gen = torch.tensor(samples_results['object_contact_gen']).view(markers_gen.shape[0], -1, 1).to(device)  #  (n, 2048, 1)
    markers_contacts_gen = torch.tensor(samples_results['markers_contact_gen']).view(markers_gen.shape[0], -1, 1).to(device)   #  (n, 143, 1)

    print('Fitting {} {} samples for {}...'.format(n_samples, cfg.gender, obj.upper()))

    fittingconfig={ 'init_lr_h': 0.008,
                            'num_iter': [300,400,500],
                            'batch_size': 1,
                            'num_markers': 143,
                            'device': device,
                            'cfg': cfg,
                            'verbose': False,
                            'hand_ncomps': 24,
                            'only_rec': False,     # True / False 
                            'contact_loss': 'contact',  # contact / prior / False
                            'logger': logger,
                            'data_type': 'markers_143',
                            }
    fittingop = FittingOP(fittingconfig)

    save_data_gen = {}
    for data in [save_data_gen]:
        data['markers'] = []
        data['markers_fit'] = []
        data['body'] = {}
        for key in ['betas', 'transl', 'global_orient', 'body_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose']:
            data['body'][key] = []
        data['object'] = {}
        for key in ['transl', 'global_orient', 'verts_object']:
            data['object'][key] = []
        data['contact'] = {}
        for key in ['body', 'object']:
            data['contact'][key] = []
        data['gender'] = []
        data['object_name'] = []


    for i in tqdm(range(n_samples)):
        # prepare object 
        vert_object = verts_object[None, i, :, :]
        normal_object = normals_object[None, i, :, :]

        marker_gen = markers_gen[i*n_random_samples:(i+1)*n_random_samples, :, :]
        object_contact_gen = object_contacts_gen[i*n_random_samples:(i+1)*n_random_samples, :, :]
        markers_contact_gen = markers_contacts_gen[i*n_random_samples:(i+1)*n_random_samples, :, :]

        for k in range(n_random_samples):
            print('Fitting for {}-th GEN...'.format(k+1))
            markers_fit_gen, smplxparams_gen, loss_gen = fittingop.fitting(marker_gen[None, k, :], object_contact_gen[None, k, :], markers_contact_gen[None, k], vert_object, normal_object, gender)
            fitting_data_save(save_data_gen,
                    marker_gen[k, :].detach().cpu().numpy().reshape(1, -1 ,3),
                    markers_fit_gen[-1].squeeze().reshape(1, -1 ,3),
                    smplxparams_gen[-1],
                    gender,
                    object_contact_gen[k].detach().cpu().numpy().reshape(1, -1), markers_contact_gen[k].detach().cpu().numpy().reshape(1, -1),
                    obj, vert_object.detach().cpu().numpy(), global_orients_object[i].detach().cpu().numpy(), transf_transl_object[i].detach().cpu().numpy())


    for data in [save_data_gen]:
        # for data in [save_data_gt, save_data_rec, save_data_gen]:
            data['markers'] = np.vstack(data['markers'])  
            data['markers_fit'] = np.vstack(data['markers_fit'])
            for key in ['betas', 'transl', 'global_orient', 'body_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose']:
                data['body'][key] = np.vstack(data['body'][key])
            for key in ['transl', 'global_orient', 'verts_object']:
                data['object'][key] = np.vstack(data['object'][key])
            for key in ['body', 'object']:
                data['contact'][key] = np.vstack(data['contact'][key])

    np.savez(os.path.join(save_dir, 'fitting_results.npz'), **save_data_gen)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='grabpose-Testing')

    parser.add_argument('--data_path', default = './dataset/GraspPose', type=str,
                        help='The path to the folder that contains grabpose data')

    parser.add_argument('--object', default = None, type=str,
                        help='object name')

    parser.add_argument('--gender', default=None, type=str,
                        help='The gender of dataset')

    parser.add_argument('--config_path', default = None, type=str,
                        help='The path to the confguration of the trained grabpose model')

    parser.add_argument('--exp_name', default = None, type=str,
                        help='experiment name')

    parser.add_argument('--pose_ckpt_path', default = None, type=str,
                        help='checkpoint path')

    parser.add_argument('--n_object_samples', default = 5, type=int,
                        help='The number of object samples of this object')

    parser.add_argument('--type_object_samples', default = 'random', type=str,
                        help='For the given object mesh, we provide two types of object heights and orientation sampling mode: random / uniform')

    parser.add_argument('--n_rand_samples_per_object', default = 1, type=int,
                        help='The number of whole-body poses random samples generated per object')

    args = parser.parse_args()

    cwd = os.getcwd()

    best_net = os.path.join(cwd, args.pose_ckpt_path)

    vpe_path  = '/configs/verts_per_edge.npy'
    c_weights_path = cwd + '/WholeGraspPose/configs/rhand_weight.npy'
    work_dir = cwd + '/results/{}/GraspPose'.format(args.exp_name)
    print(work_dir)
    config = {
        'dataset_dir': args.data_path,
        'work_dir':work_dir,
        'vpe_path': vpe_path,
        'c_weights_path': c_weights_path,
        'exp_name': args.exp_name,
        'gender': args.gender,
        'best_net': best_net
    }

    cfg_path = 'WholeGraspPose/configs/WholeGraspPose.yaml'
    cfg = Config(default_cfg_path=cfg_path, **config)

    save_dir = os.path.join(work_dir, args.object)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)        

    logger = makelogger(makepath(os.path.join(save_dir, '%s.log' % (args.object)), isfile=True)).info
    
    grabpose = Trainer(cfg=cfg, inference=True, logger=logger)
    
    samples_results = inference(grabpose, args.object, args.n_object_samples, args.n_rand_samples_per_object, args.type_object_samples, save_dir)
    fitting_results = pose_opt(grabpose, samples_results, args.n_rand_samples_per_object, args.object, cfg.gender, save_dir, logger, grabpose.device)

