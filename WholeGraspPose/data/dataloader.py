import gc
import json
import os
import time

import numpy as np
import torch
from smplx.lbs import batch_rodrigues
from torch.utils import data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()


class LoadData(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 ds_name='train',
                 gender=None,
                 motion_intent=None,
                 object_class=['all'],
                 dtype=torch.float32,
                 data_type = 'markers_143'):

        super().__init__()

        print('Preparing {} data...'.format(ds_name.upper()))
        self.sbj_idxs = []
        self.objs_frames = {}
        self.ds_path = os.path.join(dataset_dir, ds_name)
        self.gender = gender
        self.motion_intent = motion_intent
        self.object_class = object_class
        self.data_type = data_type

        with open('body_utils/smplx_markerset.json') as f:
            markerset = json.load(f)['markersets']
            self.markers_idx = []
            for marker in markerset:
                if marker['type'] not in ['palm_5']:   # 'palm_5' contains selected 5 markers per palm, but for training we use 'palm' set where there are 22 markers per palm. 
                    self.markers_idx += list(marker['indices'].values())
        print(len(self.markers_idx))
        self.ds = self.load_full_data(self.ds_path)

    def load_full_data(self, path):
        rec_list = []
        output = {}

        markers_list = []
        transf_transl_list = []
        verts_object_list = []
        contacts_object_list = []
        normal_object_list = []
        transl_object_list = []
        global_orient_object_list = []
        rotmat_list = []
        contacts_markers_list = []
        body_list = {}
        for key in ['transl', 'global_orient', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose', 'expression']:
            body_list[key] = []
            
        subsets_dict = {'male':['s1', 's2', 's8', 's9', 's10'],
                       'female': ['s3', 's4', 's5', 's6', 's7']}
        subsets = subsets_dict[self.gender]

        print('loading {} dataset: {}'.format(self.gender, subsets))
        for subset in subsets:
            subset_path = os.path.join(path, subset)
            rec_list += [os.path.join(subset_path, i) for i in os.listdir(subset_path)]

        index = 0

        for rec in rec_list:
            data = np.load(rec, allow_pickle=True)

            ## select object
            obj_name = rec.split('/')[-1].split('_')[0]
            if 'all' not in self.object_class:
                if obj_name not in self.object_class:
                    continue

            verts_object_list.append(data['verts_object'])
            markers_list.append(data[self.data_type])
            transf_transl_list.append(data['transf_transl'])
            normal_object_list.append(data['normal_object'])
            global_orient_object_list.append(data['global_orient_object'])

            orient = torch.tensor(data['global_orient_object'])
            rot_mats = batch_rodrigues(orient.view(-1, 3)).view([orient.shape[0], 9]).numpy()
            rotmat_list.append(rot_mats)

            object_contact = data['contact_object']
            markers_contact = data['contact_body'][:, self.markers_idx]
            object_contact_binary = (object_contact>0).astype(int)
            contacts_object_list.append(object_contact_binary)
            markers_contact_binary = (markers_contact>0).astype(int)
            contacts_markers_list.append(markers_contact_binary)

            # SMPLX parameters (optional)
            for key in data['body'][()].keys():
                body_list[key].append(data['body'][()][key])

            sbj_id = rec.split('/')[-2]
            self.sbj_idxs += [sbj_id]*data['verts_object'].shape[0]
            if obj_name in self.objs_frames.keys():
                self.objs_frames[obj_name] += list(range(index, index+data['verts_object'].shape[0]))
            else:
                self.objs_frames[obj_name] = list(range(index, index+data['verts_object'].shape[0]))
            index += data['verts_object'].shape[0]
        output['transf_transl'] = torch.tensor(np.concatenate(transf_transl_list, axis=0))
        output['markers'] = torch.tensor(np.concatenate(markers_list, axis=0))              # (B, 99, 3)
        output['verts_object'] = torch.tensor(np.concatenate(verts_object_list, axis=0))    # (B, 2048, 3)
        output['contacts_object'] = torch.tensor(np.concatenate(contacts_object_list, axis=0))    # (B, 2048, 3)
        output['contacts_markers'] = torch.tensor(np.concatenate(contacts_markers_list, axis=0))    # (B, 2048, 3)
        output['normal_object'] = torch.tensor(np.concatenate(normal_object_list, axis=0))    # (B, 2048, 3)
        output['global_orient_object'] = torch.tensor(np.concatenate(global_orient_object_list, axis=0))    # (B, 2048, 3)
        output['rotmat'] = torch.tensor(np.concatenate(rotmat_list, axis=0))    # (B, 2048, 3)

        # SMPLX parameters
        output['smplxparams'] = {}
        for key in ['transl', 'global_orient', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose', 'expression']:
            output['smplxparams'][key] = torch.tensor(np.concatenate(body_list[key], axis=0))

        return output

    def __len__(self):
        k = list(self.ds.keys())[0]
        return self.ds[k].shape[0]

    def __getitem__(self, idx):
        
        data_out = {}

        data_out['markers'] = self.ds['markers'][idx]
        data_out['contacts_markers'] = self.ds['contacts_markers'][idx]
        data_out['verts_object'] = self.ds['verts_object'][idx]
        data_out['normal_object'] = self.ds['normal_object'][idx]
        data_out['global_orient_object'] = self.ds['global_orient_object'][idx]
        data_out['transf_transl'] = self.ds['transf_transl'][idx]
        data_out['contacts_object'] = self.ds['contacts_object'][idx]
        if len(data_out['verts_object'].shape) == 2:
            data_out['feat_object'] = torch.cat([self.ds['normal_object'][idx], self.ds['rotmat'][idx, :6].view(1, 6).repeat(2048, 1)], -1)
        else:
            data_out['feat_object'] = torch.cat([self.ds['normal_object'][idx], self.ds['rotmat'][idx, :6].view(-1, 1, 6).repeat(1, 2048, 1)], -1)

        """You may want to uncomment it when you need smplxparams!!!"""
        data_out['smplxparams'] = {}
        for key in ['transl', 'global_orient', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose', 'expression']:
            data_out['smplxparams'][key] = self.ds['smplxparams'][key][idx]

        ## random rotation augmentation
        bsz = 1
        theta = torch.FloatTensor(np.random.uniform(-np.pi/6, np.pi/6, bsz))
        orient = torch.zeros((bsz, 3))
        orient[:, -1] = theta
        rot_mats = batch_rodrigues(orient.view(-1, 3)).view([bsz, 3, 3])
        if len(data_out['verts_object'].shape) == 3:
            data_out['markers'] = torch.matmul(data_out['markers'][:, :, :3], rot_mats.squeeze())
            data_out['verts_object'] = torch.matmul(data_out['verts_object'][:, :, :3], rot_mats.squeeze())
            data_out['normal_object'][:, :, :3] = torch.matmul(data_out['normal_object'][:, :, :3], rot_mats.squeeze())
        else:
            data_out['markers'] = torch.matmul(data_out['markers'][:, :3], rot_mats.squeeze())
            data_out['verts_object'] = torch.matmul(data_out['verts_object'][:, :3], rot_mats.squeeze())
            data_out['normal_object'][:, :3] = torch.matmul(data_out['normal_object'][:, :3], rot_mats.squeeze())

        return data_out
