from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from smplx.lbs import batch_rodrigues

model_output = namedtuple('output', ['vertices', 'vertex_normals', 'global_orient', 'transl'])

class ObjectModel(nn.Module):

    def __init__(self,
                 v_template,
                 normal_template,
                 batch_size=1,
                 dtype=torch.float32):

        super(ObjectModel, self).__init__()


        self.dtype = dtype
        self.batch_size = batch_size


    def forward(self, global_orient=None, transl=None, v_template=None, n_template=None, rotmat=False, **kwargs):
        

        if global_orient is None:
            global_orient = self.global_orient
        if transl is None:
            transl = self.transl
        if v_template is None:
            v_template = self.v_template
        if n_template is None:
            n_template = self.n_template

        if not rotmat:
            rot_mats = batch_rodrigues(global_orient.view(-1, 3)).view([self.batch_size, 3, 3])
        else:
            rot_mats = global_orient.view([self.batch_size, 3, 3])

        vertices = torch.matmul(v_template, rot_mats) + transl.unsqueeze(dim=1)

        vertex_normals = torch.matmul(n_template, rot_mats)

        output = model_output(vertices=vertices,
                              vertex_normals = vertex_normals,
                              global_orient=global_orient,
                              transl=transl)

        return output

