import os
import sys

import torch

sys.path.append(os.getcwd())
from utils.Quaternions_torch import Quaternions_torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Pivots_torch:    
    """
    Pivots is an ndarray of angular rotations

    This wrapper provides some functions for
    working with pivots.

    These are particularly useful as a number 
    of atomic operations (such as adding or 
    subtracting) cannot be achieved using
    the standard arithmatic and need to be
    defined differently to work correctly
    """
    
    def __init__(self, ps): self.ps = torch.tensor(ps).to(device)
    def __str__(self): return "Pivots("+ str(self.ps) + ")"
    def __repr__(self): return "Pivots("+ repr(self.ps) + ")"
    
    def __add__(self, other): return Pivots_torch(torch.atan2(torch.sin(self.ps + other.ps), torch.cos(self.ps + other.ps)))
    def __sub__(self, other): return Pivots_torch(torch.atan2(torch.sin(self.ps - other.ps), torch.cos(self.ps - other.ps)))
    def __mul__(self, other): return Pivots_torch(self.ps  * other.ps)
    def __div__(self, other): return Pivots_torch(self.ps  / other.ps)
    def __mod__(self, other): return Pivots_torch(self.ps  % other.ps)
    def __pow__(self, other): return Pivots_torch(self.ps ** other.ps)
    
    def __lt__(self, other): return self.ps <  other.ps
    def __le__(self, other): return self.ps <= other.ps
    def __eq__(self, other): return self.ps == other.ps
    def __ne__(self, other): return self.ps != other.ps
    def __ge__(self, other): return self.ps >= other.ps
    def __gt__(self, other): return self.ps >  other.ps
    
    def __abs__(self): return Pivots_torch(torch.abs(self.ps))
    def __neg__(self): return Pivots_torch(-self.ps)
    
    def __iter__(self): return iter(self.ps)
    def __len__(self): return len(self.ps)
    
    def __getitem__(self, k):    return Pivots_torch(self.ps[k]) 
    def __setitem__(self, k, v): self.ps[k] = v.ps
    
    def _ellipsis(self): return tuple(map(lambda x: slice(None), self.shape))
    
    def quaternions(self, plane='xz'):
        fa = self._ellipsis()
        axises = torch.ones(self.ps.shape + (3,)).to(device)
        axises[fa + ("xyz".index(plane[0]),)] = 0.0
        axises[fa + ("xyz".index(plane[1]),)] = 0.0
        return Quaternions_torch.from_angle_axis(self.ps, axises)
    
    def directions(self, plane='xz'):
        dirs = torch.zeros((len(self.ps), 3)).to(device)
        dirs["xyz".index(plane[0])] = torch.sin(self.ps)
        dirs["xyz".index(plane[1])] = torch.cos(self.ps)
        return dirs
    
    def normalized(self):
        xs = self.ps.clone()
        while torch.any(xs >  torch.pi): xs[xs >  torch.pi] = xs[xs >  torch.pi] - 2 * torch.pi
        while torch.any(xs < -torch.pi): xs[xs < -torch.pi] = xs[xs < -torch.pi] + 2 * torch.pi
        return Pivots_torch(xs)
    
    # def interpolate(self, ws):
    #     dir = np.average(self.directions, weights=ws, axis=0)
    #     return torch.atan2(dir[2], dir[0])
    
    def clone(self):
        return Pivots_torch((self.ps).clone())
    
    @property
    def shape(self):
        return self.ps.shape
    
    @classmethod
    def from_quaternions(cls, qs, forward='z', plane='xz'):
        ds = torch.zeros(qs.shape + (3,)).to(device)
        ds[...,'xyz'.index(forward)] = 1.0
        return Pivots_torch.from_directions(qs * ds, plane=plane)
        
    @classmethod
    def from_directions(cls, ds, plane='xz'):
        ys = ds[...,'xyz'.index(plane[0])]
        xs = ds[...,'xyz'.index(plane[1])]
        return Pivots_torch(torch.atan2(ys, xs))
    
