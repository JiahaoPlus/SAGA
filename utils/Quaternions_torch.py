import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Quaternions_torch:
    """
    Quaternions is a wrapper around a numpy ndarray
    that allows it to act as if it were an narray of
    a quaternion data type.
    
    Therefore addition, subtraction, multiplication,
    division, negation, absolute, are all defined
    in terms of quaternion operations such as quaternion
    multiplication.
    
    This allows for much neater code and many routines
    which conceptually do the same thing to be written
    in the same way for point data and for rotation data.
    
    The Quaternions class has been desgined such that it
    should support broadcasting and slicing in all of the
    usual ways.
    """
    
    def __init__(self, qs):
        if isinstance(qs, torch.Tensor):
        
            if len(qs.shape) == 1: qs = torch.tensor([qs])
            self.qs = qs
            return
            
        if isinstance(qs, Quaternions_torch):
            self.qs = qs.qs
            return
            
        raise TypeError('Quaternions must be constructed from iterable, numpy array, or Quaternions, not %s' % type(qs))
    
    def __str__(self): return "Quaternions("+ str(self.qs) + ")"
    def __repr__(self): return "Quaternions("+ repr(self.qs) + ")"
    
    """ Helper Methods for Broadcasting and Data extraction """
    
    @classmethod
    def _broadcast(cls, sqs, oqs, scalar=False):
        
        if isinstance(oqs, float): return sqs, oqs * torch.ones(sqs.shape[:-1])
        
        ss = torch.tensor(sqs.shape).to(device) if not scalar else torch.tensor(sqs.shape[:-1]).to(device)
        os = torch.tensor(oqs.shape).to(device)
        
        if len(ss) != len(os):
            raise TypeError('Quaternions cannot broadcast together shapes %s and %s' % (sqs.shape, oqs.shape))
            
        if torch.all(ss == os): return sqs, oqs  # TODO: check torch.all
        # ipdb.set_trace()
        if not torch.all((ss.to(device) == os.to(device)) | (os == torch.ones(len(os)).to(device)) | (ss == torch.ones(len(ss)).to(device))):
            raise TypeError('Quaternions cannot broadcast together shapes %s and %s' % (sqs.shape, oqs.shape))
            
        sqsn, oqsn = sqs.clone(), oqs.clone()
        
        for a in torch.where(ss == 1)[0]: sqsn = sqsn.repeat_interleave(os[a], dim=a)
        for a in torch.where(os == 1)[0]: oqsn = oqsn.repeat_interleave(ss[a], dim=a)
        
        return sqsn, oqsn
        
    """ Adding Quaterions is just Defined as Multiplication """
    
    def __add__(self, other): return self * other
    def __sub__(self, other): return self / other
    
    """ Quaterion Multiplication """
    
    def __mul__(self, other):
        """
        Quaternion multiplication has three main methods.
        
        When multiplying a Quaternions array by Quaternions
        normal quaternion multiplication is performed.
        
        When multiplying a Quaternions array by a vector
        array of the same shape, where the last axis is 3,
        it is assumed to be a Quaternion by 3D-Vector 
        multiplication and the 3D-Vectors are rotated
        in space by the Quaternions.
        
        When multipplying a Quaternions array by a scalar
        or vector of different shape it is assumed to be
        a Quaternions by Scalars multiplication and the
        Quaternions are scaled using Slerp and the identity
        quaternions.
        """
        
        """ If Quaternions type do Quaternions * Quaternions """
        if isinstance(other, Quaternions_torch):
            
            sqs, oqs = Quaternions_torch._broadcast(self.qs, other.qs)
            
            q0 = sqs[...,0]; q1 = sqs[...,1]; 
            q2 = sqs[...,2]; q3 = sqs[...,3]; 
            r0 = oqs[...,0]; r1 = oqs[...,1]; 
            r2 = oqs[...,2]; r3 = oqs[...,3]; 
            
            qs = torch.empty(sqs.shape).to(device)
            qs[...,0] = r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3
            qs[...,1] = r0 * q1 + r1 * q0 - r2 * q3 + r3 * q2
            qs[...,2] = r0 * q2 + r1 * q3 + r2 * q0 - r3 * q1
            qs[...,3] = r0 * q3 - r1 * q2 + r2 * q1 + r3 * q0
            
            return Quaternions_torch(qs)
        
        """ If array type do Quaternions * Vectors """
        if isinstance(other, torch.Tensor) and other.shape[-1] == 3:
            vs = Quaternions_torch(torch.cat([torch.zeros(other.shape[:-1] + (1,)).to(device), other], dim=-1))
            # ipdb.set_trace()
            return (self * (vs * -self)).imaginaries
        
        """ If float do Quaternions * Scalars """
        if isinstance(other,torch.Tensor) or isinstance(other, float):
            return Quaternions_torch.slerp(Quaternions_torch.id_like(self), self, other)
        
        raise TypeError('Cannot multiply/add Quaternions with type %s' % str(type(other)))
        
    def __div__(self, other):
        """
        When a Quaternion type is supplied, division is defined
        as multiplication by the inverse of that Quaternion.
        
        When a scalar or vector is supplied it is defined
        as multiplicaion of one over the supplied value.
        Essentially a scaling.
        """
        
        if isinstance(other, Quaternions_torch): return self * (-other)
        if isinstance(other, torch.Tensor): return self * (1.0 / other)
        if isinstance(other, float): return self * (1.0 / other)
        raise TypeError('Cannot divide/subtract Quaternions with type %s' + str(type(other)))
        
    def __eq__(self, other): return self.qs == other.qs
    def __ne__(self, other): return self.qs != other.qs
    
    def __neg__(self):
        """ Invert Quaternions """
        return Quaternions_torch(self.qs * torch.tensor([[1, -1, -1, -1]]).to(device))
    
    def __abs__(self):
        """ Unify Quaternions To Single Pole """
        qabs = self.normalized().copy()
        top = torch.sum(( qabs.qs) * torch.tensor([1,0,0,0]).to(device), dim=-1)
        bot = torch.sum((-qabs.qs) * torch.tensor([1,0,0,0]).to(device), dim=-1)
        qabs.qs[top < bot] = -qabs.qs[top <  bot]
        return qabs
    
    def __iter__(self): return iter(self.qs)
    def __len__(self): return len(self.qs)
    
    def __getitem__(self, k):    return Quaternions_torch(self.qs[k]) 
    def __setitem__(self, k, v): self.qs[k] = v.qs
        
    @property
    def lengths(self):
        return torch.sum(self.qs**2.0, axis=-1)**0.5
    
    @property
    def reals(self):
        return self.qs[...,0]
        
    @property
    def imaginaries(self):
        return self.qs[...,1:4]
    
    @property
    def shape(self): return self.qs.shape[:-1]
    
    def repeat(self, n, **kwargs):
        return Quaternions_torch(self.qs.repeat(n, **kwargs))
    
    def normalized(self):
        return Quaternions_torch(self.qs / self.lengths.unsqueeze(-1))

    def unsqueeze(self, dim):
        return Quaternions_torch(self.qs.unsqueeze(dim))
    
    def log(self):
        norm = torch.abs(self.normalized())
        imgs = norm.imaginaries
        lens = torch.sqrt(torch.sum(imgs**2, dim=-1))
        lens = torch.arctan2(lens, norm.reals) / (lens + 1e-10)
        return imgs * lens.unsqueeze(-1)
    
    def constrained(self, axis):
        
        rl = self.reals
        im = torch.sum(axis * self.imaginaries, dim=-1)
        
        t1 = -2 * torch.arctan2(rl, im) + torch.pi
        t2 = -2 * torch.arctan2(rl, im) - torch.pi
        
        top = Quaternions_torch.exp(axis.unsqueeze(-1) * (t1.unsqueeze(-1) / 2.0))
        bot = Quaternions_torch.exp(axis.unsqueeze(-1) * (t2.unsqueeze(-1) / 2.0))
        img = self.dot(top) > self.dot(bot)
        
        ret = top.detach().clone()  #.copy()
        ret[ img] = top[ img]
        ret[~img] = bot[~img]
        return ret
    
    def constrained_x(self): return self.constrained(torch.tensor([1,0,0]).to(device))
    def constrained_y(self): return self.constrained(torch.tensor([0,1,0]).to(device))
    def constrained_z(self): return self.constrained(torch.tensor([0,0,1]).to(device))
    
    def dot(self, q): return torch.sum(self.qs * q.qs, axis=-1)
    
    def copy(self): return Quaternions_torch(self.qs.detach().clone())
    
    def reshape(self, s):
        self.qs.reshape(s)
        return self

    @classmethod
    def between(cls, v0s, v1s):
        a = torch.cross(v0s, v1s)
        w = torch.sqrt((v0s**2).sum(dim=-1) * (v1s**2).sum(dim=-1)) + (v0s * v1s).sum(dim=-1)
        return Quaternions_torch(torch.cat([w.unsqueeze(-1), a], dim=-1)).normalized()
    
