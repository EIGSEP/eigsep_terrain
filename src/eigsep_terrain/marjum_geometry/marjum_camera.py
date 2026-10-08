"""Residual radial distortion in processed, bottom-up Marjum image coordinates.

Coefficients act in focal-normalized coordinates, so isotropic crops/resizes
and 90-degree image rotations do not require different radial coefficients.
No claim is made that these are Apple's unprocessed optical calibration.
"""
import numpy as np
from marjum_bundle import rotation


def distort(x, k):
    x = np.asarray(x, float)
    r2 = np.sum(x*x, axis=-1, keepdims=True)
    return x*(1+k[0]*r2+k[1]*r2*r2)


def normalized(p, shape, xy, k):
    h,w = shape
    d = (np.atleast_2d(xy)-[w//2,h//2])/p[6]
    rd = np.linalg.norm(d,axis=1)
    r = rd.copy()
    for _ in range(25):
        r2 = r*r
        value = r*(1+k[0]*r2+k[1]*r2*r2)-rd
        derivative = 1+3*k[0]*r2+5*k[1]*r2*r2
        step = value/np.maximum(derivative,.05)
        r = np.maximum(0,r-np.clip(step,-.25,.25))
    return d*(r/np.maximum(rd,1e-30))[:,None]


def rays(p,shape,xy,k=(0.,0.)):
    q = normalized(p,shape,xy,k)
    body = np.c_[-q[:,1],-q[:,0],np.ones(len(q))]
    world = body@rotation(p).T
    return world/np.linalg.norm(world,axis=1,keepdims=True)


def project(p,shape,xyz,k=(0.,0.)):
    h,w = shape
    v = (np.atleast_2d(xyz)-p[:3])@rotation(p)
    q = -v[:,[1,0]]/np.maximum(v[:,2:3],.1)
    return distort(q,k)*p[6]+[w//2,h//2],v[:,2]


def radial_support(p,shape,k):
    """Sample monotonicity across the entire observed field, including corners."""
    h,w = shape
    rd = np.hypot(w/2,h/2)/p[6]
    corners = normalized(p,shape,[[0,0],[w-1,h-1]],k)
    radius = max(rd,float(np.linalg.norm(corners,axis=1).max()))
    r = np.linspace(0,radius,24)
    return 1+3*k[0]*r*r+5*k[1]*r**4


def epipolar_error(p1,s1,k1,p2,s2,k2,x,y):
    """Generalized Sampson error, native pixels, including distortion Jacobians."""
    baseline = p2[:3]-p1[:3]
    if np.linalg.norm(baseline)<.01:
        return np.full(len(x),np.nan)
    baseline /= np.linalg.norm(baseline)
    def constraint(a,b):
        return np.einsum('ij,j->i',np.cross(rays(p1,s1,a,k1),rays(p2,s2,b,k2)),baseline)
    value = constraint(x,y)
    gradients=[]
    for image in [0,1]:
        for axis in [0,1]:
            d=np.zeros_like(x,dtype=float);d[:,axis]=.05
            gradients.append((constraint(x+d,y)-constraint(x-d,y))/.1 if image==0 else
                             (constraint(x,y+d)-constraint(x,y-d))/.1)
    return abs(value)/np.maximum(np.sqrt(np.sum(np.array(gradients)**2,axis=0)),1e-15)
