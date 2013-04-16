from __future__ import division
from visual import *
import numpy as np
from numpy import linalg as la
from euler import euler
from time import clock

params = {}
params['eqlen'] = 1 #m
params['m'] = 1 #kg
params['k'] = 100 #N/m
params['g'] = np.matrix((0,-9.81,0)) #m/s^2

def springPendulum(inlist, params):
    r = np.matrix(inlist[0:3])
    v = np.matrix(inlist[3:])
    drdt = v
    dvdt = params['g'] + params['k']*(params['eqlen']/la.norm(r) - 1)*r
    out = drdt.tolist()[0] + dvdt.tolist()[0]
    #Because it's a matrix, returns a list of lists.
    return out

def visualize(vals, params):
    r = vals[0:3]
    v = vals[3:]
    rod.axis=r
    ball.pos=r
    # For energy checking
    print params['m']/2 * la.norm(v)**2 + params['k']/2*(la.norm(r)-params['eqlen'])**2 - params['m'] * np.dot(params['g'],r)

simspring = (lambda l: springPendulum(l,params))
vis = (lambda l: visualize(l,params))

dt = 0.001
rod = cylinder(pos=(0,0,0), axis=(1,0,0), radius=0.05)
ball = sphere(pos=(1,0,0), radius = 0.1)
output = euler([1,0,0,0,0,0],simspring, visualize=vis, exittime=10, dt=dt, realtime=True)
