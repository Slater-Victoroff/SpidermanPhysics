from __future__ import division
import numpy as np
from numpy import linalg as la
from euler import euler

params = {}
params['eqlen'] = 1 #m
params['m'] = 1 #kg
params['k'] = 10 #N/m
params['g'] = np.matrix((0,0,-9.81)) #m/s^2

def springPendulum(inlist, params):
    r = np.matrix(inlist[0:3])
    v = np.matrix(inlist[3:])

    drdt = v
    dvdt = params['g'] + params['k']*(params['eqlen']/la.norm(r) - 1)*r
    out = drdt.tolist()[0] + dvdt.tolist()[0]
    #Because it's a matrix, returns a list of lists.
    return out

simspring = (lambda l: springPendulum(l,params))

output = euler([1,0,0,0,0,0],simspring, exittime=1)
