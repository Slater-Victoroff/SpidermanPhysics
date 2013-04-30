import numpy as np
from numpy import linalg as la
import 3Dsim

"""Worth noting that in the coordinate system here gravity is assumed to be in the
negative q2 direction. Other two axes are arbitrary as far as I can tell"""
class Params(object):
    def __init__(self):
        self.mass = 50 #kg
        self.k = 100 #N/m
        self.area =
        self.gravity = np.array((0,-9.81,0)) #m/s^2
        self.webDensity = 0.0001 #kg/m Assuming a constant thickness for web
        self.maxSlingSpeed = 1000 #m/s
        self.streetWidth = 20;
        self.left = True

def goGetEmTiger():
    """The top-level optimization function. Currently is foiled by
    The  Green Goblin"""
    print 'Damn you, Norman Osborn!'

def websling(v0, r0, where, when, iterations):
    """Simulates Spiderman's webslinging for the given parameters"""
    params = Params()
    # the where function takes a position and velocity in global space
    # and returns radius to spiderman from the new pendulum center,
    # along with the equilibrium length of that web.
    v0, rGlobal = [v0], [r0]
    r0 = []
    for i in xrange(0, iterations):
        (r0[i], l) = where(v0[i], rGlobal[i], params)
        params.equilibriumLength = l
        outvec = 3Dsim.switchingSpring(np.array(r[0]+v0[i]),
                                           params, when)
        rf, v0[i+1] = outvec[:3], outvec[3:]
        rGlobal[i+1] = rGlobal[i] - r0[i] + rf
    # Call the swing function here

def simplewhere(v, r, params)
    """simple implementation of a where function. It always chooses
    to shoot 10m forward and 10m up, and sends out 1.1 times the
    radius of web."""
    r = r + np.array([0,10,10])
    r[0] = (params.streetWidth if params.left else 0)
    params.left = not params.left
    return r, r * 1.1

def simplewhen(vec, params):
    r, v = vec[:3], vec[3:]
    if params.left:
        ang = np.arctan(r[2]/r[0])
    else:
        ang = np.arctan(r[2]/(params.streetWidth - r[0]))
    if ang > (np.pi / 6):
        return False
    else:
        return True

if __name__ == '__main__':
    print 'sup'
