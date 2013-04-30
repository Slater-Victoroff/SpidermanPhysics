import numpy as np
from numpy import linalg as la
import ThreeDsim

"""Worth noting that in the coordinate system here gravity is assumed to be in the
negative q2 direction. Other two axes are arbitrary as far as I can tell"""
class Params(object):
    def __init__(self):
        self.mass = 50 #kg
        self.k = 100 #N/m
        self.area = 0.5 #m^2
        self.gravity = np.array((0,-9.81,0)) #m/s^2
        self.airDensity = 1.2 #kg/m^3
        self.webDensity = 0.0001 #kg/m Assuming a constant thickness for web
        self.terminalVelocity = 70 #m/s
        self.dragCoefficient = 1.0/((self.airDensity*self.area)*(self.terminalVelocity**2/(2*self.mass*self.gravity)))
        self.maxSlingSpeed = 1000 #m/s
        self.streetWidth = 20;
        self.left = True
        self.dt = 0.01

def goGetEmTiger():
    """The top-level optimization function. Currently is foiled by
    The  Green Goblin"""
    print 'Damn you, Norman Osborn!'

def websling(r0, v0, where, when, iterations):
    """Simulates Spiderman's webslinging for the given parameters"""
    params = Params()
    # the where function takes a position and velocity in global space
    # and returns radius to spiderman from the new pendulum center,
    # along with the equilibrium length of that web.
    v0, rGlobal = [v0], [r0]
    r0 = []
    t = 0
    for i in xrange(0, iterations):
        (r0new, l) = where(v0[i], rGlobal[i], params)
        r0.append(r0new)
        params.equilibriumLength = l
        t, outvec = ThreeDsim.switchingSpring(t, np.array(r0[0]+v0[i]),
                                              params, when)
        rf, v0new = outvec[:3], outvec[3:]
        v0.append(v0new)
        rGlobal.append(rGlobal[i] - r0[i] + rf)
    # Call the swing function here

def simplewhere(v, r, params):
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
    websling(np.array([8,0,0]),np.array([2,0,5]), simplewhere, simplewhen, 2)
