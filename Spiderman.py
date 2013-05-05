import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import ThreeDsim

"""Worth noting that in the coordinate system here gravity is assumed to be in the
negative q2 direction. Other two axes are arbitrary as far as I can tell"""
class Params(object):
    def __init__(self):
        self.mass = 50 #kg
        self.k = 1000 #N/m
        self.area = 0.5 #m^2
        self.gravity = np.array((0,-9.81,0)) #m/s^2
        self.airDensity = 1.2 #kg/m^3
        self.webDensity = 0.0001 #kg/m Assuming a constant thickness for web
        self.terminalVelocity = 70 #m/s
        #took norm of self.gravity to make coefficient scalar
        self.dragCoefficient = 1.0/((self.airDensity*self.area)*(self.terminalVelocity**2/(2*self.mass*la.norm(self.gravity))))
        self.maxSlingSpeed = 1000 #m/s
        self.streetWidth = 20;
        self.left = True
        self.dt = 0.01
        self.energyTracker = []

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
    ThreeDsim.initVisualization(rGlobal, params)
    for i in xrange(0, iterations):
        (r0new, l) = where(v0[i], rGlobal[i], params)
        r0.append(r0new)
        print rGlobal[i] - r0[i]
        params.rod.pos = rGlobal[i] - r0[i]
        params.equilibriumLength = l
        t, outvec = ThreeDsim.switchingSpring(
            t, np.concatenate((r0[i],v0[i])),params, when)
        rf, v0new = outvec[:3], outvec[3:]
        v0.append(v0new)
        rGlobal.append(rGlobal[i] - r0[i] + rf)
        params.left = not params.left
    plt.plot(params.energyTracker)
    plt.savefig('fig.png')
    print rGlobal, v0

def simplewhere(v, r, params):
    """simple implementation of a where function. It always chooses
    to shoot 10m forward and 10m up, and sends out 1.1 times the
    radius of web."""
    #rnew is spiderman's position in terms of the web
    rnew = np.array([0,-10,-10], dtype=np.float)
    rnew[0] = (r[0] - params.streetWidth if not params.left else r[0])
    return rnew, la.norm(rnew) * 0.84

def simplewhen(vec, params):
    r, v = vec[:3], vec[3:]
    if params.left:
        ang = np.arctan(r[2]/r[0])
    else:
        ang = np.arctan(r[2]/(-r[0]))
    if ang > (np.pi / 20):
        return False
    else:
        return True

if __name__ == '__main__':
    websling(np.array([12,0,0]),np.array([0,0,10]), simplewhere, simplewhen, 10)
