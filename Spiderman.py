import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import ThreeDsim
import cProfile, pstats, io
from PIL import Image
import json

"""Worth noting that x is distance along the width of the street,
y is vertical, and z is distance along the length of the street."""
class Params(object):
    def __init__(self, k=1000, extension = 0.84):
        self.mass = 50 #kg
        self.k = k #N/m
        self.area = 0.5 #m^2
        self.extension = extension #Percentage of new length
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
        self.kinetic = []
        self.potential = []

def energyCalculator(velocity, r, parameters):
    kinetic = 0.5*parameters.mass*(la.norm(velocity)**2)
    potential = -parameters.mass * np.dot(parameters.gravity,r)
    d = la.norm(r)-parameters.equilibriumLength
    spring = (parameters.k/2*(d)**2 if d > 0 else 0)
    return (kinetic, potential, spring, (kinetic+potential+spring))

def goGetEmTiger():
    """The top-level optimization function. Currently is foiled by
    The  Green Goblin"""
    print 'Damn you, Norman Osborn!'

def simplewhere(v, r, params):
    """simple implementation of a where function. It always chooses
    to shoot 10m forward and 10m up, and sends out 1.1 times the
    radius of web."""
    #rnew is spiderman's position in terms of the web
    rnew = np.array([0,-10,-10], dtype=np.float)
    rnew[0] = (r[0] - params.streetWidth if not params.left else r[0])
    return rnew, la.norm(rnew) * params.extension

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

def eqwhen(vec, params):
    r, v = vec[:3], vec[3:]
    return energyCalculator(v,r, params)[3] > params.initialEnergy

def websling(r0=np.array([12,0,0]), v0=np.array([0,0,10]), where=simplewhere, when=simplewhen, iterations=50, k=1000, extension = 0.84, vis=False, plot=False):
    """Simulates Spiderman's webslinging for the given parameters"""
    params = Params(k=k, extension=extension)
    # the where function takes a position and velocity in global space
    # and returns radius to spiderman from the new pendulum center,
    # along with the equilibrium length of that web.
    v0, rGlobal = [v0], [r0]
    r0 = []
    t = 0
    if vis: ThreeDsim.initVisualization(rGlobal, params)
    else: ThreeDsim.initFalseVisualization(rGlobal, params)
    (r0new, l) = where(v0[-1], rGlobal[-1], params)
    r0.append(r0new)
    params.equilibriumLength = la.norm(r0[-1])
    params.initialEnergy = energyCalculator(v0[-1],r0[-1],params)[3]
    for i in xrange(0, iterations):
        params.equilibriumLength = l
        t, outvec = ThreeDsim.switchingSpring(
            t, np.concatenate((r0[i],v0[i])),params, when, vis)
        rf, v0new = outvec[:3], outvec[3:]
        v0.append(v0new)
        rGlobal.append(rGlobal[i] - r0[i] + rf)
        params.initialEnergy = energyCalculator(v0[-1],r0[-1],params)[3]
        params.left = not params.left
        (r0new, l) = where(v0[i], rGlobal[i], params)
        r0.append(r0new)
        params.rod.pos = rGlobal[i] - r0[i]
    if plot:
        plt.plot(params.energyTracker)
        plt.savefig('fig.png')
    return rGlobal, v0, params.energyTracker, t

def singleIter(r0=np.array([-12,-10,-10]), v0=np.array([0,0,10]), params=Params(), when=eqwhen, vis=False):
    """Simulates Spiderman's webslinging for the given parameters"""
    # the where function takes a position and velocity in global space
    # and returns radius to spiderman from the new pendulum center,
    # along with the equilibrium length of that web.
    t = 0
    #trick it into thinking there's no initial stretchiness
    params.equilibriumLength = la.norm(r0)
    params.initialEnergy = energyCalculator(v0,r0,params)[3]
    params.equilibriumLength = la.norm(r0) * params.extension
    t, outvec = ThreeDsim.switchingSpring(
         t, np.concatenate((r0,v0)),params, when, vis)
    rf, vf = outvec[:3], outvec[3:]
    return rf, vf, params.energyTracker, t

def plotify(r0=np.array([-12,-10,-10]), v0=np.array([0,0,10]), plot=True, k=1000, extension=0.84, filename='fig.png'):
    """Simulates Spiderman's webslinging for the given parameters"""
    params = Params(k=k, extension=extension)
    print params.extension
    v0, rf, energy, t = singleIter(r0,v0, params, eqwhen, vis=False)
    if plot:
        plt.plot(energy)
        plt.savefig(filename)
    return rf, v0, energy, t


def searchFor(condition, ranges, granularities, r0=[-12,-10,-10], v0=np.array([0,0,10]), when=eqwhen):
    params = Params()
    xSpan = ranges[0][1]-ranges[0][0]
    ySpan = ranges[1][1] - ranges[1][0]
    xTrend = int(xSpan/granularities[0])
    yTrend = int(ySpan/granularities[1])
    goodlist = []
    for y in range(0,yTrend):
        params.k = ranges[1][0] + y*granularities[1]
        for x in range(0,xTrend):
            params.extension = ranges[0][0]+x*granularities[0]
            stats = singleIter(r0, v0, params)
            print stats
            if condition(stats, r0, v0):
                goodlist.append((params.k, params.extension))
                print 'Good data!'
    return goodlist


def lowDU(stats, r0, v0):
    return abs(r0[1] - stats[0][1]) < 0.001

def lossFunction(params, k, extension):
    params.k = k
    params.extension = extension
    rGlobal, v0, energyTracker, t = singleIter(params=params)
    potentialLoss = (energyTracker[0][1] - energyTracker[-1][1])/len(energyTracker)
    return potentialLoss, t

    
def normalize(array):
    maximum = np.max(array)
    minimum = np.min(array)
    array -= minimum
    array *= 255*(minimum/maximum)

def manualPhasePlot(function, ranges, granularities):
    params = Params()
    xSpan = ranges[0][1]-ranges[0][0]
    ySpan = ranges[1][1] - ranges[1][0]
    xTrend = int(xSpan/granularities[0])
    yTrend = int(ySpan/granularities[1])
    potentialPhasePlot=[]
    timePhasePlot = []
    for y in range(0,yTrend):
        potentialRow = []
        timeRow = []
        usefulY = ranges[1][0] + y*granularities[1]
        for x in range(0,xTrend):
            usefulX = ranges[0][0]+x*granularities[0]
            stats = function(params, usefulX, usefulY)
            potentialRow.append(stats[0])
            timeRow.append(stats[1])
        potentialPhasePlot.append(potentialRow)
        timePhasePlot.append(timeRow)
    potentialPhasePlot = np.array(potentialPhasePlot)
    timePhasePlot = np.array(timePhasePlot)
    normalize(potentialPhasePlot)
    normalize(timePhasePlot)
    #Normalizing
    potImage = Image.fromarray(potentialPhasePlot,'L')
    potImage.save("potentialPhasePlot.png")
    timeImage = Image.fromarray(timePhasePlot, 'L')
    timeImage.save("timePhasePlot.png")

def cramAwayAllData(function, ranges, granularities):
    xSpan = ranges[0][1]-ranges[0][0]
    ySpan = ranges[1][1] - ranges[1][0]
    xTrend = int(xSpan/granularities[0])
    yTrend = int(ySpan/granularities[1])
    fullPhasePlot = []
    for y in range(0,yTrend):
        currentRow = []
        usefulY = ranges[1][0] + y*granularities[1]
        for x in range(0,xTrend):
            usefulX = ranges[0][0]+x*granularities[0]
            currentRow.append(function(k=usefulX,extension=usefulY))
        fullPhasePlot.append(currentRow)
    with open("JSONDUMP.json", 'wb') as jsonData:
        pickle.dump(fullPhasePlot,jsonData)

if __name__ == '__main__':
    statify=False
    if statify:
        pr = cProfile.Profile()
        pr.enable()
    #rGlobal, v0, E, t = websling(np.array([12,0,0]),np.array([0,0,10]), simplewhere, eqwhen, 10, vis=True, plot=True)
    #rGlobal, v0, E, t = plotify(plot=True)
    #print searchFor(lowDU, np.array([[500,2000],[0.8,1.0]]), np.array([37.5,.005]))
    params=Params()
    potentialChanges = []
    granularity = 20
    for change in range (0,100):
        print change
        potentialChanges.append(lossFunction(params,1500, 0.7+(change*0.003)))
    plt.plot(potentialChanges)
    plt.show()
    #plotify(k=362.5, extension=0.90, plot=True, filename='fig4.png')
    #manualPhasePlot(lossFunction, np.array([[500,2000],[0.8,1.0]]), np.array([37.5,.005]))
    #cramAwayAllData(websling, np.array([[1000,5000],[0.6,0.9]]), np.array([37,5.0075]))
    if statify:
        pr.disable()
        pstats.Stats(pr).print_stats()
