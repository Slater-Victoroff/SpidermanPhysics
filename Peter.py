import matplotlib.pyplot as plot
import numpy as np
from numpy import linalg as la
from scipy.integrate import ode
import pickle
import cProfile

class Params(object):
    def __init__(self, k=1000, extension = 0.84):
        self.mass = 50 #kg
        self.k = k #N/m
        self.area = 0.5 #m^2
        webDiameter = 0.002 #m^2
        webArea = np.pi*webDiameter**2
        stiffness = 10000000000.0 #Pa
        self.youngsConstant = stiffness*webArea
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
        self.dt = 0.005 
        self.energyBuffer = 10

class Tracker:
    def __init__(self):
        self.gPotential = []
        self.sPotential = []
        self.kinetic = []
        self.nonSpring = []
        self.total = []

def getGPotential(v, r, params):
    return -params.mass * np.dot(params.gravity,r)

def getSPotential(v, r, params):
    d = la.norm(r)-params.equilibriumLength
    return (params.k/2*(d)**2 if d > 0 else 0)

def getKinetic(v, r, params):
    return 0.5*params.mass*(la.norm(v)**2)

def updateEnergy(v, r, params, tracker):
    tracker.sPotential.append(getSPotential(v, r, params))
    tracker.gPotential.append(getGPotential(v, r, params))
    tracker.kinetic.append(getKinetic(v, r, params))
    tracker.nonSpring.append(tracker.gPotential[-1] + tracker.kinetic[-1])
    tracker.total.append(tracker.nonSpring[-1] + tracker.sPotential[-1])

def clearTracker(tracker):
    tracker.gPotential = []
    tracker.sPotential = []
    tracker.kinetic = []
    tracker.nonSpring = []
    tracker.total = []

def swing(v0, r0, params, Tracker, when):
    tracker = Tracker()
    #get a good initial energy
    params.equilibriumLength = la.norm(r0) * params.extension
    params.k = params.youngsConstant / params.equilibriumLength
    updateEnergy(v0, r0, params, tracker)
    swinger = lambda t,v : iterate(t, v, params, tracker)
    sim = ode(swinger).set_integrator('dopri5')
    sim.set_initial_value(np.concatenate((r0,v0)),0)
    while sim.successful() and not when(sim.y[:3], sim.y[3:], params, tracker):
        sim.integrate(sim.t + params.dt)
        updateEnergy(sim.y[3:], sim.y[:3], params, tracker)
    rf, vf = sim.y[3:], sim.y[:3]
    print vf
    return rf, vf, sim.t, tracker

def iterate(t, v, params, tracker):
    r, v = v[:3], v[3:]
    d = la.norm(r) - params.equilibriumLength
    if d < 0:
        d = 0
    dvdt = params.gravity - d*params.k*(r/la.norm(r))/params.mass +\
            dragVector(v, params)/params.mass
    return np.concatenate((v, dvdt))

def dragVector(v, params):
    return -0.5*params.airDensity*la.norm(v)*v*params.dragCoefficient*params.area

def equalEnergy(v, r, params, tracker):
    return tracker.nonSpring[-1] < tracker.nonSpring[0] - params.energyBuffer

def timePlotter(data, lineLabels, ylabel, xlabel = "Time", dt = 0.01, filename="fig.png", display=False, plotTitle=""):
    """Wants a 1d array of data in data,
    multiple rows will be interpreted as multiple lines to plot
    will set axes and whatnot"""
    endTime = len(data[0])*dt
    time = np.linspace(0.0,endTime,len(data[0])) 
    for i in range(0, len(data)):
        plot.plot(time, data[i], label=lineLabels[i])
    plot.legend()
    plot.title(plotTitle)
    plot.savefig(filename)
    if display:
        plot.show()

def phasePlotCollect(bounds, simfunc, granularities=[20,20], collector=lambda x: x):
    """iterates simfunc over params found in bounds, and maps that data to a
    final value via collectionFunc. By default collectionFunc is the identity.
    returns a list of lists"""
    sets = []
    for bound, granularity in zip(bounds, granularities):
        sets.append([bound[1] * n + bound[0] * (1 - n) for n in 
            [float(x) / granularity for x in range(0,granularity)]])
    return [[ collector(simfunc(p1,p2)) for p2 in sets[1] ] for p1 in sets[0]]

def phasePlot(data, xAxis=[0,40], yAxis=[0,40], xLabel="", yLabel="", filename="phase", display=True, plotTitle=""):
    """data holds everything and then the bins are how many bins we want to
    have in each direction"""
    fig = plot.figure(figsize=(6, 3.2))
    xAxis.extend(yAxis)

    filename += ".png"
    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    plot.imshow(data, extent = xAxis)
    #ax.set_aspect('equal')
    ax.axis(xAxis)

    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plot.colorbar(orientation='vertical')
    plot.title(plotTitle)
    plot.savefig(filename)
    if display:
        plot.show()

if __name__ == '__main__':
    #Will generate a phase plot if set to true
    collecting = True
    savename = 'reasonableInitialVelocity'
    params = Params()
    vf, rf, t, tracker = swing([0,0,20],[-10,-8,-25], params, Tracker, equalEnergy)
    timePlotter([tracker.gPotential, tracker.sPotential, tracker.kinetic], \
                ["Gravitational Potential","Spring Potential","Kinetic Energy"], \
                "Energy of the System over Time", display=True, dt=params.dt)
    if collecting:
        #resetting the params and tracker
        #example simfunc for use with phasePlotCollect
        simFunc = lambda p1, p2: swing([-3,-2,10],[-10,p1,p2], Params(), Tracker, equalEnergy)
        #example collection function for use with phasePlotCollect
        collector = lambda data: data[3].gPotential[-1] - data[3].gPotential[0]
        #Example use of phasePlotCollect
        data = phasePlotCollect([[-1,-40],[-1,-40]],simFunc, granularities=[30,30])
        with open(savename + '.data', 'w') as dump:
            pickle.dump(data, dump)

    with open(savename + '.data') as source:
        data = pickle.load(source)
    data = [[datum[0][2] for datum in datarow] for datarow in data]
    data = [[datum[3].gPotential[-1] - datum[3].gPotential[0] for datum in datarow] for datarow in data]
    phasePlot(data, [1,40], [40,1], xLabel="x position", yLabel="y position", filename=savename)