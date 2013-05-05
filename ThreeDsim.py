from __future__ import division
from visual import *
import numpy as np
from numpy import linalg as la
from scipy.integrate import ode
from time import clock, sleep
from scipy.optimize import minimize
from scipy.optimize import fmin_slsqp

import math

"""Gravity acts along y which is parameters 1, width of street is along x (first axis),
and going down the street is along the z (third) axis"""
def normalSpring(t, v, parameters):
    r = v[0:3]
    velocity = v[3:]
    drdt = velocity
    d = la.norm(r) - parameters.equilibriumLength
    dvdt = parameters.gravity - d*parameters.k*(r/la.norm(r))
    out = np.concatenate((drdt, dvdt))
    return out

# The function that implements the spring pendulum odes
def tensionSpring(t, v, parameters):
    r = v[0:3]
    velocity = v[3:]
    drdt = velocity
    d = la.norm(r) - parameters.equilibriumLength
    #If the sim is a stretchy string
    if d < 0:
        d = 0
    dvdt = parameters.gravity - d*parameters.k*(r/la.norm(r))
    out = np.concatenate((drdt, dvdt))
    return out

def switchingSpring(t, v, parameters, slingingRule):
    """Assumes that slingingRule is a lambda function that takes in
    the parameters object and evaluates to zero when spiderman should
    send out a new web."""
    switcher = (lambda t, v: switchingIteration(t, v, parameters, slingingRule))
    sim = ode(switcher).set_integrator('dopri5')
    sim.set_initial_value(v,t)
    while sim.successful():
        rate(100)
        try:
            sim.integrate(sim.t+parameters.dt)
            visualize(sim.y, parameters)
        except Exception as finish:
            return finish.args


def switchingIteration(t, v, parameters, slingingRule):
    if (not slingingRule(v, parameters)):
        raise Exception(t, v)
    r = v[0:3]
    velocity = v[3:]
    drdt = velocity
    d = la.norm(r) - parameters.equilibriumLength
    #If the sim is a stretchy string
    if d < 0:
        d = 0
    dvdt = parameters.gravity - d*parameters.k*(r/la.norm(r))/parameters.mass - dragVector(velocity, parameters)/parameters.mass
    #print dragVector(velocity, parameters)/parameters.mass
    out = np.concatenate((drdt, dvdt))
    return out

def dragVector(velocity, parameters):
    dragParameter = lambda x: 0.5*parameters.airDensity*(x**2)*parameters.dragCoefficient*parameters.area
    drag = np.array([dragParameter(direction) for direction in velocity])
    return drag

def minimizeWebEnergyLost(x, parameters):
    """values = [theta, velocity]"""
    firstTerm = lambda values: (x * values[1]**2 / 2.0)
    sqrtTerm = lambda values: np.sqrt((parameters.gravity[1]**2 * x**2) / (4 * values[1]**4 * np.cos(values[0])**4) + 1)
    secondTerm = lambda values: (values[1]**4 * np.cos(values[0])**2) / parameters.gravity[1]
    arcsinhTerm = lambda values: np.arcsinh((parameters.gravity[1] * x) / (2 * values[1]**2 * np.cos(values[0])**2))
    costFunction = lambda values: firstTerm(values)*sqrtTerm(values)+secondTerm(values)*arcsinhTerm(values)
    bounds = ((0,(math.pi/2.0)),(0,parameters.maxSlingSpeed))

    leftTerm = lambda values: ((-parameters.gravity[1]/2)*parameters.distanceToWall**2)/((values[1]**2)*(math.cos(values[0])))
    fullConstraint = lambda values: leftTerm(values)+(math.tan(0)*parameters.distanceToWall)-parameters.heightChange
    cons = ({'type': 'eq', 'fun': lambda values: fullConstraint(values)})
    #ineq means not equals to zero. eq is equal to zero
    #return costFunction
    minimum = minimize(costFunction, (pi/4, 100), method="SLSQP", bounds=bounds, constraints=cons)
    return minimum


#A function that will update the vpython sim
def visualize(vals, parameters):
    """Updates the vpython sim with current values"""
    r = vals[0:3]
    velocity = vals[3:]
    parameters.rod.axis=r
    parameters.ball.pos=parameters.rod.pos + r
    # Printing the current total energy
    test = (parameters.mass/2 * la.norm(velocity)**2,
            0 - parameters.mass * np.dot(parameters.gravity,r), 0)
    parameters.energyTracker.append([parameters.mass/2 * la.norm(velocity)**2,
             0 - parameters.mass * np.dot(parameters.gravity,parameters.rod.pos + r), 0])
    if la.norm(r)-parameters.equilibriumLength > 0:
        parameters.energyTracker[-1][2] = parameters.k/2*(la.norm(r)-parameters.equilibriumLength)**2
    parameters.energyTracker[-1].append(sum(parameters.energyTracker[-1]))


def initVisualization(r0, parameters):
    """Initializes the vpython visualization"""
    scene.forward = (0,1,0)
    scene.range = (100,100,100)
    scene.center = (10,0,30)
    scene.autoscale = False
    parameters.rod = cylinder(pos=(0,0,0), axis=(1,0,0), radius=0.05)
    parameters.ball = sphere(pos=(1,0,0), radius = 0.1)
