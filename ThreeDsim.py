from __future__ import division
from visual import *
import numpy as np
from numpy import linalg as la
from scipy.integrate import ode
from time import clock, sleep
from scipy.optimize import minimize
from scipy.optimize import fmin_slsqp
import Spiderman as Spidey
from matplotlib import pyplot

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

def switchingSpring(t, v, parameters, slingingRule, vis=False):
    """Assumes that slingingRule is a lambda function that takes in
    the parameters object and evaluates to zero when spiderman should
    send out a new web."""
    switcher = (lambda t, v: switchingIteration(t, v, parameters, slingingRule))
    sim = ode(switcher).set_integrator('dopri5')
    sim.set_initial_value(v,t)
    while sim.successful():
        if vis: rate(100)
        if (slingingRule(sim.y, parameters)):
            sim.integrate(sim.t+parameters.dt)
            visualize(sim.y, parameters)
        else:
            return sim.t, sim.y


def switchingIteration(t, v, parameters, slingingRule):
    r = v[0:3]
    velocity = v[3:]
    drdt = velocity
    d = la.norm(r) - parameters.equilibriumLength
    #If the sim is a stretchy string
    if d < 0:
        d = 0
    dvdt = parameters.gravity - d*parameters.k*(r/la.norm(r))/parameters.mass + dragVector(velocity, parameters)/parameters.mass
    #print dragVector(velocity, parameters)/parameters.mass
    out = np.concatenate((drdt, dvdt))
    return out

def dragVector(velocity, parameters):
    dragParameter = lambda x: -0.5*parameters.airDensity*abs(x)*x*parameters.dragCoefficient*parameters.area
    drag = np.array([(dragParameter(direction)) for direction in velocity])
    return drag

#A function that will update the vpython sim
def visualize(vals, parameters):
    """Updates the vpython sim with current values"""
    r = vals[0:3]
    velocity = vals[3:]
    parameters.energyTracker.append(Spidey.energyCalculator(velocity,r,parameters))


def initVisualization(r0, parameters):
    """Initializes the vpython visualization"""
    scene.forward = (0,1,0)
    scene.range = (100,100,100)
    scene.center = (10,0,30)
    scene.autoscale = True
    parameters.rod = cylinder(pos=(0,0,0), axis=(1,0,0), radius=0.05)
    parameters.ball = sphere(pos=(1,0,0), radius = 0.1)

def initFalseVisualization(r0, parameters):
    class Bunch:
        def __init__(self, **kwds):
            self.__dict__.update(kwds)
    parameters.rod = Bunch(pos=(0,0,0), axis=(1,0,0), radius=0.05)
    parameters.ball = Bunch(pos=(1,0,0), radius = 0.1)
