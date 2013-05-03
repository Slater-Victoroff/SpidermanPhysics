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
        try:
            sim.integrate(sim.t+parameters.dt)
        except Exception as finish:
            print 'Exception Caught'
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
    dvdt = parameters.gravity - d*parameters.k*(r/la.norm(r))
    out = np.concatenate((drdt, dvdt))
    return out



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


def fireWeb(direction, parameters):
    """Feed in the parameters of the system and the direction that the web is being
    fired in and it should return an appropriate"""


#A function that will update the vpython sim
def visualize(vals, parameters):
    r = vals[0:3]
    velocity = vals[3:]
    rod.axis=r
    ball.pos=r
    # Printing the current total energy
    #E = p.m/2 * la.norm(v)**2 + p.k/2*(la.norm(r)-p.eq)**2 - p.m * np.dot(p.g,r)
    E = parameters.mass/2 * la.norm(velocity)**2 - parameters.mass * np.dot(parameters.gravity,r)
    if la.norm(r)-parameters.equilibriumLength > 0:
        E += parameters.k/2*(la.norm(r)-parameters.equilibriumLength)**2
    print E

#Instantiating the parameters

#rod = cylinder(pos=(0,0,0), axis=(1,0,0), radius=0.05)
#ball = sphere(pos=(1,0,0), radius = 0.1)
