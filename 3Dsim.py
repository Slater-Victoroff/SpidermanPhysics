from __future__ import division
from visual import *
import numpy as np
from numpy import linalg as la
from scipy.integrate import ode
from euler import euler
from time import clock, sleep

#Some parameters for the simulation - this way, they only get defined once
class Params(object):
    def __init__(self):
        self.eq = 1 #m
        self.m = 1 #kg
        self.k = 100 #N/m
        self.g = np.array((0,-9.81,0)) #m/s^2

# The function that implements the spring pendulum odes
def spodes(t, v, p):
    r = v[0:3]
    v = v[3:]
    drdt = v
    d = la.norm(r) - p.eq
    #If the sim is a stretchy string
    if d < 0:
        d = 0
    dvdt = p.g - d*p.k*(r/la.norm(r))
    out = np.concatenate((drdt, dvdt))
    return out

#A function that will update the vpython sim
def visualize(vals, p):
    r = vals[0:3]
    v = vals[3:]
    rod.axis=r
    ball.pos=r
    # Printing the current total energy
    #E = p.m/2 * la.norm(v)**2 + p.k/2*(la.norm(r)-p.eq)**2 - p.m * np.dot(p.g,r)
    E = p.m/2 * la.norm(v)**2 - p.m * np.dot(p.g,r)
    if la.norm(r)-p.eq > 0:
        E += p.k/2*(la.norm(r)-p.eq)**2
    print E

#Instantiating the parameters
params = Params()

#Defining sp as a function only of t and v
sp = (lambda t,v: spodes(t,v,params))

#Setting up the vpython stuff
rod = cylinder(pos=(0,0,0), axis=(1,0,0), radius=0.05)
ball = sphere(pos=(1,0,0), radius = 0.1)

# creating the sim object, dopri5 is basically ode45
sim = ode(sp).set_integrator('dopri5')
sim.set_initial_value([1,0,0,0,0,0],0)
endtime = 10
dt = 0.01

#This follows the form of the example on the scipy site
while sim.successful() and sim.t < endtime:
    sim.integrate(sim.t+dt)
    sleep(dt)
    visualize(sim.y, params)
