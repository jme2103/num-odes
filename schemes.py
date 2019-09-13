# This script implements several standard numerical 
# integration schemes first-order systems of the form
# 	z' = f(z)
# For simplicity, this only solves conservative 
# systems with a potential, q'' = -phi'(q). As a 
# first-order system, this is equivalent to q' = v, 
# v' = -phi'(q). We let z = [q, v] be the state vector, 
# where q is the position and v is the velocity.

import numpy as np
import matplotlib.pyplot as plt

def f(phiPrime, z):
	return np.array([z[1],-1*phiPrime(z[0])])

def eulerstep(phiPrime, z, deltat):
	return z + deltat*f(phiPrime, z)

def rk4step(phiPrime, z, deltat):
	Z_1 = z
	Z_2 = z + 0.5*deltat*f(phiPrime, Z_1)
	Z_3 = z + 0.5*deltat*f(phiPrime, Z_2)
	Z_4 = z + deltat*f(phiPrime, Z_3)
	return z + (deltat/6)*(f(phiPrime, Z_1) 
		+ 2*f(phiPrime, Z_2) 
		+ 2*f(phiPrime, Z_3) 
		+ f(phiPrime, Z_4))

# Euler A, Euler B, and Stormer-Verlet methods cannot
# be written in terms of the function f, because they
# are partitioned RK methods. To step forward some of
# the variables, one needs to step forward the others

def eulerAstep(phiPrime, z, deltat):
	q = z[0] + deltat*z[1]
	v = z[1] + deltat*(-1*phiPrime(q))
	return np.array([q,v])

def eulerBstep(phiPrime, z, deltat):
	v = z[1] + deltat*(-1*phiPrime(z[0]))
	q = z[0] + deltat*v
	return np.array([q,v])

def stormerstep(phiPrime, z, deltat):
	vmid = z[1] + 0.5*deltat*(-1*phiPrime(z[0]))
	q = z[0] + deltat*vmid
	v = vmid + 0.5*deltat*(-1*phiPrime(q))
	return np.array([q,v])

def energy(phi, z):
	return z[1]**2/2 + phi(z[0])

z0 = np.array([2,0])
deltat = 0.001
numSteps = 100000
timeSteps = np.linspace(0,deltat*numSteps,numSteps+1)

# Because it's the gradient of the potential that 
# causes change in the velocity, a lambda expression 
# for both phi and its derivative are defined.

morsePot = lambda q : (1 - np.exp(-q))**2
morsePotGrad = lambda q : 2*np.exp(-q)*(1 - np.exp(-q))
LJPot = lambda q : q**(-12) - 2*q**(-6)
LJPotGrad = lambda q : -12*q**(-13) + 12*q**(-7)

eulersoln = np.zeros((numSteps,2))
rk4soln = np.zeros((numSteps,2))
eulerAsoln = np.zeros((numSteps,2))
eulerBsoln = np.zeros((numSteps,2))
stormersoln = np.zeros((numSteps,2))

eulersoln[0] = z0
rk4soln[0] = z0
eulerAsoln[0] = z0
eulerBsoln[0] = z0
stormersoln[0] = z0


eulerEnergyErr = np.zeros(numSteps+1)
rk4EnergyErr = np.zeros(numSteps+1)
eulerAEnergyErr = np.zeros(numSteps+1)
eulerBEnergyErr = np.zeros(numSteps+1)
stormerEnergyErr = np.zeros(numSteps+1)


# choose potential
phi = morsePot
phiPrime = morsePotGrad

for i in range(1,numSteps):
	eulersoln[i] = eulerstep(phiPrime, eulersoln[i-1], deltat)
	rk4soln[i] = rk4step(phiPrime, rk4soln[i-1], deltat)
	eulerAsoln[i] = eulerAstep(phiPrime, eulerAsoln[i-1], deltat)
	eulerBsoln[i] = eulerBstep(phiPrime, eulerBsoln[i-1], deltat)
	stormersoln[i] = stormerstep(phiPrime, stormersoln[i-1], deltat)

for i in range(1,numSteps+1):
	eulerEnergyErr[i] = energy(phi,eulersoln[i-1]) - energy(phi,eulersoln[0])
	rk4EnergyErr[i] = energy(phi,rk4soln[i-1]) - energy(phi,rk4soln[0])
	eulerAEnergyErr[i] = energy(phi,eulerAsoln[i-1]) - energy(phi,eulerAsoln[0])
	eulerBEnergyErr[i] = energy(phi,eulerBsoln[i-1]) - energy(phi,eulerBsoln[0])
	stormerEnergyErr[i] = energy(phi,stormersoln[i-1]) - energy(phi,stormersoln[0])

fig, axs = plt.subplots(5, 2)
axs[0,0].plot(eulersoln[:,0], eulersoln[:,1])
axs[0,1].plot(timeSteps, eulerEnergyErr)
axs[1,0].plot(rk4soln[:,0], rk4soln[:,1])
axs[1,1].plot(timeSteps, rk4EnergyErr)
axs[2,0].plot(eulerAsoln[:,0], eulerAsoln[:,1])
axs[2,1].plot(timeSteps, eulerAEnergyErr)
axs[3,0].plot(eulerBsoln[:,0], eulerBsoln[:,1])
axs[3,1].plot(timeSteps, eulerBEnergyErr)
axs[4,0].plot(stormersoln[:,0], stormersoln[:,1])
axs[4,1].plot(timeSteps, stormerEnergyErr)

plt.show()
