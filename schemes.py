# This script implements several standard numerical integration schemes first-order systems of the form
# 	z' = f(z)
# For simplicity, this only solves conservative systems with a potential, q'' = -phi'(q).
# As a first-order system, this is equivalent to q' = v, v' = -phi'(q). We let z = [q, v] be the
# state vector, where q is the position and v is the velocity.

import numpy as np
import matplotlib.pyplot as plt

def eulerstep(phi, z, deltat):
	nextStep = np.zeros(2)
	nextStep[0] = z[0] + deltat*z[1]
	nextStep[1] = z[1] + deltat*phi(z[0])
	return nextStep

z0 = np.array([2,0])
deltat = 0.0001
numSteps = 500000

# potential for Morse oscillator
# phi = lambda q : (1 - np.exp(-q))**2

# potential for Lennard-Jones oscillator
phi = lambda q : q**(-12) - 2*q**(-6)

soln = np.zeros((numSteps,2))
soln[0] = z0

for i in range(1,numSteps):
	soln[i] = eulerstep(phi, soln[i-1], deltat)

plt.subplot(121)
plt.xlabel("position")
plt.ylabel("velocity")
plt.title("a single trajectory")
plt.plot(np.transpose(soln)[0], np.transpose(soln)[1])

plt.subplot(122)
plt.xlabel("time")
plt.ylabel("energy")
plt.title("a single trajectory")
plt.plot(np.transpose(soln)[0], np.transpose(soln)[1])




plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=1, hspace=0.2)
plt.show()

#print(np.transpose(soln)[0])
#print(np.transpose(soln)[1])
