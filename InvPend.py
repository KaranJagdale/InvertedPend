import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
#define constants
g = 9.80665 #acceleration due to gravity
m = 1    #mass of the bar
l = 0.5      #length of the bar
TauRange = 1  
Ts = 0.05 #50 ms sample time
k = 0.005 #damping constant
pi = 3.1415926535
print("Hey")
def thetaDDot(Theta, Tau):
    return 1.5*g/l*np.sin(Theta) - Tau*3/m/l**2

def reward(Theta):
    return 1/(Theta + 0.001)

def DynSS(y, t, m, l, k, Tau, g):
    Theta, Omega = y
    dydt = [Omega, -1.5*g/l*np.sin(Theta) - 3*Tau/m/l**2 - 3*k*Omega/m/l**2]
    return dydt

y0 = [pi/2, 0]
t= [0, 10, 100]
Tau =0
sol = odeint(DynSS, y0, t, args=(m,l,k,Tau,g))
#print(type(sol))
print('Karan')

plt.plot(t, sol[:, 0], 'b', label='theta(t)')

plt.plot(t, sol[:, 1], 'g', label='omega(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()
