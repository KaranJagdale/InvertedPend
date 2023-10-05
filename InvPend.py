import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
#define constants
g = 9.80665 #acceleration due to gravity
m = 1    #mass of the bar
l = 0.5      #length of the bar
pi = 3.1415926535
Gamma = 0.9 #Discount factor

TauRange = m*g*l/2*np.sin(pi/4)  #max torque is such that it balances the mass in 45 degree configuration 

res = 3  #resolution of the control input or the totan number of discretized values of the control inpput

ThetaRes = 101 #resolution of theta : creating the discrete state from the originally continouos state theta
                #Odd number in the resolution will ensure that the theta for the vertically up position is included in the discrete state

Ts = 0.05 #50 ms sample time

k = 0.05 #damping constant

ThetaDisc = np.linspace(0, 2*pi, ThetaRes)
TauDisc = np.linspace(-TauRange, TauRange, res)

def thetaDDot(Theta, Tau):
    return 1.5*g/l*np.sin(Theta) - Tau*3/m/l**2

def reward(Theta, Tau):
    return 1/(Theta - pi + 0.001)

def DynSS(y, t, m,l,k,Tau,g):
    Theta, Omega = y
    dydt = [Omega, -1.5*g/l*np.sin(Theta) - 3*Tau/m/l**2 - 3*k*Omega/m/l**2]
    return dydt
#y0 = [pi,0]
# Tau = 0
# sol = odeint(DynSS, y0, [0, Ts, 2*Ts], args=(m,l,k,Tau,g))
# print(sol[:,0])
def nextState(y,Tau):
    y0 = y
    sol = odeint(DynSS, y0, [0, Ts], args=(m,l,k,Tau,g))
    return sol[1,:]
#a =nextState([0,pi],0)
#print(a)

Q_new = np.zeros((ThetaRes, res))
def Q_learning(Q_new):
    Theta = pi
    Omega = 0
    Tau = 0
    Iterations = 1000000 
    eps = 0.1
    TauInd = np.where(TauDisc == Tau)[0][0]
    ThetaHist = [Theta]
    for i in range(Iterations):
        #First find the next state
        Theta_n, Omega_n = nextState([Theta, Omega], Tau)
        ThetaHist.append(Theta_n)

        #Update the Q function
        #First find the closest Theta value in the discretized theta to the current Theta and Theta_n
        diff = np.absolute(ThetaDisc - Theta)
        ThetaInd = np.argmin(diff)

        diff = np.absolute(ThetaDisc - Theta_n)
        Theta_nInd = np.argmin(diff)
        

        Q_new[ThetaInd, TauInd] = (1-eps)*Q_new[ThetaInd, TauInd] + eps*(reward(ThetaDisc[ThetaInd], Tau) + Gamma*Q_new[Theta_nInd, :].max())
        #generate a random action for the next instant
        TauInd = np.random.randint(res)
        Tau = TauDisc[TauInd]
        Theta, Omega = Theta_n, Omega_n
        if i%100000 == 0:
            print(i)
    return Q_new, ThetaHist

Q, ThetaHist = Q_learning(Q_new)
plt.plot(ThetaHist)        
plt.show()





#sol = odeint(DynSS, y0, t, args=(m,l,k,Tau,g))



'''
plt.plot(t, sol[:, 0], 'b', label='theta(t)')

plt.plot(t, sol[:, 1], 'g', label='omega(t)')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.show()
'''
