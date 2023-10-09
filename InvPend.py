import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
#define constants
g = 9.80665 #acceleration due to gravity
m = 1    #mass of the bar
l = 0.5      #length of the bar
pi = 3.1415926535
Gamma = 0.9 #Discount factor

TauRange = 2*m*g*l  #max torque is twice of what is required to balances the mass in 90 degree configuration 

res = 101  #resolution of the control input or the totan number of discretized values of the control inpput. This value should ne odd number

ThetaRes = 501 #resolution of theta : creating the discrete state from the originally continouos state theta
                #Odd number in the resolution will ensure that the theta for the vertically up position is included in the discrete state

Ts = 0.05 #50 ms sample time

k = 0.1 #damping constant

ThetaTarget = pi #Angle at which we intend to stabilize

ThetaDisc = np.linspace(0, 2*pi, ThetaRes)
TauDisc = np.linspace(-TauRange, TauRange, res)

def thetaDDot(Theta, Tau):
    return 1.5*g/l*np.sin(Theta) - Tau*3/m/l**2

def reward(Theta, Tau, ThetaTarget):
    #1/(Theta - ThetaTarget + 0.001)
    thetadiff = abs(Theta - ThetaTarget)
    return 1/(thetadiff**2 + 0.00001)

def DynSS(y, t, m,l,k,Tau,g):
    Theta, Omega = y
    dydt = [Omega, -1.5*g/l*np.sin(Theta) + 3*Tau/m/l**2 - 3*k*Omega/m/l**2]
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
def Q_learning(Q_new, ThetaTarget):
    Theta = pi
    Omega = 0
    Tau = 0
    Iterations = 2000000
    eps = 0.1
    TauInd = np.where(TauDisc == Tau)[0][0]
    ThetaHist = [Theta]
    CritThetaCount = 0
    ThetaVisitCount = np.zeros(ThetaRes)
    for i in range(Iterations):
        #First find the next state
        Theta_n, Omega_n = nextState([Theta, Omega], Tau)
        if Theta_n >= 2*pi:
            Theta_n = Theta_n - 2*pi
        elif Theta_n < 0:
            Theta_n = Theta_n + 2*pi
        ThetaHist.append(Theta_n)

        #Update the Q function
        #First find the closest Theta value in the discretized theta to the current Theta and Theta_n
        diff = np.absolute(ThetaDisc - Theta)
        ThetaInd = np.argmin(diff)

        ThetaVisitCount[ThetaInd] = ThetaVisitCount[ThetaInd] + 1
        if ThetaInd > 90 and ThetaInd < 110:
            CritThetaCount = CritThetaCount + 1
        

        diff = np.absolute(ThetaDisc - Theta_n)
        Theta_nInd = np.argmin(diff)
        

        Q_new[ThetaInd, TauInd] = (1-eps)*Q_new[ThetaInd, TauInd] + eps*(reward(ThetaDisc[ThetaInd], Tau, ThetaTarget) + Gamma*Q_new[Theta_nInd, :].max())
        #generate a random action for the next instant
        TauInd = np.random.randint(res)
        Tau = TauDisc[TauInd]
        

        Theta, Omega = Theta_n, Omega_n
        if i%100000 == 0:
            print(i/100000)
    print(CritThetaCount/Iterations*100, "Explored critical region")
    print(ThetaVisitCount)
    return Q_new, ThetaHist

Q, ThetaHist = Q_learning(Q_new, ThetaTarget)
#print(ThetaHist)
# plt.plot(ThetaHist)  
   
# plt.show()

# Controling pendulum using the learned Q-function
Theta0 = 2
Omega0 = 0

# Want to stabilize the Theta at pi (Vertically inverted)

simIter = 10000
Theta, Omega = Theta0, Omega0
ThetaSim = [Theta]
TauSim = []
for i in range(simIter):
        
    diff = np.absolute(ThetaDisc - Theta)
    ThetaInd = np.argmin(diff)

    #Finding optimal action for the current state

    optTauInd = np.argmax(Q[ThetaInd, :])
    optTau = TauDisc[optTauInd]
    TauSim.append(optTau)
    Theta_n, Omega_n = nextState([Theta, Omega], optTau)
    if Theta_n >= 2*pi:
        Theta_n = Theta_n - 2*pi
    elif Theta_n < 0:
        Theta_n = Theta_n + 2*pi
    ThetaSim.append(Theta_n)
    Theta, Omega = Theta_n, Omega_n

plt.figure(1)
plt.plot(ThetaSim)
plt.figure(2)
plt.plot(TauSim)
plt.show()


