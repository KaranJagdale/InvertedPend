import numpy as np
#define constants
g = 9.80665 #acceleration due to gravity
m = 1000    #mass of the bar
l = 50      #length of the bar

def dynamics(Theta, Tau):
    return 1.5*g/l*np.sin(Theta) - Tau*3/m/l**2

