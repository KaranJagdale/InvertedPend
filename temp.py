import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

a = np.linspace(0,6.28,101)
print(a)

def temp():
    print(a)

temp()

b = np.array([2,4,1])
c = np.where(b == 4)[0]
print((c[0]))