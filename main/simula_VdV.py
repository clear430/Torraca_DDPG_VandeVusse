#Funções simulando o comportamento do reator de Van De Vusse

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import math

CBMAX = 1.13

# function that returns dy/dt
def modelVdV(y,t,Q,Tk,Tin,CAin): #Tk in °C, Tin in °C, Q in L/h, CAin in molA/L
    CA, CB, T= y
    #fixed parameters
    V= 10 # in L
    k10= 1.287E12 # in h-1
    k20= 1.287E12 # in h-1
    k30= 9.043E9 # in L/molA.h
    mE1R= -9758.3 # in K
    mE2R= -9758.3 # in K
    mE3R= -8560.0 # in K
    mdeltaHAB= -4.20 # in kJ/molA
    mdeltaHBC= 11.0 # in kJ/molB
    mdeltaHAD= 41.85 # in kJ/molA
    rho= 0.9342 # in kg/L
    Cp= 3.01 # in kJ/kg.K
    Kw= 4032.0 # in kJ/h.K.m2
    Ar= 0.215 # in m2
    #differential equations
    dydt=[(Q/V)*(CAin-CA) - k10*np.exp( mE1R/( T+273.15 ) )*CA - k30*np.exp( mE3R/(T+273.15) )*CA**2,
          -(Q/V)*CB      +k10*np.exp( mE1R/( T+273.15 ) )*CA - k20*np.exp( mE2R/(T+273.15) )*CB,
          1/rho/Cp*( k10*np.exp( mE1R/( T+273.15 ) )*CA*mdeltaHAB + k20*np.exp( mE2R/( T+273.15 ) )*CB*mdeltaHBC + k30*np.exp( mE3R/( T+273.15 ) )*(CA**2)*mdeltaHAD) + Q/V*(Tin-T) +Kw*Ar/(rho*Cp*V)*(Tk-T)]
    return dydt

def stepVdV(deltaT,s,Q,Tk,Tin,CAin,a,FBSP):

    if Q < 200 or Tk < 20 or Q > 1750 or Tk > 500:
        terminal = 1
    else:
        terminal = 0
    #print(terminal)

    #let's integrate our model
    # time points
    t = np.linspace(0,deltaT,2)

    # solve ODE
    s_and_s2 = odeint(modelVdV,s,t,args=(Q + a[0],Tk + a[1],Tin,CAin))
    s2 = s_and_s2[-1,:]

    #r = reward(FBSP,Q, a, s[1], s2[1])
    r = reward(FBSP,Q, a, s[1], s2[1])

    return s2, r, terminal

def reward(FBSP, Q, a, CB, CB2):

    X_T_1 = np.array([(Q + a[0]) *CB2, CB2])

    X_SP = np.array([FBSP, CBMAX])

    U = np.array([[1.0, 0],[0, 1.0]])

    A_T = np.array([a[0], a[1]])

    D = np.array([[1.0, 0],[0, 1.0]])

    X_D = (np.array([0.8e-5, 1.0]) * X_T_1) - (np.array([0.8e-5, 1.0]) * X_SP)

    X_R = (X_D.T @ U @ X_D) + (A_T.T @ D @ A_T)

    r = math.exp(-X_R)

    return r



# # find steady steate using fsolve
#
#
# yss= fsolve(modelVdV,y0,args=(0,Q,Tk,Tin,CAin))
# print(yss)
#
# # solve ODE
# y = odeint(modelVdV,y0,t,args=(Q,Tk,Tin,CAin))
# print(y)
# # plot results
# plt.plot(t,y[:,1])
# plt.xlabel('time')
# plt.ylabel('y(t)')
#
# plt.show()
