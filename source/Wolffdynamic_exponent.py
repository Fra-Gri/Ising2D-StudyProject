import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as sm
from func_ising import conf_rnd, simulation, wolff
from numba import jit
from scipy.optimize import curve_fit


def func(x,a,tau,c):
    return a * np.exp(-x/tau) + c  

def func2(x,m,q):
    return m*x+q


N=[10,20,30,40,50,60,70,80,90,100]
Tc=2/np.log(1+np.sqrt(2))
t_eq=int(1e3)
t_mis=int(1e5)
t_max=int(100)# lag massimo
tau=[]
tau2=[]

for i in range (0,len(N)):
    conf=conf_rnd(N[i])
    M=simulation(Tc,conf,t_eq,t_mis,wolff)
    M_autocorr=sm.acf(np.abs(M),True,nlags=t_max,fft=True)
    xdata=np.linspace(0,t_max,t_max+1)
    popt, pcov = curve_fit(func,xdata,  M_autocorr)
    tau.append(np.sum(M_autocorr))
    tau2.append(popt[1])
    print(popt,'    ',np.diag(pcov),'   ',np.sum(M_autocorr),'\n')


N=np.array(N)
tau2=np.array(tau2)
sizes=np.linspace(10,100,10)
popt2,pcov2=curve_fit(func2,np.log10(sizes),np.log10(tau2))
print('\n',popt2,np.diag(pcov2))
plt.plot(N,tau2,'o')

plt.xscale('log')
plt.yscale('log')
plt.ylabel("Correlation time tau_steps")
plt.xlabel("Lattice Size L")
plt.show()

