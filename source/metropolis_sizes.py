import numpy as np
import matplotlib.pyplot as plt
from fising2D import simulation, conf_rnd, oneSweep2D
from numba import jit


Tinitial=2.5 
Tfinal=1.5

Tc=2/np.log(1+np.sqrt(2))

t_term=int(1e3) # Numero di MCS necessari affinchè la CDM campioni la distribuzione canonica
t_step=int(1e3) # MCS di misura
L=[20,40,60] # Taglia del sistema: N*N spin

#liste valori per taglie diverse
m_size=[]
e_size=[]
susc_size=[]
c_size=[]

for N in L:

    T=Tinitial

    #liste per i grafici
    m_list=[] 
    e_list=[]
    susc_list=[]
    c_list=[]

    conf=conf_rnd(N)

    while(T>Tfinal):
        E_mean,M_mean,susceptibility,specific_heat=simulation(conf,T,t_term,t_step,oneSweep2D)
        m_list.append([np.abs(M_mean),T])
        e_list.append([E_mean,T])
        susc_list.append([susceptibility,T])
        c_list.append([specific_heat,T])
        T-=0.01
    
    m_size.append(m_list)
    e_size.append(e_list)
    susc_size.append(susc_list)
    c_size.append(c_list)


m=np.array(m_size)
e=np.array(e_size)
susc=np.array(susc_size)
c=np.array(c_size)

print(m)

for i in range(len(L)):
    plt.plot(m[i,:,1],m[i,:,0],label="L = {}".format(L[i]))
    plt.xlabel('Temperature')
    plt.ylabel('Magnetisation')
    plt.legend()
plt.show()

for i in range(len(L)):
    plt.plot(e[i,:,1],e[i,:,0],label="L = {}".format(L[i]))
    plt.xlabel('Temperature')
    plt.ylabel('energy')
    plt.legend()
plt.show()

for i in range(len(L)):
    plt.plot(susc[i,:,1],susc[i,:,0],label="L = {}".format(L[i]))
    plt.yscale('log')
    plt.xlabel('Temperature')
    plt.ylabel('Susceptibility')
    plt.legend()
plt.show()

for i in range(len(L)):
    plt.plot(c[i,:,1],c[i,:,0],label="L = {}".format(L[i]))
    plt.xlabel('Temperature')
    plt.ylabel('specific heat')
    plt.legend()
plt.show()