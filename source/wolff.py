import numpy as np
import matplotlib.pyplot as plt
from fising2D import conf_rnd, wolff, simulation
from numba import jit

Tinitial=3.5 
Tfinal=1.0
T=Tinitial
Tc=2/np.log(1+np.sqrt(2))

t_eq=int(1e5) # Numero di MCS necessari affinchè la CDM campioni la distribuzione canonica
t_mis=int(1e4) # MCS di misura
N=20 # Taglia del sistema: N*N spin


#liste per i grafici
m_list=[] 
e_list=[]
susc_list=[]
c_list=[]

# Seleziono una configurazione per partire con la simulazione
conf=conf_rnd(N)

# simulazione
while(T>Tfinal):
    E_mean,M_mean,susceptibility,specific_heat=simulation(conf, T, t_eq, t_mis, wolff)
    m_list.append([np.abs(M_mean)/(N**2),T])
    e_list.append([E_mean/(N**2),T])
    susc_list.append([susceptibility,T])
    c_list.append([specific_heat,T])
    T-=0.1

m=np.array(m_list)
e=np.array(e_list)
susc=np.array(susc_list)
c=np.array(c_list)


plt.plot(m[:,1],m[:,0],'o')
plt.xlabel('Temperature')
plt.ylabel('Magnetisation')
plt.show()
plt.plot(e[:,1],e[:,0],'o')
plt.xlabel('Temperature')
plt.ylabel('energy')
plt.show()
plt.plot(susc[:,1],susc[:,0],'o')
plt.yscale('log')
plt.xlabel('Temperature')
plt.ylabel('Susceptibility')

plt.show()
plt.plot(c[:,1],c[:,0],'o')
plt.xlabel('Temperature')
plt.ylabel('specific heat')
plt.show()
