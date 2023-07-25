import numpy as np
import matplotlib.pyplot as plt
from fising2D import conf_rnd, simulation_binder, Binder
from numba import jit

Tinitial=3.0 
Tfinal=2.0
Tc=2/np.log(1+np.sqrt(2))

t_term=int(1e4) # Numero di MCS necessari affinchè la CDM campioni la distribuzione canonica
t_step=int(1e5) # MCS di misura
n=np.array([10,20,40]) # Taglie del sistema: N*N spin

#lista per i grafici
B_all = []

# simulazione per vari valori di N
for N in n:
    conf=conf_rnd(N)
    B_list=[]
    T=Tinitial
    while(T>Tfinal):
        m2,m4=simulation_binder(T,conf,t_term,t_step,N)
        B_list.append([Binder(m2,m4),T])
        T-=0.01
    B_all.append(B_list)


# PLOT

B = np.array(B_all)

for i in range(len(n)):
    plt.plot(B[i,:,1],B[i,:,0], label="L = {}".format(n[i]))
plt.xlabel("T")
plt.ylabel("B(T,L)")
plt.legend(loc="upper right")
plt.axvline(Tc, linestyle="--", color="k")
plt.savefig("images/binder.png")
plt.show()