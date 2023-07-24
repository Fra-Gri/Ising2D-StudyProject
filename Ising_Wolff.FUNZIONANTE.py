import numpy as np
import matplotlib.pyplot as plt
from numba import jit


def configurazione_iniziale(N):
    # Crea una matrice quadrata di dimensione N; ogni elemento assume valore +1 o -1 con la stessa probabilità p=1/2
    spinvariable=np.array([1,-1])
    return np.random.choice(spinvariable,size=(N,N))

@jit
def energy(conf): # Energia di una configurazione
    # Funzione che ha in input la configurazione iniziale (una matrice) e calcola la corrispondente energia
    N = conf.shape[0] 
    E=0 
    for i in range (N):
        for j in range (N):
            heff=conf[(i-1)%N,j]+conf[(i+1)%N,j]+conf[i,(j+1)%N]+conf[i,(j-1)%N] # PBC
            E=E-heff*conf[i,j]
    return(E/2)


@jit
def wolff2(conf,T):
    N=conf.shape[0]
    i, j = np.random.randint(0,N), np.random.randint(0,N)
    spin_seed= conf[i,j]
    cluster = [[i,j]]
    old_spin = [[i,j]]
    p_add = 1. - np.exp(-2/T)

    while (len(old_spin) != 0) :
        new_spin = []

        for i,j in old_spin:
            nn = [[(i+1)%N,j], [(i-1)%N,j], [i,(j+1)%N], [i,(j-1)%N]]

            for state in nn:
                if conf[state[0],state[1]] == spin_seed and state not in cluster:
                    if np.random.rand() < p_add:
                        new_spin.append(state)
                        cluster.append(state)

        old_spin= new_spin
        

    for i,j in cluster:
        conf[i,j] *= -1
    return conf
    
@jit
def simulation(T,conf,t_eq,t_mis):
    # ciclo per raggiungere la distribuzione asintotica di equilibrio
    E=np.empty(t_mis)
    M=np.empty(t_mis)
    for i in range(t_eq):
       
        conf=wolff2(conf,T)
    for k in range(t_mis):
        
        conf=wolff2(conf,T)
        E[k]=energy(conf)
        M[k]=np.abs(np.sum(conf))
    
    E_mean=np.mean(E)
    M_mean=np.mean(M)
    #susceptibility=(1/T)*np.std(M)**2/(N*N)
    susceptibility=(1/T)*np.var(M)/(N**2)
    specific_heat=(1/T)**2*np.var(E)/(N**2)
    return E_mean,M_mean,susceptibility,specific_heat

#########################################################################################################################################################

Tinitial=3.5 
Tfinal=1.0
Tc=2/np.log(1+np.sqrt(2))

t_eq=int(1e3) # Numero di MCS necessari affinchè la CDM campioni la distribuzione canonica
t_mis=int(1e3) # MCS di misura
N=20 # Taglia del sistema: N*N spin


#liste per i grafici
m_list=[] 
e_list=[]
susc_list=[]
c_list=[]

# Seleziono una configurazione per partire con la simulazione
conf=configurazione_iniziale(N)
# simulazione

while(Tinitial>Tfinal):
    E_mean,M_mean,susceptibility,specific_heat=simulation(Tinitial,conf,t_eq,t_mis)
    m_list.append([np.abs(M_mean)/(N**2),Tinitial])
    e_list.append([E_mean/(N**2),Tinitial])
    susc_list.append([susceptibility,Tinitial])
    c_list.append([specific_heat,Tinitial])
    Tinitial-=0.1

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