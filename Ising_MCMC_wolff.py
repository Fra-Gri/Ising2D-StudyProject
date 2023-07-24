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
def neighbours(state,N):
      '''
      Input: indici di uno spin
      Output: matrice 4*2 contenente gli indici dei primi vicini

      '''
      i=state[0]
      j=state[1]
      nn=np.array([[(i-1)%N,j],[(i+1)%N,j],[i,(j+1)%N],[i,(j-1)%N]]) # matrice 4*2 contenente gli indici dei primi vicini di state
      return nn


def wolff(conf,T,seed):

    N=conf.shape[0]
    new_spin=[] # lista 
    P_add=1-np.exp(-2/T)
    cluster=[[seed[0],seed[1]]]
    new_spin=cluster
    new_spin_temp=[]
    while(len(new_spin)!=0):
        for elem in new_spin:
            nn=neighbours(elem,N)
            for state in nn:
                if (conf[state[0], state[1]] == conf[seed[0],seed[1]] and [state[0],state[1]] not in cluster) and np.random.uniform(0., 1.) < P_add:
            
                    new_spin_temp.append([state[0], state[1]])
            new_spin=new_spin_temp
            new_spin_temp=[]
            cluster=cluster+new_spin
    for elem in cluster:
        conf[elem[0],elem[1]]=-1*conf[elem[0],elem[1]]

    return conf

def simulation(T,conf,t_eq,t_mis):
    # ciclo per raggiungere la distribuzione asintotica di equilibrio
    E=np.empty(t_mis)
    M=np.empty(t_mis)
    for i in range(t_eq):
        seed=[np.random.randint(0,N),np.random.randint(0,N)]
        conf=wolff(conf,T,seed)
    for k in range(t_mis):
        seed=[np.random.randint(0,N),np.random.randint(0,N)]
        conf=wolff(conf,T,seed)
        E[k]=energy(conf)
        M[k]=np.sum(conf)
    
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

t_eq=int(1e5) # Numero di MCS necessari affinchè la CDM campioni la distribuzione canonica
t_mis=int(1e4) # MCS di misura
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
