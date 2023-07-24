import numpy as np
import matplotlib.pyplot as plt
from numba import jit

# look-up table
@jit
def LUT(T):
    prob=np.zeros(5)
    for i in range (2,5,2):
        prob[i]=np.exp(-2*i/T)
    return prob


def configurazione_iniziale(N):
    # Crea una matrice quadrata di dimensione N; ogni elemento assume valore 1 o -1 con la stessa probabilità p=1/2
    spinvariable=np.array([1,-1])
    return np.random.choice(spinvariable,size=(N,N))

@jit
def energy(conf): # Energia di una configurazione
    # Funzione che ha in input la configurazione iniziale (una matrice) e calcola la corrispondente energia
    N = conf.shape[0] # la funzione shape restituisce le le dimensioni dell'oggetto; per una matrice restituisce numero di righe e colonne. Per specificare che vogliamo le righe si usa shape([0])
    E=0 
    for i in range (N):
        for j in range (N):
            heff=conf[(i-1)%N,j]+conf[(i+1)%N,j]+conf[i,(j+1)%N]+conf[i,(j-1)%N] # PBC
            E=E-heff*conf[i,j]
    return(E/2)
@jit
def oneSweep2D(conf,T):# T temperatura
    # Realizzo un MCS selezionando gli spin in modo sequenziale (MCS sta per MonteCarlo sweep; in un MCS si propone una modifica di ogni spin)
    N=conf.shape[0]

    for i in range(N):
        for j in range(N):
            heff=conf[(i-1)%N,j]+conf[(i+1)%N,j]+conf[i,(j+1)%N]+conf[i,(j-1)%N]
            #deltaE=2*heff*conf[i,j]
            sum=heff*conf[i,j]
            prob=LUT(T)
            if (sum<=0) or np.random.rand()<prob[sum]:                        
                conf[i,j]=-1.*conf[i,j]
    return conf


def simulation(T,conf,t_eq,t_mis,N):
    # ciclo per raggiungere la distribuzione asintotica di equilibrio
    m_mis=np.empty(t_mis)
    m2_mis=np.empty(t_mis)
    m4_mis=np.empty(t_mis)

    for i in range(t_eq):
        conf=oneSweep2D(conf,T)
    for k in range(t_mis):
        conf=oneSweep2D(conf,T)
        m_mis[k]=np.sum(conf)/N**2
        m2_mis[k]=m_mis[k]**2
        m4_mis[k]=m_mis[k]**4
        
    m_mean = np.mean(m_mis)
    m2_mean = np.mean(m2_mis)
    m4_mean = np.mean(m4_mis)
    return m2_mean, m4_mean

def Binder(m2, m4):
    return 0.5*(3-m4/(m2)**2)

#######################################################################################################################

Tinitial=3.0 
Tfinal=2.0
Tc=2/np.log(1+np.sqrt(2))

t_term=int(1e4) # Numero di MCS necessari affinchè la CDM campioni la distribuzione canonica
t_step=int(1e5) # MCS di misura
n=np.array([10,20,40]) # Taglia del sistema: N*N spin


#liste per i grafici
 
B_all = []

# simulazione per vari valori di N

for N in n:
    conf=configurazione_iniziale(N)
    B_list=[]
    T=Tinitial
    while(T>Tfinal):
        m2,m4=simulation(T,conf,t_term,t_step,N)
        B_list.append([Binder(m2,m4),T])
        T-=0.01
    B_all.append(B_list)




# PLOT

B = np.array(B_all)
print(B)

for i in range(len(n)):
    plt.plot(B[i,:,1],B[i,:,0], label="L = {}".format(n[i]))
plt.xlabel("T")
plt.ylabel("B(T,L)")
plt.legend(loc="upper right")
plt.axvline(Tc, linestyle="--", color="k")
plt.savefig("binder.png")
plt.show()