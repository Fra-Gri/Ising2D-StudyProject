import numpy as np
import matplotlib.pyplot as plt
from numba import jit


@jit
def LUT(T):
    # look-up table
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
    # Realizzo un MCS selezionando gli spin in modo sequenziale 
    # MCS sta per MonteCarlo sweep: in un MCS si propone una modifica di ogni spin
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


def simulation(T,conf,t_eq,t_mis):
    N=conf.shape[0]
    # ciclo per raggiungere la distribuzione asintotica di equilibrio
    E=np.empty(t_mis)
    M=np.empty(t_mis)
    for i in range(t_eq):
        conf=oneSweep2D(conf,T)
    for k in range(t_mis):
        conf=oneSweep2D(conf,T)
        E[k]=energy(conf)
        M[k]=np.sum(conf)
    
    E_mean=np.mean(E)
    M_mean=np.mean(M)
    #susceptibility=(1/T)*np.std(M)**2/(N*N)
    susceptibility=(1/T)*np.var(M)/(N**2)
    specific_heat=(1/T)**2*np.var(E)/(N**2)
    return E_mean,M_mean,susceptibility,specific_heat

#########################################################################################################################################################

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

    # Seleziono una configurazione per partire con la simulazione
    conf=configurazione_iniziale(N)

    while(T>Tfinal):
        E_mean,M_mean,susceptibility,specific_heat=simulation(T,conf,t_term,t_step)
        m_list.append([np.abs(M_mean)/(N**2),T])
        e_list.append([E_mean/(N**2),T])
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