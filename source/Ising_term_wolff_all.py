import numpy as np
import matplotlib.pyplot as plt
from numba import jit

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
def energy(conf): # Energia di una configurazione
    # Funzione che ha in input la configurazione iniziale (una matrice) e calcola la corrispondente energia
    N = conf.shape[0] # la funzione shape restituisce le le dimensioni dell'oggetto; per una matrice restituisce numero di righe e colonne. Per specificare che vogliamo le righe si usa shape([0])
    E=0 
    for i in range (N):
        for j in range (N):
            heff=conf[(i-1)%N,j]+conf[(i+1)%N,j]+conf[i,(j+1)%N]+conf[i,(j-1)%N] # PBC
            E=E-heff*conf[i,j]
    return(E/2)

def fconf_rnd(N):
    # Crea una matrice quadrata di dimensione N; ogni elemento assume valore 1 o -1 con la stessa probabilità p=1/2
    spinvariable=np.array([1,-1])
    return np.random.choice(spinvariable,size=(N,N))


def fconf_up(N):
   return np.ones((N,N), dtype=int)

def fconf_down(N):
    return -np.ones((N,N), dtype=int)

N=100
##################################################################################

#definisco tre diverse configurazioni iniziali

conf_up = np.ones((N,N), dtype=int)
conf_down = -np.ones((N,N), dtype=int)
conf_rnd = fconf_rnd(N)

# definisco vettore di temperature, alle quali verrà fatta la termalizzazione per i tre sistemi
# con temperatura inizialmente maggiore di quella critica e poi via via sempre più vicina

Temp=np.array([2.5,2.4,2.3,2.27])


# liste di valori definite in modo molto poco elegante per ogni configurazione

m_list_up=[]
t_list_up=[]
e_list_up=[]
m_list_down=[]
t_list_down=[]
e_list_down=[]
m_list_rnd=[]
t_list_rnd=[]
e_list_rnd=[]


# inizia la parte delle simulazioi per ogni configurazione definita e con varie temperature

# configurazione tutti up

for T in Temp:
    conf_up = np.ones((N,N), dtype=int)
    measTime = 0
    m_temp=[]
    t_temp=[]
    e_temp=[]
    for t in range(1000):
        if t == measTime:
            t_temp.append(t)
            m_temp.append(np.sum(conf_up)/N**2)
            e_temp.append(energy(conf_up)/N**2)
            # faccio la misura per tempi che incremento esponenzialmente
            measTime = int(measTime * 1.1 + 1)
        conf_up = wolff2(conf_up, T)
    e_list_up.append(e_temp)
    m_list_up.append(m_temp)
    t_list_up.append(t_temp)


# configurazione tutti down

for T in Temp:
    conf_down = -np.ones((N,N), dtype=int)
    measTime = 0
    m_temp=[]
    t_temp=[]
    e_temp=[]
    for t in range(1000):
        if t == measTime:
            m_temp.append(np.sum(conf_down)/N**2)
            e_temp.append(energy(conf_down)/N**2)
            # faccio la misura per tempi che incremento esponenzialmente
            measTime = int(measTime * 1.1 + 1)
        conf_down = wolff2(conf_down, T)
    e_list_down.append(e_temp)
    m_list_down.append(m_temp)

# configurazione tutti psin casuali

for T in Temp:
    conf_rnd = fconf_rnd(N)
    measTime = 0
    m_temp=[]
    t_temp=[]
    e_temp=[]
    for t in range(1000):
        if t == measTime:
            m_temp.append(np.sum(conf_rnd)/N**2)
            e_temp.append(energy(conf_rnd)/N**2)
            # faccio la misura per tempi che incremento esponenzialmente
            measTime = int(measTime * 1.1 + 1)
        conf_rnd = wolff2(conf_rnd, T)
    e_list_rnd.append(e_temp)
    m_list_rnd.append(m_temp)


# ridefinisco le liste come array per comodità nel plot

t=np.array(t_list_up)
m_up=np.array(m_list_up)
e_up=np.array(e_list_up)
m_down=np.array(m_list_down)
e_down=np.array(e_list_down)
m_rnd=np.array(m_list_rnd)
e_rnd=np.array(e_list_rnd)

#plot del tempo di termalizzazione della configurazione up per varie temperature

for i in range(len(Temp)):
    plt.plot(t[i], m_up[i], label="T = {}".format(Temp[i]))
plt.xscale("symlog")
plt.xlabel("time")
plt.ylabel("m(+)")
plt.title("Magnetizzazione")
plt.legend()
plt.show()

# plot tempo termalizzazione energie (poco interessante)

for i in range(len(Temp)):
    plt.plot(t[i], e_up[i], label="T = {}".format(Temp[i]))
plt.xscale("symlog")
plt.xlabel("time")
plt.ylabel("e(+)")
plt.title("Energie")
plt.legend()
plt.show()

# confronto tra le tre simulazioni per la stessa temperatura
for i in range(len(Temp)):
    plt.title("Energia e Magnetizzazione a T = {}".format(Temp[i])) 
    plt.plot(t[i], m_up[i], label="m(+)")
    plt.plot(t[i], e_up[i], label="e(+)")
    plt.plot(t[i], m_down[i], label="m(-)")
    plt.plot(t[i], e_down[i], label="e(-)")
    plt.plot(t[i], m_rnd[i], label="m(0)")
    plt.plot(t[i], e_rnd[i], label="e(0)")
    plt.xscale("symlog")
    plt.xlabel("time")
    plt.ylabel("e     m")
    plt.legend(loc="upper right", ncol=3)
    plt.show()

