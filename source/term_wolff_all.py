import numpy as np
import matplotlib.pyplot as plt
from fising2D import conf_rnd, wolff, energy
from numba import jit


N=100

#definisco tre diverse configurazioni iniziali

conf_up = np.ones((N,N), dtype=int)
conf_down = -np.ones((N,N), dtype=int)
conf_rnd = conf_rnd(N)

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
        conf_up = wolff(conf_up, T)
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
        conf_down = wolff(conf_down, T)
    e_list_down.append(e_temp)
    m_list_down.append(m_temp)

# configurazione tutti psin casuali

for T in Temp:
    conf0 = conf_rnd(N)
    measTime = 0
    m_temp=[]
    t_temp=[]
    e_temp=[]
    for t in range(1000):
        if t == measTime:
            m_temp.append(np.sum(conf0)/N**2)
            e_temp.append(energy(conf0)/N**2)
            # faccio la misura per tempi che incremento esponenzialmente
            measTime = int(measTime * 1.1 + 1)
        conf0 = wolff(conf0, T)
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

