import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import os
my_path = os.path.dirname(__file__)

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

N=100

Temp=np.array([2.8, 2.6, 2.4, 2.3])


m_list=[]
t_list=[]
m = []

for T in Temp:
    conf_up = np.ones((N,N), dtype=int)
    measTime = 0
    m_temp=[]
    t_temp=[]
    for t in range(1000):
        if t == measTime:
            t_temp.append(t)
            m_temp.append(np.abs(np.sum(conf_up))/N**2)
            # e_list.append(energy(conf_up))
            # faccio la misura per tempi che incremento esponenzialmente
            measTime = int(measTime * 1.025 + 1)
        conf_up = wolff2(conf_up, T)
    m_list.append(m_temp)
    t_list.append(t_temp)


m=np.array(m_list)
t=np.array(t_list)
#e=np.array(e_list)

print(m)

#plot magnetization to respect the time

for i in range(len(Temp)):
    plt.plot(t[i], m[i], label="T = {}".format(Temp[i]))
plt.xscale("log")
plt.xlabel("time")
plt.ylabel("m")
plt.legend(loc='upper right')
plt.savefig(os.path.join(my_path,"Immagini/termalization_wolff.png"))
plt.show()
