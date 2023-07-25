import numpy as np
import matplotlib.pyplot as plt
from fising2D import wolff
from numba import jit


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
            # faccio la misura per tempi che incremento esponenzialmente
            measTime = int(measTime * 1.025 + 1)
        conf_up = wolff(conf_up, T)
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
plt.show()
