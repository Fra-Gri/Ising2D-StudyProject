import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as sm
from numba import jit
from scipy.optimize import curve_fit




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
            if (sum<=0) or np.random.rand()<prob[sum]: # potremmo ottimizzare l'algoritmo con una look-up table                         
                conf[i,j]=-1.*conf[i,j]
    return conf

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
    N=conf.shape[0]
    E=np.empty(t_mis)
    M=np.empty(t_mis)
    for i in range(t_eq):
        conf=oneSweep2D(conf,T)
        #conf=wolff2(conf,T)
    for k in range(t_mis):
        conf=oneSweep2D(conf,T)
        #conf=wolff2(conf,T)
        E[k]=energy(conf)
        M[k]=np.sum(conf)

    
    E_mean=np.mean(E)
    M_mean=np.mean(M)
    #susceptibility=(1/T)*np.std(M)**2/(N*N)
    susceptibility=(1/T)*np.var(M)/(N**2)
    specific_heat=(1/T)**2*np.var(E)/(N**2)
    
    #return E_mean,M_mean,susceptibility,specific_heat
    
    return M

'''
def autocorrelazione(A,delta_t):
    N=A.shape[0]
    sum1=0
    sum2=0
    sum3=0
    for i in range (0,N-delta_t,delta_t):
        sum1+=A[i]*A[i+delta_t]
        sum2+=A[i]
        sum3+=A[i+delta_t]
    
    sum1=sum1/(N-delta_t)
    sum2=sum2/(N-delta_t)
    sum3=sum3/(N-delta_t)
    
    return sum1-sum2*sum3
'''


def func(x,a,tau,c):
    return a * np.exp(-x/tau) + c  

def func2(x,m,q):
    return m*x+q
#######################################################################################################################################



N=[10,20,30,40,50,60,70,80,90,100]
Tc=2/np.log(1+np.sqrt(2))
t_eq=int(1e3)
t_mis=int(1e5)
t_max=int(1e4)# lag massimo
tau=[]
tau2=[]
for i in range (0,len(N)):
    conf=configurazione_iniziale(N[i])
    M=simulation(Tc,conf,t_eq,t_mis)
    M_autocorr=sm.acf(M,True,nlags=t_max,fft=True)
    xdata=np.linspace(0,t_max,t_max+1)
    popt, pcov = curve_fit(func,xdata,  M_autocorr)
    tau.append(np.sum(M_autocorr))
    tau2.append(popt[1])
    print(popt,'    ',np.diag(pcov),'   ',np.sum(M_autocorr),'\n')


N=np.array(N)
tau2=np.array(tau2)
sizes=np.linspace(10,80,8)
popt2,pcov2=curve_fit(func2,np.log10(sizes),np.log10(tau2))
print('\n',popt2,np.diag(pcov2))
plt.plot(N,tau2,'o')
plt.xscale('log')
plt.yscale('log')
plt.show()
'''
E,M=simulation(T[1],conf,t_eq,t_mis)
M_autocorr=sm.acf(M,True,nlags=t_max,fft=True)
plt.plot(M_autocorr, label='T='+str(T[1]))  
plt.xlabel('deltaT')
plt.ylabel('Magnetisation Autocorellation Function')
plt.legend()
plt.title("Size="+str(N))
plt.show()
xdata=np.linspace(0,t_max,t_max+1)
popt, pcov = curve_fit(func,xdata,  M_autocorr)
print(popt,'\n',np.diag(pcov),'\n',np.sum(M_autocorr))
'''