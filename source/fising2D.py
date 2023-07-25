import numpy as np
from numba import jit

def conf_rnd(N:int):
    '''Create an initial square configuration of the dimension given, generating every 
    element randomly between +1 or -1, with the same probability.
    
    Parameters
    ----------
    N : int
        shape of the configuration
    
    Returns
    -------
    array_like
        the random configuration created

    '''

    spin_variable=np.array([1,-1])
    return np.random.choice(spin_variable,size=(N,N))


@jit
def energy(conf:np.ndarray): 
    '''Compute the energy per spin of a configuration: for every spin compute the 
    interaction with the nearest neighbour heff, and add it to the energy E. The divide
    2 is for not counting double the interactions.
    
    
    Parameters
    ---------
    conf : array_like
        configuration of the system
    
    Returns
    -------
    E : int
        energy computed
    '''

    N = conf.shape[0] # shape of the configuration
    E=0 
    for i in range (N):
        for j in range (N):
            heff=conf[(i-1)%N,j]+conf[(i+1)%N,j]+conf[i,(j+1)%N]+conf[i,(j-1)%N] 
            E=E-heff*conf[i,j]/2
    return E


def wolff(conf:np.ndarray, T:float):
    '''Implementation of one step the Wolff algorithm:
            1 - select the seed
            2 - compute the cluster
            3 - flip the cluster
    Parameters
    ----------
    conf : array_like
        configuration of the system
    T : float
        temperature of the system
    
    Returns
    -------
    array_like
        the new configuration with the cluster flipped
    '''

    N=conf.shape[0]
    i, j = np.random.randint(0,N), np.random.randint(0,N)
    spin_seed= conf[i,j]

    cluster = [[i,j]]
    old_spin = [[i,j]]
    p_add = 1. - np.exp(-2/T)

    while (len(old_spin) != 0) :
        new_spin = []

        for i,j in old_spin:
            nn = [[(i-1)%N,j],[(i+1)%N,j],[i,(j+1)%N],[i,(j-1)%N]]
            for state in nn:
                if conf[state[0],state[1]] == spin_seed and state not in cluster:
                    if np.random.rand() < p_add:
                        new_spin.append(state)
                        cluster.append(state)

        old_spin = new_spin

    for i,j in cluster:
        conf[i,j] *= -1
    return conf


@jit
def oneSweep2D(conf:np.ndarray, T:float):
    '''Implementation of Metropolis Algorithm, with sequencial selection of the spins 
    (sweep), and use of the look up table (LUT), given the discrete and finite number of
    energies.
    
    Parameters
    ----------
    conf : array_like
        configuration of the system
    T : float
        temperature of the system
    
    Returns
    -------
    array_like
        the new configuration with the spins flipped
    '''

    N=conf.shape[0]

    for i in range(N):
        for j in range(N):
            heff=conf[(i-1)%N,j]+conf[(i+1)%N,j]+conf[i,(j+1)%N]+conf[i,(j-1)%N]
            sum=heff*conf[i,j]
            prob=LUT(T)
            if (sum<=0) or np.random.rand()<prob[sum]:                        
                conf[i,j]=-1.*conf[i,j]
    return conf

@jit
def LUT(T:float):
    '''Generate look-up-table for a given temperatures. In a rectancular lattice the 
    possible energies are 5, and only 2 grater than 0. We do not need the negatives one 
    for the implementation of Metropolis algorithm.
    
    Parameters
    ----------
    T : float
        temperature of the system
        
    Returns
    -------
    prob: list_like
        possible energies for flipped spin
    '''
    
    prob=np.zeros(5)
    for i in range (2,5,2):
        prob[i]=np.exp(-2*i/T)
    return prob

def simulation_binder(conf:np.ndarray, T:float, t_eq:int, t_mis:int):
    '''Simulate the system  and misure the second and forth moment of magnetization per 
    spin for computing the Binder parameter. During the first for the system thermalize,
    and during the second we made the measurements. 
    
    Parameters
    ----------
    conf : array_like
        configuration of the system
    T : float
        temperature of the simulation
    t_eq : int
        steps for thermalization of the system before the measurement
    t_mis : int
        steps for measurement
    
    Returns
    -------
    m2_mean : float
        second moment of magnetization per spin
    m4_mean : float
        forth moment of magnetization per spin
    '''

    N=conf.shape[0]

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
    '''Computation of Binder parameter for function simulation_binder.
    
    Parameters
    ----------
    m2: float
    '''
    return 0.5*(3-m4/(m2)**2)

def simulation(conf:np.ndarray, T:float, t_eq:int, t_mis:int, fstep):
    '''Simulate the system using the selected algorithm and compute the following
    quantities: energy per spin, magnetization per spin, susceptibility, specific heat. 
    During the first for the system thermalize, and during the second we made the 
    measurements.
    
    Parameters
    ----------
    conf : array_like
        configuration of the system
    T : float
        temperature of the simulation
    t_eq : int
        steps for thermalization of the system before the measurement
    t_mis : int
        steps for measurement
    fstep : function
        function that define the step of the simulation. It may be a Metropolis step 
        (onesweep2D) or a Wolff step.
    
    Returns
    -------
    e_mean : float
        energy per spin
    m_mean : float
        magnetization per spin
    s : float
        susceptibility
    c : float
        specific heat
    '''

    N=conf.shape[0]
    E = np.empty(t_mis)
    M = np.empty(t_mis)

    for i in range(t_eq):
        conf=fstep(conf,T)
    for k in range(t_mis):
        conf = fstep(conf,T)
        E[k] = energy(conf)
        M[k] = np.sum(conf)
    
    e_mean = np.mean(E)/(N**2)
    m_mean = np.mean(M)/(N**2)
    s = (1/T)*np.var(M)/(N**2)
    c = (1/T)**2*np.var(E)/(N**2)

    return e_mean, m_mean, s, c


def fconf_up(N:int):
    '''Generate configuration of all spin positive
   
    Parameters
    ----------
    N : int
        size of the system

    Returns
    -------
    array_like
        configuration of all positive spins
    '''
    return np.ones((N,N), dtype=int)

def fconf_down(N):
    '''Generate configuration of all spin negative
   
    Parameters
    ----------
    N : int
        size of the system

    Returns
    -------
    array_like
        configuration of all negativespins
    '''
    return -np.ones((N,N), dtype=int)