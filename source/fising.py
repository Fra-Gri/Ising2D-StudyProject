import numpy as np
from numba import jit

def conf_rnd(N:int):
    '''Create an initial square configuration of the dimension given, generating every element randomly between +1 or -1,
    with the same probability.
    
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
    '''Compute the energy per spin of a configuration: for every spin compute the interaction with the nearest neighbour
    heff, and add it to the energy E. The divide 2 is for not counting double the interactions.
    
    
    Parameters
    ---------
    conf : array_like
        configuration of the sistem
    
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

@jit
def wolff2(conf:np.ndarray, T:float):
    '''Implementation of one step the Wolff algorithm:
            1 - select the seed
            2 - compute the cluster
            3 - flip the cluster
    Parameters
    ----------
    conf : array_like
        configuration of the sistem
    T : float
        temperature of the sistem
    
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
            nn = [[(i+1)%N,j], [(i-1)%N,j], [i,(j+1)%N], [i,(j-1)%N]]

            for state in nn:
                if conf[state[0],state[1]] == spin_seed and state not in cluster:
                    if np.random.rand() < p_add:
                        new_spin.append(state)
                        cluster.append(state)

        old_spin = new_spin

    for i,j in cluster:
        conf[i,j] *= -1
    return conf