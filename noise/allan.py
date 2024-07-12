import numpy as np

from common.constants import PI


def compute_allan_variance(samples, sample_rate, max_clusters_num=100):
    """
    Compute de allan variance

    Arguments
    ---------
    samples : numpy.ndarray
        N array of the sensor measured data
    sample_rate : float
        Number of samples taken by second
    max_cluster_num : int
        Max number of clusters to use in the analysis
        More clusters requires more computation time
        More than 100 clusters doesnt improve precision    
    """
    t0 = 1/sample_rate
    theta = np.cumsum(samples)*t0
    Ns = samples.size
    m = np.floor((Ns-1)/2)
    clusters = np.unique(np.geomspace(1, m, max_clusters_num, dtype=int))
    tau = clusters*t0

    avar = np.zeros(clusters.size)
    for k in range(clusters.size):
        mk = clusters[k]
        avar[k] = np.sum((theta[2*mk:Ns] - 2*theta[mk:Ns-mk] + theta[0:Ns-2*mk])**2)
    avar = avar / (2*(tau**2)*(Ns - 2*clusters))
    return avar, tau


def get_allan_N(tau, adev):
    slope = -1/2
    logtau = np.log10(tau)
    logadev = np.log10(adev)
    dlogadev = np.diff(logadev)/np.diff(logtau)
    index = np.searchsorted(dlogadev, slope)
    b = logadev[index] - slope*logtau[index]
    logN = b + slope*np.log10(1)
    N = 10**logN
    return N


def get_allan_B(tau, adev):
    slope = 0
    logtau = np.log10(tau)
    logadev = np.log10(adev)
    dlogadev = np.diff(logadev)/np.diff(logtau)
    index = np.searchsorted(dlogadev, slope)
    b = logadev[index] - slope*logtau[index]
    scfB = np.sqrt(2*np.log(2)/PI)
    logB = b - np.log10(scfB)
    B = 10**logB
    tauB = tau[index]
    return B, tauB


def get_allan_K(tau, adev):
    slope = 1/2
    logtau = np.log10(tau)
    logadev = np.log10(adev)
    dlogadev = np.diff(logadev)/np.diff(logtau)
    index = np.searchsorted(dlogadev, slope)
    b = logadev[index] - slope*logtau[index]
    logK = b + slope*np.log10(3)
    K = 10**logK
    return K