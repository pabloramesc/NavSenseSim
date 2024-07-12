import numpy as np



def eul2quat(angles: np.ndarray = None):
    """
    Converts euler angles to quaternions.

    Arguments
    ----------
    angles : numpy.ndarray, default: None
        N-by-3 array with euler angles as columns (roll, pitch, yaw) in rads.

    Returns
    -------
    q : numpy.ndarray
        N-by-4 array with attitude in quaternion form.
    """
    if angles.shape[-1] != 3:
            raise ValueError("Input data must be of shape (N, 3) or (3,). Got shape {}".format(angles.shape))
    # get the num of samples
    N = len(angles)
    # compute sin and cos of each angle for later use
    r = angles[:,0]; p = angles[:,1]; y = angles[:,2]
    sr = np.sin(0.5*r); cr = np.cos(0.5*r)
    sp = np.sin(0.5*p); cp = np.cos(0.5*p)
    sy = np.sin(0.5*y); cy = np.cos(0.5*y)
    # compute quaternions
    q = np.zeros((N, 4))
    q[:,0] = cy*cp*cr + sy*sp*sr
    q[:,1] = cy*cp*sr - sy*sp*cr
    q[:,2] = sy*cp*sr + cy*sp*cr
    q[:,3] = sy*cp*cr - cy*sp*sr
    return q/np.linalg.norm(q, axis=1)[:,None]



def quat2eul(q: np.ndarray = None):
    """
    Converts quaternions to euler angles.

    Arguments
    ----------
    q : numpy.ndarray, default: None
        N-by-4 array with attitude in quaternion form.
    
    Returns
    -------
    angles : numpy.ndarray
        N-by-3 array with euler angles as columns (roll, pitch, yaw) in rads.
    """
    if q.shape[-1] != 4:
            raise ValueError("Input data must be of shape (N, 4) or (4,). Got shape {}".format(q.shape))
    # get the num of samples
    N = len(q)
    # compute euler angles
    q0 = q[:,0]; q1 = q[:,1]; q2 = q[:,2]; q3 = q[:,3]
    angles = np.zeros((N,3))
    angles[:,0] = np.arctan2(2.0*(q0*q1+q2*q3), 1.0-2.0*(q1*q1+q2*q2))
    angles[:,1] = np.arcsin(2.0*(q0*q2-q3*q1))
    angles[:,2] = np.arctan2(2.0*(q0*q3+q1*q2), 1.0-2.0*(q2*q2+q3*q3))
    return angles



def eul2dcm(angles: np.ndarray = None):
    """
    Convert a sequence of N euler angles to a sequence of N DCM stack.

    Arguments
    ----------
    angles : numpy.ndarray, default : None
        N-by-3 array with euler angles as columns (roll, pitch, yaw) in rads.

    Returns
    -------
    R : numpy.ndarray
        N-by-3-by-3 Direction Cosine Matrix stack.
    """
    if angles.shape[-1]!= 3:
        raise ValueError("Input data must be of shape (N, 3) or (3,). Got shape {}".format(angles.shape))
    # get the num of samples
    N = len(angles)
    # compute sin and cos of each angle for later use
    r = angles[:,0]; p = angles[:,1]; y = angles[:,2]
    sr = np.sin(0.5*r); cr = np.cos(0.5*r)
    sp = np.sin(0.5*p); cp = np.cos(0.5*p)
    sy = np.sin(0.5*y); cy = np.cos(0.5*y)
    # compute the Direction Cosine Matrix (DCM)
    R = np.zeros((N,3,3))
    R[:,0,0] = cp*cy
    R[:,1,0] = sr*sp*cy - cp*sy
    R[:,2,0] = cr*sp*cy + sr*sy
    R[:,0,1] = cp*sy
    R[:,1,1] = sr*sp*sy + cr*cy
    R[:,2,1] = cr*sp*sy - sr*cy
    R[:,0,2] = -sp
    R[:,1,2] = sr*cp
    R[:,2,2] = cr*cp
    return R



def quat2dcm(q: np.ndarray = None):
    """
    Convert a sequence of N quaternions to a sequence of N DCM stack.

    Arguments
    ----------
    q : numpy.ndarray, default : None
        N-by-4 array with attitude in quaternion form.

    Returns
    -------
    R : numpy.ndarray
        N-by-3-by-3 Direction Cosine Matrix stack.
    """
    if q.shape[-1]!= 4:
        raise ValueError("Input data must be of shape (N, 4) or (4,). Got shape {}".format(q.shape))
    # normalize all quaternions
    q /= np.linalg.norm(q, axis=1)[:,None]
    # get the num of samples
    N = len(q)
    # compute the Direction Cosine Matrix (DCM)
    q0 = q[:,0]; q1 = q[:,1]; q2 = q[:,2]; q3 = q[:,3]
    R = np.zeros((N,3,3))
    R[:,0,0] = 1.0 - 2.0*(q2**2 + q3**2)
    R[:,1,0] = 2.0*(q1*q2 + q0*q3)
    R[:,2,0] = 2.0*(q1*q3 - q0*q2)
    R[:,0,1] = 2.0*(q1*q2 - q0*q3)
    R[:,1,1] = 1.0 - 2.0*(q1**2 + q3**2)
    R[:,2,1] = 2.0*(q0*q1 + q2*q3)
    R[:,0,2] = 2.0*(q1*q3 + q0*q2)
    R[:,1,2] = 2.0*(q2*q3 - q0*q1)
    R[:,2,2] = 1.0 - 2.0*(q1**2 + q2**2)
    return R


def quatmul(p: np.ndarray = None, q: np.ndarray = None):
    """
    Compute p and q quaternions multiplication.

    Arguments
    ----------
    p : numpy.ndarray, default : None
        N-by-4 array quaternion.
    q : numpy.ndarray, default : None
        N-by-4 array quaternion.

    Returns
    -------
    pq : numpy.ndarray
        N-by-4 array quaternion multiplication result.
    """
    if p.shape[-1]!= 4:
        raise ValueError("Input data must be of shape (N, 4) or (4,). Got shape {}".format(q.shape))
    if q.shape[-1]!= 4:
        raise ValueError("Input data must be of shape (N, 4) or (4,). Got shape {}".format(q.shape))
    if p.shape != q.shape:
        raise ValueError("Input data quaternions array p and q must be on same shape.")
    # get the num of samples
    N = len(p)
    # compute the quaternion multiplication
    p0 = p[:,0]; p1 = p[:,1]; p2 = p[:,2]; p3 = p[:,3]
    q0 = q[:,0]; q1 = q[:,1]; q2 = q[:,2]; q3 = q[:,3]
    pq = np.zeros((N,4))
    pq[:,0] = p0*q0 - p1*q1 - p2*q2 - p3*q3
    pq[:,1] = p0*q1 + p1*q0 + p2*q3 - p3*q2
    pq[:,2] = p0*q2 - p1*q3 + p2*q0 + p3*q1
    pq[:,3] = p0*q3 + p1*q2 - p2*q1 + p3*q0
    return pq

def rotvec(u: np.ndarray = None, R: np.ndarray = np.diag([1,1,1])):
    """
    Compute the rotation of a N-by-3 array given a list of N rotation
    matrix (DCM).

    Arguments
    ----------
    u : numpy.ndarray, default : None
        N-by-3 array of N vectors.
    R : numpy.ndarray, default : I (ident matrix)
        N-by-3-by-3 Direction Cosine Matrix stack.

    Returns
    -------
    v : numpy.ndarray
        N-by-3 rotated array of N vectors.
    """
    if u.shape[-1]!= 3:
        raise ValueError("Input vector must be of shape (N, 3) or (3,). Got shape {}".format(u.shape))
    if R.shape[-2:]!= (3,3):
        raise ValueError("Input vector must be of shape (N, 3, 3) or (3, 3). Got shape {}".format(R.shape))
    if len(R) != len(u):
        raise ValueError("Input vector and rotation matrix list must have same number of samples.")
    # get the num of samples
    N = len(u)
    # compute the quaternion multiplication
    v = np.zeros((N,3))
    u0 = u[:,0]; u1 = u[:,1]; u2 = u[:,2]
    v[:,0] = R[:,0,0]*u0 + R[:,0,1]*u1 + R[:,0,2]*u2
    v[:,1] = R[:,1,0]*u0 + R[:,1,1]*u1 + R[:,1,2]*u2
    v[:,2] = R[:,2,0]*u0 + R[:,2,1]*u1 + R[:,2,2]*u2
    return v