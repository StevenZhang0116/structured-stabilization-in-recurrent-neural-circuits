import numpy as np 
from brian2 import *

########################################
# ring structure functions
########################################
def vonmisesnotnorm(theta, mu, k):
    """
    Calculate the value of the un-normalized von Mises distribution at a given angle.
    Always peaks at 1.0

    Parameters:
    theta (float): The angle at which to evaluate the distribution.
    mu (float): The mean angle of the distribution.
    k (float): The concentration parameter of the distribution.

    Returns:
    float: The value of the normalized von Mises distribution at the given angle.
    """
    assert 0.0 <= theta <= 2 * np.pi, "θ must be in the range [0, 2π]"
    return np.exp(k * np.cos(theta - mu)) / np.exp(k)

def vonmisesnorm(theta, mu, k):
    """
    Calculate the value of the von Mises distribution at a given angle.

    Parameters:
    theta (float): The angle at which to evaluate the distribution.
    mu (float): The mean angle of the distribution.
    k (float): The concentration parameter of the distribution.

    Returns:
    float: The value of the normalized von Mises distribution at the given angle.
    """
    assert 0.0 <= theta <= 2 * np.pi, "θ must be in the range [0, 2π]"
    return np.exp(k * np.cos(theta - mu)) / (2 * np.pi * i0(k))

def assign_angles(n):
    """
    Generates equally spaced angles.

    Parameters:
        n (int): The number of angles to generate.

    Returns:
        numpy.ndarray: An array of equally spaced angles.
    """
    angles = np.linspace(0, 2 * np.pi, n + 1)[:-1]  
    return angles


def make_ring_pre_post_fixed_sum(npre, npost, wsum, k_vm, w_thresh=0.0,avoid_self_connections=False):    
    """
    Creates connectivity between pre and post layers in a ring model with a fixed sum of weights along the rows

    Parameters:
        npre (int): Number of pre neurons
        npost (int): Number of post neurons
        wsum (float): Sum of weights.
        k_vm (float): Concentration parameter for the von Mises distribution.
        w_thresh (float, optional): Threshold value for weights. Defaults to 0.0.
        avoid_self_connections (bool, optional): Avoid self connections. Defaults to False.

    Returns:
        numpy.ndarray: Connectivity matrix between pre and post layers.

    """
    muspre = assign_angles(npre)
    muspost = assign_angles(npost)
    w = np.zeros((npost, npre))
    for i in range(npost):
        for j in range(npre):
            if avoid_self_connections and i == j:
                continue
            else:
                w[i, j] = vonmisesnorm(np.abs(muspre[j] - muspost[i]), 0.0, k_vm)
               
    w[w < w_thresh] = 0.0
   
    expected_sum = 1 / (np.pi*2/npost)
    print(f"expected_sum: {expected_sum}")
    if avoid_self_connections:
        expected_sum -= vonmisesnorm(0, 0, k_vm) # Subtract self-connection if present
    w *= wsum/expected_sum
    return w
  
def wmat_to_pre_post_list(wmat, w_threshold=1E-6):
    """Converts weight matrix to list representation"""
    post_all, pre_all = np.where(wmat > w_threshold)  # Efficiently find non-zero indices
    w_all = wmat[post_all,pre_all]
    return pre_all.tolist(), post_all.tolist(), w_all.tolist() 

def make_ring_pre_post_fixedsum_list(npre, npost, wsum, k_vm, 
                                     w_thresh=0.0, avoid_self_connections=False
    ):
    """
    Generates a weight matrix for a ring model with fixed sum of weights and converts it into a pre-post list.

    Args:
        npre (int): Number of presynaptic neurons.
        npost (int): Number of postsynaptic neurons.
        wsum (float): Sum of weights.
        k_vm (float): Scaling factor for Von Mises
        w_thresh (float, optional): Threshold weight value. Defaults to 0.0.

    Returns:
        list: Pre-post list representation of the weight matrix.
             format is (all_pre_indexes,all_post_indexes,all_weights)      

    """
    wmat = make_ring_pre_post_fixed_sum(npre, npost, wsum, k_vm, w_thresh, avoid_self_connections)
    return wmat_to_pre_post_list(wmat, w_thresh)

########################################
# torus structure functions
########################################

def assign_positions_2d(n, Lx=1.0, Ly=1.0):
    """
    """
    nx = int(np.round(np.sqrt(n * Lx / Ly)))
    nx = max(nx, 1)
    ny = int(np.ceil(n / nx))
    dx = Lx / nx
    dy = Ly / ny
    # xs = np.linspace(0, Lx, nx, endpoint=False)
    # ys = np.linspace(0, Ly, ny, endpoint=False)
    xs = (np.arange(nx) + 0.5) * dx
    ys = (np.arange(ny) + 0.5) * dy
    GX, GY = np.meshgrid(xs, ys, indexing='xy')
    P = np.column_stack([GX.ravel(), GY.ravel()])[:n]

    return P

def expected_sum_torus_gaussian(npre, npost, sx, sy, Lx=1.0, Ly=1.0, avoid_self=False):
    """
    """
    exp_sum = (npost * (2.0 * np.pi * sx * sy)) / (Lx * Ly)
    if avoid_self and (npre == npost):
        exp_sum -= 1.0 
    return exp_sum

def periodic_diff(a, b, L):
    """
    """
    d = a[:, None] - b[None, :]
    d = (d + L/2) % L - L/2
    return d

def torus_distance_matrix(pre_pos, post_pos, Lx=1.0, Ly=1.0):
    """
    """
    dx = periodic_diff(post_pos[:, 0], pre_pos[:, 0], Lx)
    dy = periodic_diff(post_pos[:, 1], pre_pos[:, 1], Ly)
    R = np.sqrt(dx*dx + dy*dy)
    return dx, dy, R

def torus_kernel_gaussian(dx, dy, sx, sy):
    """
    """
    return np.exp(-0.5*((dx/sx)**2 + (dy/sy)**2))

def make_torus_pre_post_fixed_sum(pre_pos, post_pos, wsum, sx, sy,
                                  Lx=1.0, Ly=1.0, r_max=None, w_thresh=0.0,
                                  avoid_self_connections=False):
    """
    """
    dx, dy, R = torus_distance_matrix(pre_pos, post_pos, Lx, Ly)
    W = torus_kernel_gaussian(dx, dy, sx, sy)

    if r_max is not None:
        W[R > r_max] = 0.0
    if avoid_self_connections and (pre_pos.shape[0] == post_pos.shape[0]):
        np.fill_diagonal(W, 0.0)
    if w_thresh > 0:
        W[W < w_thresh] = 0.0
    
    npre, npost = pre_pos.shape[0], post_pos.shape[0]
    exp_sum = expected_sum_torus_gaussian(npre, npost, sx, sy, Lx, Ly, \
        avoid_self=avoid_self_connections and (npre == npost)
    )
    if exp_sum > 0:
        W *= (wsum / exp_sum)
    
    return W

def wmat_to_lists(W, w_threshold=1e-9):
    """
    """
    post, pre = np.where(W > w_threshold)
    w = W[post, pre]
    return pre.tolist(), post.tolist(), w.tolist()
