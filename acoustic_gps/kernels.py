"""Summary
"""
import numpy as np


def rbf(x1, x2, params):
    """Radial basis function iso and anisotropic
    
    Parameters
    ----------
    x1 : array [N_positions1, N_dimensions]
    x2 : array [N_positions2, N_dimensions]
    alpha : real
        Scale factor
    rho : array [N_dimensions]
        Length scale
    
    Returns
    -------
    array [N_samples, N_positions1, N_positions2]
        Covariance function
    """
    alpha = params['alpha']
    rho = params['rho']
    if not np.shape(rho):
        rho = np.array([rho])
    D = x1[:, None] - x2[None]
    K = alpha ** 2 * np.exp(-np.einsum("ijd, d -> ij", D ** 2, 1 / (2 * rho ** 2)))
    return K

def sinc(x1, x2, params):
    """Summary
    
    Parameters
    ----------
    x1 : TYPE
        Description
    x2 : TYPE
        Description
    a : TYPE
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    k = params['k']
    alpha = params['alpha']
    D = x1[:, None] - x2[None]
    K = alpha ** 2 * np.sinc(k*np.sqrt(np.sum(D**2, axis = -1))/(np.pi))
    return K


# def plane_wave(x1, x2, k, alpha):
#     """Summary
    
#     Parameters
#     ----------
#     x1 : TYPE
#         Description
#     x2 : TYPE
#         Description
#     k : TYPE
#         Description
#     alpha : TYPE
#         Description
    
#     Returns
#     -------
#     TYPE
#         Description
#     """
#     x1_proj = np.einsum("ij, kj -> ik", k, x1)
#     x2_proj = np.einsum("ij, kj -> ik", k, x2)
#     D = x1_proj[:, :, None] - x2_proj[:, None]
#     K = np.sum(alpha[:, None, None] * np.exp(-1j * D), axis=0)
#     return K


def cosine(x1, x2, params):
    """Summary
    
    Parameters
    ----------
    x1 : TYPE
        Description
    x2 : TYPE
        Description
    k : TYPE
        Description
    alpha : TYPE
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    k = params['k']
    alpha = params['alpha']
    D = x1[:, None] - x2[None]
    K = np.einsum('ijd, d -> ij', np.cos(k*D), alpha / 2)
    return K


def sine(x1, x2, params):
    """Summary
    
    Parameters
    ----------
    x1 : TYPE
        Description
    x2 : TYPE
        Description
    k : TYPE
        Description
    alpha : TYPE
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    k = params['k']
    alpha = params['alpha']
    D = x1[:, None] - x2[None]
    K = np.einsum('ijd, d -> ij',np.sin(k*D), alpha / 2)
    return K

def sine_neg(x1, x2, params):
    """Summary
    
    Parameters
    ----------
    x1 : TYPE
        Description
    x2 : TYPE
        Description
    k : TYPE
        Description
    alpha : TYPE
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    k = params['k']
    alpha = params['alpha']
    D = x1[:, None] - x2[None]
    K = np.einsum('ijd, d -> ij',-np.sin(k*D), alpha / 2)
    return K

def zero(x1, x2, **kwargs):
    """Summary
    
    Parameters
    ----------
    x1 : TYPE
        Description
    x2 : TYPE
        Description
    **kwargs
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    return np.zeros((x1.shape[0], x2.shape[0]))
