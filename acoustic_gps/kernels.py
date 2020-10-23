"""Summary
"""
import numpy as np
import scipy as sc


def rbf_isotropic(x1, x2, params):
    """Radial basis function iso and anisotropic

    Parameters
    ----------
    x1 : array [N_positions1, N_dimensions]
    x2 : array [N_positions2, N_dimensions]
    params: dict
        alpha : [N_samples]
            Scale factor
        rho : array [N_samples, N_dimensions]
            Length scale
    Returns
    -------
    array [N_samples, N_positions1, N_positions2]
        Covariance function
    """
    alpha = params['alpha']
    rho = params['rho']
    
    D = x1[:, None] - x2[None]
    
    if len(np.shape(rho))==1: # Isotropic
        rho = np.repeat(rho[:, None], D.shape[-1], axis = -1)

    K = np.einsum(
        'n, nij -> nij',
        alpha ** 2,
        np.exp(
            -np.einsum(
                "ijd, nd -> nij",
                D ** 2,
                1 / (2 * rho ** 2)
            )
        )
    )
    return K

def rbf_anisotropic(x1, x2, params):
    """Radial basis function iso and anisotropic

    Parameters
    ----------
    x1 : array [N_positions1, N_dimensions]
    x2 : array [N_positions2, N_dimensions]
    params: dict
        alpha : [N_samples]
            Scale factor
        rho : array [N_samples, N_dimensions]
            Length scale
    Returns
    -------
    array [N_samples, N_positions1, N_positions2]
        Covariance function
    """
    alpha = params['alpha']
    rho = params['rho']
    directions = params['directions']

    x1 = np.einsum("dj, ij -> id", directions, x1)
    x2 = np.einsum("dj, ij -> id", directions, x2)
    D = x1[:, None] - x2[None]

    K = np.einsum(
        'n, nij -> nij',
        alpha ** 2,
        np.exp(
            -np.einsum(
                "ijd, nd -> nij",
                D ** 2,
                1 / (2 * rho ** 2)
            )
        )
    )
    return K

def rbf_anisotropic_periodic(x1, x2, params):
    """Radial basis function iso and anisotropic

    Parameters
    ----------
    x1 : array [N_positions1, N_dimensions]
    x2 : array [N_positions2, N_dimensions]
    params: dict
        alpha : [N_samples]
            Scale factor
        rho : array [N_samples, N_dimensions]
            Length scale
    Returns
    -------
    array [N_samples, N_positions1, N_positions2]
        Covariance function
    """
    alpha = params['alpha']
    rho = params['rho']
    directions = params['directions']
    k = params['k']

    x1 = np.einsum("dj, ij -> id", directions, x1)
    x2 = np.einsum("dj, ij -> id", directions, x2)
    D = x1[:, None] - x2[None]

    K = np.einsum(
        'n, nij -> nij',
        alpha ** 2,
        np.exp(
            -np.einsum(
                "ijd, nd -> nij",
                np.sin(k*np.abs(D)/2) ** 2,
                1 / (2 * rho ** 2)
            )
        )
    )
    return K

def sinc(x1, x2, params):
    """Summary

    Parameters
    ----------
    x1 : array [N_positions1, N_dimensions]
    x2 : array [N_positions2, N_dimensions]
    params: dict
        alpha : [N_samples]
            Scale factor
    Returns
    -------
    TYPE
        Description
    """
    k = params['k']
    alpha = params['alpha']
    D = x1[:, None] - x2[None]
    K = np.einsum('n, ij -> nij',alpha ** 2, np.sinc(k*np.sqrt(np.sum(D**2, axis=-1))/(np.pi)))
    return K

def bessel0(x1, x2, params):
    """Summary

    Parameters
    ----------
    x1 : array [N_positions1, N_dimensions]
    x2 : array [N_positions2, N_dimensions]
    params: dict
        sigma : [N_samples]
            Scale factor
    Returns
    -------
    TYPE
        Description
    """
    k = params['k']
    sigma = params['sigma']
    D = x1[:, None] - x2[None]
    K = np.einsum('n, ij -> nij',sigma ** 2, sc.special.jv(0, k*np.sqrt(np.sum(D**2, axis=-1))))
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
    x1 : array [N_positions1, N_dimensions]
    x2 : array [N_positions2, N_dimensions]
    params: dict
        sigma_l : [N_samples]
            Scale factor
        k : wavenumber

    Returns
    -------
    TYPE
        Description
    """
    k = params['k']
    sigma_l = params['sigma_l']
    directions = params['directions']
    x1 = np.einsum("dj, ij -> id", directions, x1)
    x2 = np.einsum("dj, ij -> id", directions, x2)
    D = x1[:, None] - x2[None]
    K = np.einsum('ijd, nd -> nij', np.cos(k*D), sigma_l / 2)
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
    sigma_l : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    k = params['k']
    sigma_l = params['sigma_l']
    directions = params['directions']
    x1 = np.einsum("dj, ij -> id", directions, x1)
    x2 = np.einsum("dj, ij -> id", directions, x2)
    D = x1[:, None] - x2[None]
    K = np.einsum('ijd, nd -> nij', np.sin(k*D), sigma_l / 2)
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
    sigma_l : TYPE
        Description

    Returns
    -------
    TYPE
        Description
    """
    k = params['k']
    sigma_l = params['sigma_l']
    directions = params['directions']
    x1 = np.einsum("dj, ij -> id", directions, x1)
    x2 = np.einsum("dj, ij -> id", directions, x2)
    D = x1[:, None] - x2[None]
    K = np.einsum('ijd, nd -> nij', -np.sin(k*D), sigma_l / 2)
    return K


def zero(x1, x2, params):
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
    N_samples = params['N_samples']
    return np.zeros((N_samples, x1.shape[0], x2.shape[0]))
