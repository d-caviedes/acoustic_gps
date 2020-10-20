import numpy as np
from pyDOE import lhs
import random

def init_model(x, kernel, n_basis_functions):
    """Load necessary parameters for both Bayesian inference and predictions

    Parameters
    ----------
    x : ndarray [N_locations, spatial dimensions]
        Measured locations on the sound field
    kernel : str
        kernel name
    n_basis_functions : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """     
    stan_params = {}
    kernel_names = {}
    kernel_params = {}
    prior_params  = {}
    if kernel == 'rbf_isotropic':
        # RBF Isotropic
        prior_params  = dict(
            D=x.shape[1],
            a_tau_alpha=1,
            b_tau_alpha=1e-2,
            a_tau_rho=1,
            b_tau_rho=1e-2
        )
        stan_params = ["alpha", "rho", "tau_alpha", "tau_rho"]
        kernel_names = [
            "rbf_isotropic", "rbf_isotropic", "zero", "zero"]
        kernel_params = ["alpha", "rho"]

    if kernel == 'rbf_anisotropic':

        stan_params = ["alpha", "rho", "tau_alpha", "tau_rho"]
        kernel_names = [
            "rbf_anisotropic", "rbf_anisotropic", "zero", "zero"]
        kernel_params = ["alpha", "rho"]
        # RBF Anisotropic
        angle_range = np.pi
        delta_angle = angle_range / n_basis_functions
        angles = np.arange(0, angle_range, angle_range / n_basis_functions)
        posible_directions = np.concatenate(
            (np.cos(angles[:, None]), np.sin(angles[:, None])), axis=-1
        )

        prior_params  = dict(
            D=posible_directions.shape[0],
            a_tau_alpha=1,
            b_tau_alpha=1e-2,
            a_tau_rho=1,
            b_tau_rho=40 * np.ones(posible_directions.shape[0]),
            directions=posible_directions,
        )

    if kernel == 'rbf_anisotropic_periodic':

        stan_params = ["alpha", "rho", "tau_alpha", "tau_rho"]
        kernel_names = [
            "rbf_anisotropic_periodic", "rbf_anisotropic_periodic", "zero", "zero"]
        kernel_params = ["alpha", "rho"]
        angle_range = np.pi
        delta_angle = angle_range / n_basis_functions
        angles = np.arange(0, angle_range, angle_range / n_basis_functions)
        posible_directions = np.concatenate(
            (np.cos(angles[:, None]), np.sin(angles[:, None])), axis=-1
        )
        prior_params  = dict(
            D=posible_directions.shape[0],
            a_tau_alpha=1,
            b_tau_alpha=1e-2,
            a_tau_rho=1,
            b_tau_rho=40 * np.ones(posible_directions.shape[0]),
            directions=posible_directions,
        )

    if kernel == 'plane_wave_hierarchical':
        stan_params= ["alpha", "b", "b_log"]
        kernel_names= ["cosine", "cosine", "sine", "sine_neg"]
        kernel_params= ["alpha"]
        # PLANE WAVE
        # Whole circle
        angle_range = 2 * np.pi
        delta_angle = angle_range / n_basis_functions
        angles = np.arange(0, angle_range, angle_range / n_basis_functions)
        posible_directions = np.concatenate(
            (np.cos(angles[:, None]), np.sin(angles[:, None])), axis=-1
        )
        prior_params = dict(
            D=posible_directions.shape[0],
            wave_directions=posible_directions,
            a=1,
            b_log_mean = 2, # prior mean of the b hyperparameter
            b_log_std= 1 # prior std of the b hyperparameter
        )

    if kernel == 'bessel_isotropic':
        stan_params = ["alpha", "tau_alpha"]
        kernel_names = ["bessel0", "bessel0", "zero", "zero"]
        kernel_params = ["alpha"]
        # Sinc
        prior_params  = dict(
            D=x.shape[1],
            a_tau_alpha=1,
            b_tau_alpha=1e-2,
        )
    return kernel_names, kernel_params, prior_params, stan_params

def plane_wave_field(
                xs,
                n_waves = 1,
                snr=20, 
                n_reps=1,
                f = np.array([300]),
                c = 343
                ):
    k = 2 * np.pi * f / c
    snr = 30
    setup = {}
    wave_direction = np.pi * np.random.uniform(
        low=-1, high=1, size=n_waves
    )  # random directions in radians
    wave_direction = np.concatenate(
        (np.cos(wave_direction)[None], np.sin(
            wave_direction)[None]), axis=0
    ).T  # wave directions in cartesian coordinates
    wave_amplitude = np.random.randn(
        n_waves) + 1j * np.random.randn(n_waves)
    k_vec = np.einsum("i, jk -> ijk", k, wave_direction)
    p_clean = (
        wave_amplitude[None, None] *
        np.exp(-1j * (np.einsum("ijk, lk -> ilj", k_vec, xs)))
    ).sum(axis=-1)
    # print(p_clean.shape)
    setup['wave_direction'] = wave_direction
    setup['wave_amplitude'] = wave_amplitude

    norm = np.mean(np.abs(p_clean), axis = -1)
    p_clean /= norm[:, None]
    p_mean = np.mean(np.abs(p_clean), axis = -1)
    noise_std = p_mean / (10**(snr/20))
    noise = np.einsum('f, rx -> rfx',noise_std, (
            np.random.randn(n_reps, xs.shape[0]) +
            1j * np.random.randn(n_reps, xs.shape[0])
        ))
    p = p_clean[None] + noise
    setup['noise'] = noise
    setup['noise_std'] = noise_std        
    setup['norm'] = norm
    setup['xs'] = xs
    setup['f'] = f
    setup['c'] = c

    return p_clean, p, setup

def get_sparse_microphones(n_mics, n_rows, x_first=0, x_last=1, y_first=0, y_last=1):
    """Use Latin Hyper Cube to randomize the chosen microphone positions in a rectangular space.
    
    Parameters
    ----------
    n_mics : int
        Number of microphones to choose.
    n_rows : int
        total number of microphone rows
    x_first : int
        First row to include.
    x_last : int
        Last row to include
    y_first : int
        First column to include
    y_last : int
        Last column to include
    
    Returns
    -------
    i_mics : array
        Indexes of the mics.
    """
    i_mics = np.random.rand(1)
    i = 1
    grid = np.asarray([x_last - x_first - 1, y_last - y_first - 1])
    if n_mics > 0:
        while len(np.unique(i_mics)) < n_mics:
            i_mics = lhs(2, samples=n_mics * i, criterion="center")
            i_mics[:, 0] = np.floor(i_mics[:, 0] * grid[0]).astype(int) + int(x_first)
            i_mics[:, 1] = np.floor(i_mics[:, 1] * grid[1]).astype(int) + int(y_first)
            # Convert coordinates to indices
            i_mics = (i_mics[:, 1] * n_rows + i_mics[:, 0]).astype(int)
            i += 1
        i_mics = np.asarray(random.sample(i_mics.tolist(), n_mics))
    else:
        i_mics = []
    return i_mics

def random_locations(n_mics_zone, n_rows, n_cols):
    x_first = 0  # First row to take into account
    x_last = n_rows  # Last row to take into account
    y_first = 0  # First column
    y_last = n_cols # Last column
    i_mics = get_sparse_microphones(
        n_mics_zone, n_rows, x_first, x_last, y_first, y_last
    )
    return i_mics.astype(int)
