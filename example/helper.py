import numpy as np
from pyDOE import lhs
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import acoustic_gps as agp


def varname_to_latexmath(varname):
    latex_varnames = dict(
        alpha='\\alpha',
        beta='\\beta',
        gamma='\\gamma',
        tau='\\tau',
        rho='\\rho',
        sigma='\\sigma'
    )
    if varname in latex_varnames:
        latexname = latex_varnames[varname]
    else:
        latexname = varname
    return '{' + latexname + '}'


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
    prior_params = {}
    if kernel == 'rbf_isotropic':
        # RBF Isotropic
        prior_params = dict(
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

        prior_params = dict(
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
        prior_params = dict(
            D=posible_directions.shape[0],
            a_tau_alpha=1,
            b_tau_alpha=1e-2,
            a_tau_rho=1,
            b_tau_rho=40 * np.ones(posible_directions.shape[0]),
            directions=posible_directions,
        )

    if kernel == 'plane_wave_hierarchical':
        stan_params = ["sigma_l", "b_log"]
        kernel_names = ["cosine", "cosine", "sine", "sine_neg"]
        kernel_params = ["sigma_l"]
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
            b_log_mean=2,  # prior mean of the b_log hyperparameter
            b_log_std=1  # prior std of the b_log hyperparameter
        )

    if kernel == 'bessel_isotropic':
        stan_params = ["sigma", "tau_sigma"]
        kernel_names = ["bessel0", "bessel0", "zero", "zero"]
        kernel_params = ["sigma"]
        # Sinc
        prior_params = dict(
            D=x.shape[1],
            a_tau_sigma=1,
            b_tau_sigma=1e-2,
        )
    return kernel_names, kernel_params, prior_params, stan_params


def plane_wave_field(
    xs,
    n_waves=1,
    snr=20,
    n_reps=1,
    f=np.array([300]),
    c=343
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
    setup['wave_direction'] = wave_direction
    setup['wave_amplitude'] = wave_amplitude

    norm = np.mean(np.abs(p_clean), axis=-1)
    p_clean /= norm[:, None]
    p_mean = np.mean(np.abs(p_clean), axis=-1)
    noise_std = p_mean / (10**(snr/20))
    noise = np.einsum('f, rx -> rfx', noise_std, (
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
            i_mics[:, 0] = np.floor(
                i_mics[:, 0] * grid[0]).astype(int) + int(x_first)
            i_mics[:, 1] = np.floor(
                i_mics[:, 1] * grid[1]).astype(int) + int(y_first)
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
    y_last = n_cols  # Last column
    i_mics = get_sparse_microphones(
        n_mics_zone, n_rows, x_first, x_last, y_first, y_last
    )
    return i_mics.astype(int)


def plot_inference_summaries(data, posterior_samples, posterior_summary):
    rhat_color = 'C3'
    ax = [[] for i in range(len(posterior_samples.keys()))]
    fig = plt.figure(figsize=(5, 7))
    nrows = len(posterior_samples.keys())
    ncols = 1
    spec = gridspec.GridSpec(nrows=nrows, ncols=ncols, figure=fig)
    for i, key in enumerate(posterior_samples.keys()):
        ax[i] = fig.add_subplot(spec[i])
        row = [
            index
            for index, s in enumerate(list(posterior_summary["summary_rownames"]))
            if s.startswith(key)
        ]
        labels = [
            label
            for label in list(posterior_summary["summary_rownames"])
            if label.startswith(key)
        ]
        col = list(posterior_summary["summary_colnames"]).index("Rhat")
        bplot = ax[i].boxplot(posterior_samples[key],
                              showfliers=False, zorder=1, patch_artist=True)
        for patch in bplot["boxes"]:
            patch.set_facecolor("gray")
        ax2 = ax[i].twinx()
        ax2.plot(
            ax[i].get_xticks(),
            np.round(posterior_summary["summary"][row, col], decimals=3),
            marker="o",
            zorder=0,
            color=rhat_color,
            alpha=0.5,
        )
        ax2.tick_params(axis="y", colors=rhat_color)
        ax2.set_ylim(0.6, 1.4)
        ax2.set_ylabel(r"Convergence, $\hat{R}$", color=rhat_color)
        if len(row) > 1:
            direction_key = [j for j in list(
                data.keys()) if "direction" in j][0]
            angles = np.arctan2(
                data[direction_key][:, -1], data[direction_key][:, 0]
            )
            # ax[i].set_xticklabels(np.round(angles, decimals=2), rotation=45)
            # ax[i].set_xlabel("rad")
        else:
            ax[i].set_xticklabels([])
        substrings = key.split(sep="_")
        sublabels = [varname_to_latexmath(s) for s in substrings]
        ylabel = "_".join(sublabels)
        ax[i].set_ylabel(r"$" + ylabel + "$")
        ax[i].grid(which='both')
    plt.tight_layout()

    plt.show()


def plot_reconstruction(xs, x, p_true, p_predict, uncertainty):
    ax_true = plt.subplot(131)
    ax_predict = plt.subplot(132)
    ax_uncertainty = plt.subplot(133)
    agp.utils.show_soundfield(ax_true, xs.T, p_true, what=None, cmap='RdBu')
    agp.utils.show_soundfield(
        ax_predict, xs.T, p_predict, what=None, cmap='RdBu')
    agp.utils.show_soundfield(ax_uncertainty, xs.T,
                              uncertainty, what=None, cmap='Greys')

    ax_true.set_title('true sound field')
    ax_predict.set_title('mean prediction')
    ax_uncertainty.set_title('uncertainty')
    ax_uncertainty.plot(x[:, 0], x[:, 1], marker='s',
                        linestyle='', color='yellowgreen', markeredgecolor='k')
    ax_true.plot(x[:, 0], x[:, 1], marker='s', linestyle='',
                 color='yellowgreen', markeredgecolor='k')
    plt.tight_layout()
