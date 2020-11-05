#!/usr/bin/env python3
'''Example of sound field reconstruction using Gaussian processes.

Usage:
    ./example.py

Author:
    Diego Caviedes Nozal - 23.10.2020
'''
import argparse
import acoustic_gps as agp
import numpy as np
from matplotlib import pyplot as plt
from helper import *
from matplotlib import rc
from distutils.spawn import find_executable
import os

rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
if find_executable('latex'): 
    print("latex installed")
    rc('text', usetex=True)

# TODO: Explain how Bivariate to Complex works in the code.
# TODO: Comment code properly
# TODO: Header and paper citing at the top (any CC license)?
# TODO: Reconstruction plot -> colorbars. axis labels
# TODO: Create folder if it doesn't exist

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
# parser.add_argument('foo', help="Name of file to process")
parser.parse_args()


def example():
    montecarlo_chains = 4
    samples_per_chain = 400
    warmup_samples_per_chain = 200
    # number of basis kernels (for anisotropic kernels)
    L = 64
    # number of observations
    N = 10

    # Definition of sound field to reconstruct
    number_of_waves = 1
    # number of measurement repetitions (maybe several sine sweeps?)
    number_meas_reps = 1
    field_grid_rows = 26
    dx = np.linspace(0, 2.5, field_grid_rows, endpoint=True)
    X, Y = np.meshgrid(dx, dx)

    all_locations = np.concatenate(
        (
            X.flatten()[:, None],
            Y.flatten()[:, None]
        ),
        axis=-1
    )  # Total grid of locations to reconstruct

    true_sound_field, noisy_sound_field, sound_field_parameters = plane_wave_field(
        all_locations,
        n_reps=number_meas_reps,
        n_waves=number_of_waves  # Number of plane waves that conform the field
    )

    # Observations: choose random locations from all_locations with Latin Hypercube Sampling
    i_measured = random_locations(N, field_grid_rows, field_grid_rows)
    measured_locations = all_locations[i_measured]
    measured_sound_field = noisy_sound_field[:, 0, i_measured]

    # choose "kernel" (uncomment one)
    kernel = [
        # 'plane_wave_hierarchical',
        # 'bessel_isotropic',
        # 'rbf_isotropic',
        # 'rbf_anisotropic',
        'rbf_anisotropic_periodic',
    ][0]

    # Stan models need to be compiled before executed (it runs on C).
    # Once the model is compiled, there is no need to compile it again
    # to make inferences. If changes are made in the .stan code,
    # compilation is needed for them to take effect.
    compile = False

    if compile is True:
        agp.utils.compile_model(
            model_name=kernel
        )

    # Data used by Stan for Monte Carlo sampling. This is common for all models.
    data = dict(
        x=measured_locations,
        N_meas=measured_sound_field.shape[-1],
        N_reps=number_meas_reps,
        y=np.concatenate(
                        (measured_sound_field.real,
                         measured_sound_field.imag),
            axis=-1),
        k=(2 * np.pi * sound_field_parameters['f']
           [0] / sound_field_parameters['c']),
        delta=1e-10,
        Sigma=(sound_field_parameters['noise_std'][0]**2) /
        2 * np.eye(2*measured_sound_field.shape[-1])
    )

    # Hyperparameters from prior distributions are loaded from helper.py
    bivariate_kernels, kernel_param_names, prior_params, stan_pars = init_model(
        measured_locations,
        kernel=kernel,
        n_basis_functions=L
    )
    data.update(prior_params)

    # Monter Carlo sampling
    posterior_samples, posterior_summary = agp.mc_sampling(
        kernel=kernel,
        data=data,
        pars=stan_pars,
        n_samples=samples_per_chain,
        chains=montecarlo_chains,
        warmup_samples=warmup_samples_per_chain
    )

    # Reconstruction: For this example, the median of the inferences is used
    kernel_params = {}
    kernel_params['N_samples'] = 1
    for i in kernel_param_names:
        kernel_params[i] = np.median(posterior_samples[i], axis=0)[None]
    kernel_params["k"] = data["k"]
    if "plane_wave" in kernel:
        kernel_params["directions"] = data["wave_directions"]
    if "rbf_anisotropic" in kernel:
        kernel_params["directions"] = data["directions"]

    bivariate_predicted_mean, bivariate_predicted_covariance, _ = agp.predict(
        y=data['y'][0],
        x=measured_locations,
        xs=all_locations,
        kernel_names=bivariate_kernels,
        Sigma=data['Sigma'],
        axis=-1,
        params=kernel_params,
        delta=1e-8,
    )

    predicted_mean = (
        bivariate_predicted_mean[
            :, :all_locations.shape[0]
        ] +
        1j*bivariate_predicted_mean[
            :, all_locations.shape[0]:
        ]
    )
    Krr, Kri, Kir, Kii = agp.utils.split_covariance_in_blocks(
        bivariate_predicted_covariance)
    predicted_covariance = Krr+Kii+1j*(Kir-Kri)
    predicted_variance = np.diag(predicted_covariance[0])

    # Show reconstruction
    plot_reconstruction(all_locations, measured_locations, true_sound_field[0].real, np.median(
        predicted_mean, axis=0).real, np.abs(predicted_variance))

    # Show inferences
    plot_inference_summaries(data, posterior_samples, posterior_summary)

    plt.show();


if __name__ == '__main__':
    example()
