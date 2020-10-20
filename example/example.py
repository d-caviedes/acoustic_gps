import acoustic_gps as agp
import numpy as np
from matplotlib import pyplot as plt
from helper import *

def example():
    MODELPATH = '../acoustic_gps/stan_models/' # Models in Stan code for Bayesian inference
    COMPILEDPATH = './compiled/' # Compiled Stan models (to load and save)
    CHAINS = 4 # number of Hamiltonian Monter Carlo chains of samples
    N_SAMPLES = 50 # number samples per chain
    WARMUP_SAMPLES = 20 # number of warmup samples (discarded after inference)
    L = 64  # number of basis kernels (for anisotropic kernels)
    N = 10 # number of observations
    
    # sound field to reconstruct (squared)
    N_reps = 1 # number of repetitions of the measurements (maybe several sine sweeps?)
    rows = 26 
    dx = np.linspace(0, 2.5, rows, endpoint = True) 
    X, Y = np.meshgrid(dx, dx)
    xs = np.concatenate(
                            (
                            X.flatten()[:, None], 
                            Y.flatten()[:, None]
                            ), 
                            axis = -1
                        ) # Total grid of locations to reconstruct
    p_clean, p, setup = plane_wave_field(
                                        xs,
                                        n_reps = N_reps, 
                                        n_waves = 1
                                        )

    # Observations: choose random locations from xs with Latin Hypercube Sampling
    i_few = random_locations(N, rows, rows)
    x = xs[i_few]
    p_measured = p[:, 0, i_few] # Single frequency

    # choose "kernel" (uncomment one)
    kernel = [ 
        # 'plane_wave_hierarchical',
        'bessel_isotropic',
        # 'rbf_isotropic',
        # 'rbf_anisotropic',
        # 'rbf_anisotropic_periodic',
        ][0]

    # Stan models need to be compiled before executed (it runs on C). 
    # Once the model is compiled, there is no need to compile it again to make inferences.
    # If changes are made in the .stan code, compilation is needed for them to take effect.
    compile = False

    if compile is True:
        agp.utils.compile_model(
                                model_name=kernel,
                                model_path=MODELPATH, 
                                compiled_save_path=COMPILEDPATH
                                )
    
    # Data used by Stan for inferences. This is common for all inferences.
    data = dict(
                x=x,
                # xs=xs,
                N_meas=p_measured.shape[-1],
                N_reps=N_reps,
                y=np.concatenate(
                    (p_measured.real, p_measured.imag), axis=-1),
                k=(2 * np.pi * setup['f'][0] / setup['c']),
                delta=1e-12,
                sigma=setup['noise_std'][0]
            )
    # Hyperparameters from prior distriutions are loaded from helper.py. It changes with model.
    kernel_names, kernel_param_names, prior_params, stan_params = init_model(
                                                                        x,
                                                                        kernel=kernel, 
                                                                        n_basis_functions=L
                                                                        )                                    
    data.update(prior_params)

    # Inferences
    posterior_samples, posterior_summary = agp.fit(
            COMPILEDPATH + kernel + ".pkl",
            data,
            pars=stan_params,
            n_samples = N_SAMPLES,
            chains=CHAINS,
            warmup_samples = WARMUP_SAMPLES
            )
    Rhat = posterior_summary['summary'][:, -1]
    
    # ax = plt.subplot(111)
    # ax.plot(Rhat)
    # ax.set_ylim(0, 1.1)
    # ax.set_xticks(np.arange(0, len(Rhat)))
    # ax.set_xticklabels(posterior_summary['summary_rownames'], rotation = 90)
    # ax.set_ylabel(r'sampling convergence $\hat{R}$')
    # ax.grid()
    # plt.show()
    N_posterior_samples = CHAINS * (N_SAMPLES - WARMUP_SAMPLES)

    kernel_params = {}
    kernel_params['N_samples'] = N_posterior_samples
    for i in kernel_param_names:
        kernel_params[i] = posterior_samples[i]
    kernel_params["k"] = data["k"]
    if "plane_wave" in kernel:
        kernel_params["directions"] = data["wave_directions"]
    if "rbf_anisotropic" in kernel:
        kernel_params["directions"] = data["directions"]
    
    mu_fs, cov_fs, _ = agp.predict(
            y=data['y'][0],
            x=x,
            xs=xs,
            kernel_names=kernel_names,
            sigma=setup['noise_std'][0],
            axis=-1,
            sample=False,
            params=kernel_params,
            delta=1e-8,
        )
            
    p_predict = mu_fs[:, :xs.shape[0]] + 1j*mu_fs[:, xs.shape[0]:]
    Krr, Kri, Kir, Kii = agp.utils.split_covariance_in_blocks(cov_fs)
    K = Krr+Kii+1j*(Kir-Kri)
    p_uncertainty = np.diag(K.mean(axis = 0))

    ax_true = plt.subplot(131)
    ax_predict = plt.subplot(132)
    ax_uncertainty = plt.subplot(133)
    agp.utils.show_soundfield(ax_true, xs.T, p_clean[0].real, what = None, cmap = 'RdBu')
    agp.utils.show_soundfield(ax_predict, xs.T, p_predict.mean(axis=0).real, what = None, cmap = 'RdBu')
    agp.utils.show_soundfield(ax_uncertainty, xs.T, p_uncertainty.real, what = None, cmap='Greys')
    
    ax_true.set_title('true sound field')
    ax_predict.set_title('mean prediction')
    ax_uncertainty.set_title('uncertainty')
    ax_uncertainty.scatter(x[:, 0], x[:, 1], marker='s', color = 'r')
    plt.tight_layout()
    plt.show()

    # kernel_names, kernel_params, prior_params, stan_params = init_model()

if __name__ == '__main__':
    example()