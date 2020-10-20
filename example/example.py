import acoustic_gps as agp
import numpy as np
from matplotlib import pyplot as plt
from helper import *

def example():
    MODELPATH = '../acoustic_gps/stan_models/'
    COMPILEDPATH = './compiled/'
    N_SAMPLES = 200
    WARMUP_SAMPLES = 100
    CHAINS = 4
    L = 64
    N_reps = 1
    # full grid of locations
    nrows = 26
    dx = np.linspace(0, 2.5, nrows, endpoint = True)
    X, Y = np.meshgrid(dx, dx)
    xs = np.concatenate((X.flatten()[:, None], Y.flatten()[:, None]), axis = -1)
    p_clean, p, setup = plane_wave_field(xs, n_reps = N_reps)

    # observations
    i_few = random_locations(10, nrows, nrows) # Generate random locations with Latin Hypercube Sampling
    x = xs[i_few]
    p_measured = p[:, 0, i_few]

    # plt.figure()
    # ax = plt.gca()
    # agp.utils.show_soundfield(ax, xs.T, p_clean[0].real, what = None)
    # plt.show()
    # model options: uncomment one :)
    model_name = [ 
        # 'plane_wave_hierarchical',
        # 'bessel_isotropic',
        # 'rbf_isotropic',
        'rbf_anisotropic',
        # 'rbf_anisotropic_periodic',
        ]
    kernel_names, kernel_param_names, prior_params, stan_params = init_model(
                                                                        x,
                                                                        model=model_name[0], 
                                                                        n_basis_functions=L
                                                                        )
    # Compile Stan model?

    # agp.utils.compile_model(
    #                         model_name=model_name[0],
    #                         model_path=MODELPATH, 
    #                         compiled_save_path=COMPILEDPATH
    #                         )
    
    data = dict(
                x=x,
                # xs=xs,
                N_meas=p_measured.shape[-1],
                N_reps=N_reps,
                y=np.concatenate(
                    (p_measured.real, p_measured.imag), axis=-1),
                k=(2 * np.pi * setup['f'] / setup['c']),
                delta=1e-12,
                sigma=setup['noise_std'][0]
            )
    data.update(prior_params)

    posterior_samples, posterior_summary = agp.fit(
            COMPILEDPATH + model_name[0] + ".pkl",
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
    kernel_params = {}
    for i in kernel_param_names:
        kernel_params[i] = posterior_samples[ii]
    kernel_params["k"] = data["k"]
    if "plane_wave" in model_name:
        kernel_params["directions"] = data["wave_directions"]
    if "rbf_anisotropic" in model_name:
        kernel_params["directions"] = data["directions"]

    mu_fs, cov_fs, _ = agp.predict(
            y=p_measured,
            x=x,
            xs=xs,
            kernel_names=kernel_names,
            sigma=setup['noise_std'][0],
            axis=-1,
            sample=False,
            params=kernel_params,
            delta=1e-8,
        )
            


    # kernel_names, kernel_params, prior_params, stan_params = init_model()

if __name__ == '__main__':
    example()