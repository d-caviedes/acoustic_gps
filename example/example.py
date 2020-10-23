import acoustic_gps as agp
import numpy as np
from matplotlib import pyplot as plt
from helper import *
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('text', usetex=True)


# TODO: Generalize noise.
# TODO: Explain how Bivariate to Complex works in the code.
# TODO: Comment code properly
# TODO: Header and paper citing at the top (any CC license)?
# TODO: colorbars?
# TODO: Inference summaries -> Shape of figures according to number of parameters

def example():
    MODELPATH = '../acoustic_gps/stan_models/' # Models in Stan code for Bayesian inference
    COMPILEDPATH = './compiled/' # Compiled Stan models (to load and save)
    CHAINS = 4 # number of Hamiltonian Monter Carlo chains of samples
    N_SAMPLES = 400 # number samples per chain
    WARMUP_SAMPLES = 200 # number of warmup samples (discarded after inference)
    L = 64  # number of basis kernels (for anisotropic kernels)
    N = 15 # number of observations
    
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
    p_true, p, setup = plane_wave_field(
                                        xs,
                                        n_reps = N_reps, 
                                        n_waves = 2
                                        )

    # Observations: choose random locations from xs with Latin Hypercube Sampling
    i_few = random_locations(N, rows, rows)
    x = xs[i_few]
    p_measured = p[:, 0, i_few] # Single frequency

    # choose "kernel" (uncomment one)
    kernel = [ 
        'plane_wave_hierarchical',
        # 'bessel_isotropic',
        # 'rbf_isotropic',
        # 'rbf_anisotropic',
        # 'rbf_anisotropic_periodic',
        ][0]

    # Stan models need to be compiled before executed (it runs on C). 
    # Once the model is compiled, there is no need to compile it again to make inferences.
    # If changes are made in the .stan code, compilation is needed for them to take effect.
    compile = True

    if compile is True:
        agp.utils.compile_model(
                                model_name=kernel,
                                model_path=MODELPATH, 
                                compiled_save_path=COMPILEDPATH
                                )
    
    # Data used by Stan for inferences. This is common for all model.
    data = dict(
                x=x,
                N_meas=p_measured.shape[-1],
                N_reps=N_reps,
                y=np.concatenate(
                    (p_measured.real, p_measured.imag), axis=-1),
                k=(2 * np.pi * setup['f'][0] / setup['c']),
                delta=1e-10,
                Sigma=(setup['noise_std'][0]**2)/2 * np.eye(2*p_measured.shape[-1])
            )
    # Hyperparameters from prior distriutions are loaded from helper.py. It changes with model.
    kernel_names, kernel_param_names, prior_params, stan_params = init_model(
                                                                        x,
                                                                        kernel=kernel, 
                                                                        n_basis_functions=L
                                                                        )                                    
    data.update(prior_params)

    # Monter Carlo sampling
    posterior_samples, posterior_summary = agp.mc_sampling(
            COMPILEDPATH + kernel + ".pkl",
            data=data,
            pars=stan_params,
            n_samples = N_SAMPLES,
            chains=CHAINS,
            warmup_samples = WARMUP_SAMPLES
            )

    

    # Reconstruction
    kernel_params = {}
    kernel_params['N_samples'] = 1
    for i in kernel_param_names:
        kernel_params[i] = np.median(posterior_samples[i], axis = 0)[None]
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
            Sigma=data['Sigma'],
            axis=-1,
            sample=False,
            params=kernel_params,
            delta=1e-8,
        )
            
    p_predict = mu_fs[:, :xs.shape[0]] + 1j*mu_fs[:, xs.shape[0]:]
    Krr, Kri, Kir, Kii = agp.utils.split_covariance_in_blocks(cov_fs)
    K = Krr+Kii+1j*(Kir-Kri)
    p_uncertainty = np.diag(np.median(K,axis = 0))


    # Show reconstruction
    plot_reconstruction(xs, x, p_true[0].real, np.median(p_predict,axis=0).real, np.abs(p_uncertainty))
    
    # Show inferences
    plot_inference_summaries(data, posterior_samples, posterior_summary)

    plt.show()

if __name__ == '__main__':
    example()