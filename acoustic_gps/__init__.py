"""Summary
"""
import numpy as np
from scipy.stats import multivariate_normal
from . import kernels
from . import utils
import pickle
import pystan

def predict(x,
            xs,
            y,
            sigma=0, # noise
            kernel_names=['rbf', 'rbf', 'zero', 'zero'], #[uu, vv, uv, vu]
            axis=-1,
            delta=1e-12,
            sample = False,
            **kwargs):
    """Summary

    Args:
        x (TYPE, optional): Description
        xs (TYPE, optional): Description
        y (TYPE, optional): Description
        sigma (int, optional): Description
        kernel_name (str, optional): Description
        N_samples (int, optional): Description
        axis (TYPE, optional): Description
        **kwargs: Description

    Returns:
        TYPE: Description
    """
    k_uu = getattr(kernels, kernel_names[0])
    k_vv = getattr(kernels, kernel_names[1])
    k_uv = getattr(kernels, kernel_names[2])
    k_vu = getattr(kernels, kernel_names[3])
    
    # Real-Real
    K_xx_uu = k_uu(x1=x, x2=x, **kwargs)
    K_xsx_uu = k_uu(x1=xs, x2=x, **kwargs)
    K_xsxs_uu = k_uu(x1=xs, x2=xs, **kwargs)

    # Imag-Imag
    K_xx_vv = k_vv(x1=x, x2=x, **kwargs)
    K_xsx_vv = k_vv(x1=xs, x2=x, **kwargs)
    K_xsxs_vv = k_vv(x1=xs, x2=xs, **kwargs)

    # Real-Imag
    K_xx_uv = k_uv(x1=x, x2=x, **kwargs)
    K_xsx_uv = k_uv(x1=xs, x2=x, **kwargs)
    K_xsx_vu = k_vu(x1=xs, x2=x, **kwargs)
    K_xsxs_uv = k_uv(x1=xs, x2=xs, **kwargs)

    # Full bivariate
    K_zz = utils.stack_block_covariance(K_xx_uu, 
                                  K_xx_uv, 
                                  K_xx_uv.T, 
                                  K_xx_vv)
    K_zsz = utils.stack_block_covariance(K_xsx_uu,
                                   K_xsx_uv, 
                                   K_xsx_vu, 
                                   K_xsx_vv)
    K_zszs = utils.stack_block_covariance(K_xsxs_uu,
                                    K_xsxs_uv,
                                    K_xsxs_uv.T,
                                    K_xsxs_vv)
    noise = sigma**2 * np.identity(y.shape[axis])
    # Construct bivariate covariance
    y_mean = K_zsz @ np.linalg.inv(K_zz + noise) @ y
    y_cov = K_zszs - K_zsz @ np.linalg.inv(K_zz + noise) @ K_zsz.T + delta*np.identity(len(y_mean))
    if sample:
        y_samples = y_mean + np.linalg.cholesky(y_cov)@np.random.randn(len(y_mean))
    else:
        y_samples = []

    return y_mean, y_cov, y_samples


def fit(model_name,
        model_path,
        data,
        compile_model=False,
        n_samples=300,
        warmup_samples=150,
        chains=3,
        pars=['alpha'],
        sample=True
        ):
    with open(model_path+model_name+'.stan', "r") as f:
        model_code = f.read()

    if compile_model is True:
        # Stan compilation
        model = pystan.StanModel(model_code=model_code)
        # Save model
        pickle.dump(model, open(model_path+"compiled/" + model_name + ".pkl", "wb"))
    else:
        # Load Model
        model = pickle.load(open(model_path+"compiled/" + model_name + ".pkl", "rb"))
    if sample:
        posterior = model.sampling(
            data=data,
            iter=n_samples,
            warmup=warmup_samples,
            chains=chains,
            pars=pars
        )
    else:
        posterior = []
    return posterior