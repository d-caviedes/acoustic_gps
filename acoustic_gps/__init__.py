"""Summary
"""
import numpy as np
from scipy.stats import multivariate_normal
from . import kernels
from . import utils
import pickle
import pystan
import os

BASE_DIRS = os.path.dirname(__file__)
STAN_MODELS = os.path.join(BASE_DIRS + os.sep, 'stan_models')
COMPILED_STAN_MODELS = os.path.join(STAN_MODELS + os.sep, 'compiled')

def predict(x,
            xs,
            y,
            Sigma,  # noise
            # [uu, vv, uv, vu] TODO: Change to dictionary instead of positional
            kernel_names=['rbf', 'rbf', 'zero', 'zero'],
            axis=-1,
            delta=1e-12,
            sample=False,
            **kwargs):
    """Summary

    Args:
        x (TYPE, optional): Description
        xs (TYPE, optional): Description
        y (TYPE, optional): Description
        Sigma (int, optional): Description
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

    # x-x
    K_xx_uu = k_uu(x1=x, x2=x, **kwargs)
    K_xx_vv = k_vv(x1=x, x2=x, **kwargs)
    K_xx_uv = k_uv(x1=x, x2=x, **kwargs)
    K_xx_vu = k_vu(x1=x, x2=x, **kwargs)

    # Full bivariate
    K_zz = utils.stack_block_covariance(K_xx_uu,
                                        K_xx_uv,
                                        K_xx_vu,
                                        K_xx_vv)

    del K_xx_uu, K_xx_vv, K_xx_uv, K_xx_vu

    # xs-x
    K_xsx_uu = k_uu(x1=xs, x2=x, **kwargs)
    K_xsx_vv = k_vv(x1=xs, x2=x, **kwargs)
    K_xsx_uv = k_uv(x1=xs, x2=x, **kwargs)
    K_xsx_vu = k_vu(x1=xs, x2=x, **kwargs)
    # Full bivariate
    K_zsz = utils.stack_block_covariance(K_xsx_uu,
                                         K_xsx_uv,
                                         K_xsx_vu,
                                         K_xsx_vv)

    del K_xsx_uu, K_xsx_vv, K_xsx_uv, K_xsx_vu

    # xs-xs
    K_xsxs_uu = k_uu(x1=xs, x2=xs, **kwargs)
    K_xsxs_vv = k_vv(x1=xs, x2=xs, **kwargs)
    K_xsxs_uv = k_uv(x1=xs, x2=xs, **kwargs)
    K_xsxs_vu = k_vu(x1=xs, x2=xs, **kwargs)
    # Full bivariate
    K_zszs = utils.stack_block_covariance(K_xsxs_uu,
                                          K_xsxs_uv,
                                          K_xsxs_vu,
                                          K_xsxs_vv)

    del K_xsxs_uu, K_xsxs_vv, K_xsxs_uv, K_xsxs_vu
    # Construct bivariate covariance
    y_mean = np.einsum(
        'nij, nj -> ni',
        K_zsz,
        np.einsum(
            'nij, j -> ni',
            np.linalg.inv(K_zz + Sigma),
            y
        )
    )
    y_cov = (K_zszs
             - np.einsum(
                 'nij, njk -> nik',
                 K_zsz,
                 np.einsum(
                     'nij, nkj -> nik',
                     np.linalg.inv(K_zz + Sigma),
                     K_zsz)
             )
             )
    if sample:
        delta_ = np.copy(delta)
        cholesky = np.empty(y_cov.shape)
        for i in range(y_mean.shape[0]):
            while True:
                try:
                    cholesky[i] = np.linalg.cholesky(y_cov[i])
                    break
                except np.linalg.LinAlgError:
                    y_cov[i] += delta_ * np.identity(y_mean.shape[-1])
                    delta_ *= 10
            delta_ = np.copy(delta)
        y_samples = (y_mean + np.einsum('nij, nj -> ni',
                                        cholesky, np.random.randn(*y_mean.shape)))
        del cholesky
    else:
        y_samples = []

    return y_mean, y_cov, y_samples


def mc_sampling(
        data,
        kernel=['rbf_isotropic'],
        model_path=COMPILED_STAN_MODELS,
        n_samples=300,
        warmup_samples=150,
        chains=3,
        pars=['alpha']
):
    model = pickle.load(
        open(os.path.join(COMPILED_STAN_MODELS + os.sep, kernel + '.pkl'), "rb"))

    posterior_ = model.sampling(
        data=data,
        iter=n_samples,
        warmup=warmup_samples,
        chains=chains,
        pars=pars
    )
    posterior_samples = posterior_.extract(pars=pars, permuted=True)
    posterior_summary = posterior_.summary(pars=pars)
    return posterior_samples, posterior_summary


def map_estimation(model_path,  # TODO: implement this properly
                   data,
                   seed=1000,
                   iter=1e6,
                   algorithm='LBFGS',
                   tol_rel_grad=1e7

                   ):
    model = pickle.load(open(model_path, "rb"))

    map = model.optimizing(
        data=data,
        seed=seed,
        iter=iter,
        algorithm=algorithm,
        tol_rel_grad=tol_rel_grad
    )
    return map
