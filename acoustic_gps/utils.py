"""Summary
"""
import numpy as np
import matplotlib.pyplot as pyplot
from scipy.interpolate import griddata
from . import kernels
import pystan
import pickle

def show_soundfield(ax_, 
                    r_xy, 
                    p, 
                    lim=None, 
                    what = 'phase',
                    **kwargs):
    """Summary
    
    Parameters
    ----------
    ax_ : TYPE
        Description
    r_xy : TYPE
        Description
    p : TYPE
        Description
    lim : None, optional
        Description
    what : str, optional
        Description
    **kwargs
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    # get dimensions from r_mics and define interpolation grid
    if what == 'phase':
        z = np.angle(p)
    if what == 'spl':
        z = 20 * np.log10(np.abs(p)/2e-5)
    if what == None:
        z = p
    xmin, ymin = r_xy.min(axis=1)
    xmax, ymax = r_xy.max(axis=1)
    xg = np.linspace(xmin, xmax, 100)
    yg = np.linspace(ymin, ymax, 100)
    Xg, Yg = np.meshgrid(xg, yg)
    if lim is None:
        lim = (z.min(), z.max())

    # interpolate data on grid
    zg = griddata(
        (r_xy[0], r_xy[1]), z, (Xg.ravel(), Yg.ravel()), method="cubic"
    )
    Zg = zg.reshape(Xg.shape)
    
    cs = ax_.pcolormesh(Xg, Yg, Zg, vmin=lim[0], vmax=lim[1], **kwargs)
    ax_.set_aspect("equal")
    return cs

def db_spl(p):
    return 20 * np.log10(np.abs(p)/2e-5)

def stack_block_covariance(Krr, Kri, Kir, Kii):
    """Summary
    
    Parameters
    ----------
    Krr : TYPE
        Description
    Kri : TYPE
        Description
    Kir : TYPE
        Description
    Kii : TYPE
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    K = np.concatenate(
        (np.concatenate((Krr, Kri), axis=-1),
         np.concatenate((Kir, Kii), axis=-1)),
        axis=-2,
    )
    return K

def complex_covariance_from_real(Krr, Kii, Kri):
    """Summary
    
    Parameters
    ----------
    Krr : TYPE
        Description
    Kii : TYPE
        Description
    Kri : TYPE
        Description
    
    Returns
    -------
    TYPE
        Description
    """
    K = Krr + Kii + 1j * (Kri.T - Kri)
    Kp = Krr - Kii + 1j * (Kri.T + Kri)
    return K, Kp

def split_covariance_in_blocks(K):
    """Summary
    Split matrix
    Parameters
    ----------
    K : ndarray [..., N, N]
        Description
    """
    N = int(K.shape[-1]/2)
    K_top_left = np.copy(K[..., :N, :N])
    K_top_right = np.copy(K[..., :N, N:])
    K_bottom_left = np.copy(K[..., N:, :N])
    K_bottom_right = np.copy(K[..., N:, N:])

    return K_top_left, K_top_right, K_bottom_left, K_bottom_right

def show_kernel(ax, 
                kernel_name, 
                x = np.linspace(0, 10, 100), 
                color = 'C0',
                alpha = 1,
                normalize=False, 
                dim='1D',
                **kwargs):
    """Summary
    
    Parameters
    ----------
    ax : TYPE
        Description
    kernel_name : TYPE
        Description
    x : TYPE, optional
        Description
    normalize : bool, optional
        Description
    dim : str, optional
        Description
    **kwargs
        Description
    """
    k_uu = getattr(kernels, kernel_name)
    K = k_uu(x1=x, x2=x, **kwargs)

    if normalize:
        K /= np.max(K)
    if dim=='1D':
        ax.plot(x, K[0, 0], label=kernel_name, color = color, alpha = alpha)
    if dim=='2D':
        ax.imshow(K[0], extent= [x[:, 0].min(), x[:, 0].max(), x[:, 1].min(), x[:, 1].max()])

def compile_model(model_name, model_path, compiled_save_path):
    """Summary
    
    Parameters
    ----------
    model_name : TYPE
        Description
    model_path : TYPE
        Description
    compiled_save_path : TYPE
        Description
    """
    with open(model_path+model_name+'.stan', "r") as f:
        model_code = f.read()
    # Stan compilation
    model = pystan.StanModel(model_code=model_code)
    # Save model
    pickle.dump(model, open(compiled_save_path + model_name + ".pkl", "wb"))

def nmse(y_meas, y_predicted, axis=(-1,)):
    norm = 1
    for i in axis:
        norm *= y_meas.shape[i]
    nmse = np.sum(np.abs(y_meas - y_predicted)**2/np.abs(y_meas)**2, axis = axis) / norm
    return nmse

def find_nearest(array, value):
    """Find nearest value in an array and its index.

    Returns
    -------
    value
        Value of nearest entry in array
    idx
        Index of that value
    """
    idx = []
    for val in value:
        idx.append((np.abs(array - val)).argmin())
    return array[idx], idx