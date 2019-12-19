import numpy as np
import matplotlib.pyplot as pyplot
from scipy.interpolate import griddata
from . import kernels

def show_soundfield(ax_, 
                    r_xy, 
                    p, 
                    lim=None, 
                    what = 'phase',
                    **kwargs):
    # get dimensions from r_mics and define interpolation grid
    if what == 'phase':
        z = np.angle(p)
    if what == 'spl':
        z = 20 * np.log10(np.abs(p)/2e-5)
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

def stack_block_covariance(Krr, Kri, Kir, Kii):
    K = np.concatenate(
        (np.concatenate((Krr, Kri), axis=-1),
         np.concatenate((Kir, Kii), axis=-1)),
        axis=0,
    )
    return K

def construct_complex_covariance(Krr, Kii, Kri):
    K = Krr + Kii + 1j * (Kri.T - Kri)
    Kp = Krr - Kii + 1j * (Kri.T + Kri)
    return K, Kp

def show_kernel(ax, kernel_name, x = np.linspace(0, 10, 100), normalize=False, **kwargs):
    k_uu = getattr(kernels, kernel_name)
    K = k_uu(x1=x, x2=x, **kwargs)
    if normalize:
        K /= np.max(K)
    ax.plot(x, K[0], label=kernel_name)