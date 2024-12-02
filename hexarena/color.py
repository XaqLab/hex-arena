import csv
from pathlib import Path
from matplotlib.colors import ListedColormap
import numpy as np
from numpy.fft import fftfreq, ifft2

from .alias import Array, RandGen

CMAP_PTH = Path(__file__).parent/'cmap.csv'

def get_cmap() -> ListedColormap:
    r"""Returns the color map used for box color.

    Returns
    -------
    cmap:
        A callable that can be called as `c = cmap(x)`, in which `x` is an float
        in [0, 1], and `c` is the RGBA value.

    """
    colors = []
    with open(CMAP_PTH, newline='') as f:
        reader = csv.reader(f)
        for line in reader:
            colors.append([int(v) for v in line])
    cmap = ListedColormap(np.array(colors).astype(float)/255)
    return cmap

def get_cue_movie(
    cue: float,
    num_steps: int,
    size: tuple[int, int] = (128, 96),
    kappa: float = 0.1,
    alpha: float = 1.,
    beta: float = -2.,
    theta_min: float = -np.pi/2,
    theta_max: float = 0,
    rng: RandGen|int|None = None,
) -> Array:
    r"""Creates color cue movie.

    Args
    ----
    cue:
        Color cue value in [0, 1], e.g., from blueish to reddish.
    num_steps:
        Number of time steps of the color cue movie.
    size:
        A tuple of `(height, width)` for the color cue movie size.
    kappa:
        A non-negative float for stimulus reliability. Greater value means lower
        noise level. `kappa=0` for no signal.
    alpha:
        Exponential coeffieicnt for temporal correlation.
    beta:
        Exponential coefficient for spatial correlation.
    theta_min, theta_max:
        Range along the color circle loaded from `CMAP_PTH`.
    rng:
        Random number generator or seed.

    Returns
    -------
    values:
        A 3D array of shape `(num_steps, height, width)` for the color cue movie,
        whose values are in [0, 1) with periodic boundary. Using the color map
        in `CMAP_PTH`, values around 1 correspond to red, and values around 0.75
        correspond to blue.

    """
    rng = np.random.default_rng(rng)

    fx, fy = np.meshgrid(fftfreq(size[1]), fftfreq(size[0]))
    f = (fx**2+fy**2)**0.5
    gamma = np.exp(-alpha*f)
    eps = 1e-5/max(size)
    S_f = (f+eps)**beta
    S_f[0, 0] = 0.

    theta = cue*(theta_max-theta_min)+theta_min
    noise, values = None, []
    for _ in range(num_steps):
        _noise = (rng.normal(size=size)+1j*rng.normal(size=size))*S_f**0.5
        if noise is None:
            noise = _noise
        else:
            noise = gamma*noise+(1-gamma)*_noise
        x_comp = ifft2(noise)+kappa*np.exp(1j*theta)
        x_circ = np.angle(x_comp)
        x_circ = 0.5*x_circ/np.pi+(x_circ<=0).astype(float)
        values.append(x_circ)
    values = np.stack(values)
    return values

def get_cue_array(
    cue: float,
    **kwargs,
) -> Array:
    r"""Creates color cue array.

    Args
    ----
    cue:
        Color cue value in [0, 1], see `get_cue_movie` for more details.
    kwargs:
        Keyword arguments for `get_cue_movie`.

    Returns
    -------
    values:
        A 2D array of shape `(height, width)` which is the first frame of a
        color cue movie, see `get_cue_movie` for more details.

    """
    values = get_cue_movie(cue, 1, **kwargs)[0]
    return values
