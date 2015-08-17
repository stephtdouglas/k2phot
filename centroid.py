"""Find the centroid of a star."""

import logging

import numpy as np
from astropy.wcs import WCS
from scipy.linalg import cho_solve, cho_factor
import photutils


def init_pos(header):
    """Find the initial position of the star using the input header."""

    ra, dec = header["RA_OBJ"], header["DEC_OBJ"]
    
    wcs = WCS(header)
    init = wcs.wcs_world2pix(ra, dec, 0.0)[::-1]

    return init


# Below is adapted from code DFM gave me in demo.ipynb
# These are some useful things to pre-compute and use later.
_x, _y = np.meshgrid(range(-1, 2), range(-1, 2), indexing="ij")
_x, _y = _x.flatten(), _y.flatten()
_AT = np.vstack((_x*_x, _y*_y, _x*_y, _x, _y, np.ones_like(_x)))
_ATA = np.dot(_AT, _AT.T)
factor = cho_factor(_ATA, overwrite_a=True)

# This function finds the centroid and second derivatives in a 3x3 patch.
def fit_3x3(img):
    a, b, c, d, e, f = cho_solve(factor, np.dot(_AT, img.flatten()))
    m = 1. / (4 * a * b - c*c)
    x = (c * e - 2 * b * d) * m
    y = (c * d - 2 * a * e) * m
    dx2, dy2, dxdy = 2 * a, 2 * b, c
    return [x, y, dx2, dy2, dxdy]

# This function finds the centroid in an image.
# You can provide an estimate of the centroid using WCS.
def quadratic_centroid(img, init=None):
    if init is None:
        xi, yi = np.unravel_index(np.argmax(img), img.shape)
    else:
        xi, yi = map(int, map(np.round, init))
        ox, oy = np.unravel_index(np.argmax(img[xi-1:xi+2, yi-1:yi+2]), (3, 3))
        xi += ox - 1
        yi += oy - 1
    assert (xi >= 1 and xi < img.shape[0] - 1), "effed, x"
    assert (yi >= 1 and yi < img.shape[1] - 1), "effed, y"
    pos = fit_3x3(img[xi-1:xi+2, yi-1:yi+2])
    pos[0] += xi
    pos[1] += yi
    return pos


def daofind_centroid(img, init=None, daofind_kwargs=None):
    """
    Find centroids using photutils.daofind.

    If init==None, all centroids found will be returned, along with 
    a flag indicating how many centroids were found. 

    If init is a (RA, Dec) pair of PIXEL coords, the closest centroid
    will be returned, along with a flag indicating how many centroids were 
    found. 

    Outputs
    -------
    coords: Table 
        xcentroid and ycentroid are the relevant columns

    num_sources:
        number of sources found. Note that if init is supplied,
        num_sources may be >1 but only one row of coords is returned

    """

    if daofind_kwargs is None:
        daofind_kwargs = dict()
    daofind_kwargs["fwhm"] = daofind_kwargs.get("fwhm", 2.5)
    daofind_kwargs["threshold"] = daofind_kwargs.get("threshold", 1e3)

    sources = photutils.daofind(img, **daofind_kwargs)

    num_sources = len(sources)
    logging.debug("%d sources", num_sources)
    logging.debug(sources)

    if (init is not None) and (num_sources>1):
        ra, dec = init
        sep = np.sqrt((sources["xcentroid"] - ra)**2 + 
                      (sources["ycentroid"] - dec)**2)
        loc = np.argmin(sep)
        coords = sources[loc]

    else:
        coords = sources
    
    return coords, num_sources
