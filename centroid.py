"""Find the centroid of a star."""

import logging

import numpy as np
from astropy.wcs import WCS
from scipy.linalg import cho_solve, cho_factor
import photutils
from photutils import morphology
import matplotlib.pyplot as plt

from k2phot import plot

def init_pos(header):
    """Find the initial position of the star using the input header."""

    ra, dec = header["RA_OBJ"], header["DEC_OBJ"]
    logging.info("Nominal position %f %f", ra, dec)
    
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
# end of code from DFM

def daofind_centroid(img, init=None, daofind_kwargs=None, max_sep=10):
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

def flux_weighted_centroid(img, box_edge, init=None, to_plot=False):
    """
    Compute the flux-weighted centroid as described on p. 105 of
    Howell (2006).

    inputs
    ------
    img: array-like

    box_edge: odd integer
        number of pixels in a box edge

    init: array-like, n=2, optional
        initial position at which to center the fitting box. 
        if None, the central pixel of the image will be chosen.

    """

    if (box_edge % 2)==0:
        logging.warning("even box edge given; increasing by one")
        L = box_edge / 2
        box_edge += 1
    else:
        L = (box_edge - 1) / 2

    if init is None:
        init = np.asarray(np.shape(img) / 2.0, int)

    Isum = np.sum(img[init[0]-L:init[0]+L+1, init[1]-L:init[1]+L+1], axis=1)
    Jsum = np.sum(img[init[0]-L:init[0]+L+1, init[1]-L:init[1]+L+1], axis=0)

    xedge = np.arange(np.shape(img)[0])[init[0]-L:init[0]+L+1]
    yedge = np.arange(np.shape(img)[1])[init[1]-L:init[1]+L+1]
#    print xedge
#    print yedge

    Ibar = (1.0 / box_edge) * np.sum(Isum)
    Jbar = (1.0 / box_edge) * np.sum(Jsum)
#    print Ibar, Jbar

    Ibar_diff = Isum - Ibar
#    print Ibar_diff, np.sum(Ibar_diff)
    Jbar_diff = Jsum - Jbar
#    print Jbar_diff, np.sum(Jbar_diff)

    xc_top = np.sum((Ibar_diff * xedge)[Ibar_diff>0])
    xc_bot = np.sum(Ibar_diff[Ibar_diff>0])

    yc_top = np.sum((Jbar_diff * yedge)[Jbar_diff>0])
    yc_bot = np.sum(Jbar_diff[Jbar_diff>0])

    xc = xc_top / xc_bot
    yc = yc_top / yc_bot

#    print xc, yc


    if to_plot:
        plt.figure(figsize=(8,10))
        grid = (4,2)
        fake_mask = np.ones_like(img)
        ax1 = plt.subplot2grid(grid, (0,0), rowspan=2)
        plot.stamp(img, fake_mask, ax=ax1)
        ax2 = plt.subplot2grid(grid, (0,1), rowspan=2)
        plot.stamp(img[init[0]-L:init[0]+L+1, init[1]-L:init[1]+L+1], 
                   fake_mask[init[0]-L:init[0]+L+1, init[1]-L:init[1]+L+1],  
                   ax=ax2)

        ax3 = plt.subplot2grid(grid, (2,0), colspan=2)
        ax3.plot(xedge, Isum, lw=2)
        ax4 = plt.subplot2grid(grid, (3,0), colspan=2)
        ax4.plot(yedge, Jsum, lw=2)

        ax3.axhline(Ibar,color="g", ls=":", lw=3)
        ax4.axhline(Jbar,color="g", ls=":", lw=3)

        ax3.axvline(init[0],color="k", ls="--", lw=2)
        ax4.axvline(init[1],color="k", ls="--", lw=2)
        ax3.axvline(xc,color="r", lw=2)
        ax4.axvline(yc,color="r", lw=2)

        # They're labeled backwards up to here, just live with it...
        ax3.set_xlabel("Y (Dec) Pixel")
        ax4.set_xlabel("X (RA) Pixel")
        ax3.set_ylabel("Flux")
        ax3.set_ylabel("Flux")

        plot.centroids(ax1, init=init, coords=(xc,yc))
        ax2.set_xticklabels(np.append(xedge,xedge[-1]+1))
        ax2.set_yticklabels(np.append(yedge,yedge[-1]+1))

        plt.suptitle("TEST",fontsize="large")
        plt.tight_layout()

    return np.array([xc, yc])

columns = ['id', 'xcentroid', 'ycentroid', 'semimajor_axis_sigma',
           'semiminor_axis_sigma', 'orientation']

def find_ellipse(img, mask, background):
    """
    Compute properties of an image (basically fit a 2D Gaussian) and
    determine the corresponding elliptical aperture. 
    
    inputs:
    -------
    filename: string
    
    r: float
        isophotal extent (multiplied by semi-major axes of fitted
        gaussian to determine the elliptical aperture)
        
    extents: array-like, optional
        xmin, xmax, ymin, ymax of sub-image
    """
    cprops = morphology.data_properties(img - background, mask=(mask==0),
                                        background = background)
    tbl = photutils.properties_table(cprops, columns=columns)
    #print tbl
    position = (cprops.xcentroid.value, cprops.ycentroid.value)
    a = cprops.semimajor_axis_sigma.value
    b = cprops.semiminor_axis_sigma.value
    theta = cprops.orientation.value
    return position, a, b, theta
