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

    # First get the position in equatorial coordinates
    ra, dec = header["RA_OBJ"], header["DEC_OBJ"]
    logging.info("Nominal position %f %f", ra, dec)
    
    # Now convert to pixel coordinates
    wcs = WCS(header)
    init = wcs.wcs_world2pix(ra, dec, 0.0)

    return init


def daofind_centroid(img, init=None, daofind_kwargs=None):
    """
    Find centroids using photutils.daofind.

    If init==None, all centroids found will be returned, along with 
    a flag indicating how many centroids were found. 

    If init is a (RA, Dec) pair of PIXEL coords, the closest centroid
    will be returned, along with a flag indicating how many centroids were 
    found. 

    Inputs
    ------
    img: array-like
        a 2-D image

    init: array-like, length 2 (optional)
        if provided, only return the source closest to the initial position

    daofind_kwargs: dict (optional)
        keyword arguments for photutils.daofind function
        default fwhm=2.5, threshold=1000

    Outputs
    -------
    coords: Table 
        xcentroid and ycentroid are the relevant columns

    num_sources:
        number of sources found. Note that if init is supplied,
        num_sources may be >1 but only one row of coords is returned

    """

    # if no daofind arguments provided, use default fwhm and background
    if daofind_kwargs is None:
        daofind_kwargs = dict()
    daofind_kwargs["fwhm"] = daofind_kwargs.get("fwhm", 2.5)
    daofind_kwargs["threshold"] = daofind_kwargs.get("threshold", 1e3)

    # find sources
    sources = photutils.daofind(img, **daofind_kwargs)

    num_sources = len(sources)
    logging.debug("%d sources", num_sources)
    logging.debug(sources)

    # if an initial position is provided, only return the closest source
    if (init is not None) and (num_sources>1):
        ra, dec = init
        sep = np.sqrt((sources["xcentroid"] - ra)**2 + 
                      (sources["ycentroid"] - dec)**2)
        loc = np.argmin(sep)
        coords = sources[loc]

    elif (init is not None) and (num_sources==1):
        coords = sources[0]

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

    to_plot: boolean 
        If True, produce a diagnostic plot of flux vs. pixel
        Default = False

    """

    if (box_edge % 2)==0:
        logging.warning("even box edge given; increasing by one")
        L = box_edge / 2
        box_edge += 1
    else:
        L = (box_edge - 1) / 2

    if init is None:
        init = np.asarray(np.shape(img) / 2.0, int)
        raw_init = init
    else:
        raw_init = np.copy(init)
        init = np.asarray(np.round(init), int)
#    print init, raw_init

    sub_img = img[init[1]-L:init[1]+L+1, init[0]-L:init[0]+L+1]

    Isum = np.sum(sub_img, axis=0)
    Jsum = np.sum(sub_img, axis=1)

#    print np.shape(img)
    xedge = np.arange(np.shape(img)[1])[init[0]-L:init[0]+L+1]
    yedge = np.arange(np.shape(img)[0])[init[1]-L:init[1]+L+1]
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

#    logging.debug("init %.2f %.2f c %.2f %.2f", init[0], init[1], xc, yc)


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
        ax3.step(xedge, Isum, lw=2, where="mid")
        ax4 = plt.subplot2grid(grid, (3,0), colspan=2)
        ax4.step(yedge, Jsum, lw=2, where="mid")

        ax3.axhline(Ibar,color="g", ls=":", lw=3)
        ax4.axhline(Jbar,color="g", ls=":", lw=3)

        ax3.axvline(raw_init[0],color="k", ls="--", lw=2)
        ax4.axvline(raw_init[1],color="k", ls="--", lw=2)
        ax3.axvline(xc,color="r", lw=2)
        ax4.axvline(yc,color="r", lw=2)

        ax3.set_xlabel("X Pixel")
        ax4.set_xlabel("Y Pixel")
        ax3.set_ylabel("Flux")
        ax3.set_ylabel("Flux")

        plot.centroids(ax1, init=init, coords=(xc,yc))
        ax2.set_xticklabels(np.append(xedge,xedge[-1]+1))
        ax2.set_yticklabels(np.append(yedge,yedge[-1]+1))

#        plt.suptitle("TEST",fontsize="large")
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
