"""Plot lightcurve outputs, etc."""

import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import astropy.io.ascii as at

def stamp(img, maskmap, ax=None, cmap="cubehelix"):
    """Plot a single pixel stamp."""

    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = plt.subplot(111)

    ax.matshow(img, cmap=cmap, origin="lower", norm=colors.LogNorm())
    ax.set_xlim(-0.5,maskmap.shape[1]-0.5)
    ax.set_ylim(-0.5,maskmap.shape[0]-0.5)
    # I remain unconvinced that these are labelled right...
    # but the coordinates are plotting right, and this matches the DSS images 
    # (except East and West are flipped)!
    ax.set_ylabel("Axis 2 (Dec pixels)")
    ax.set_xlabel("Axis 1 (RA pixels)")

    return ax

def centroids(ax, init, coords=None, sources=None):
    """Plot centroids onto an image. 

    inputs
    ------
    ax: matplotlib.Axes instance with pixel stamp already plotted

    init: array-like
        initial pixel coordinates calculated from header

    coords: array-like (optional)
        another set of coordinates to plot

    sources: Table (optional)
        table of sources from daofind
    """

    ax.plot(init[1], init[0], "ko", mec="w", ms=10)

    if coords is not None:
        ax.plot(coords[1], coords[0], "r*", ms=13)

    if sources is not None:
        for i, source in enumerate(sources):
            ax.plot(source["xcentroid"], source["ycentroid"], "gD", ms=9,
                    mew=1.5)

def apertures(ax, ap_center, ap_radii, color="w"):
    """Plot apertures onto an image.

    inputs
    ------
    ax: matplotlib.Axes instance with pixel stamp already plotted

    ap_centers: array-like
        ra and dec pixel coordinates

    ap_radii: array-like
        radii of apertures in pixel coordinates
    """

    plot_center = np.array([ap_center[1], ap_center[0]])

    for rad in ap_radii:
        logging.debug("rad %f", rad)
        ap = plt.Circle(plot_center, rad, color=color, fill=False, linewidth=2)
        ax.add_artist(ap)

def lcs(lc_filename):
    """Plot lightcurves from a file."""
    lcs = at.read(lc_filename)

    # output file same as input, but change ending
    outfile = lc_filename.split("/")[-1][:-4]

    # count how many apertures were used
    num_aps = (len(lcs[0]) - 4) / 2
    ap_cols = []
    for i in np.arange(4, (4 + num_aps * 2), 2):
        print lcs.dtype.names[i]
        ap_cols.append(lcs.dtype.names[i])

    fig = plt.figure(figsize=(11,8))
    plt.suptitle(outfile)

    colors = np.array(["r", "k", "c", "m", "b", "g"])

    t = lcs["t"]
    good = np.where(t>2065)[0]

    # Plot every lightcurve and the background level
    for i, colname in enumerate(ap_cols):
        ax = plt.subplot(num_aps, 1, i+1)

        ax.plot(t[good], lcs[colname][good], ".", color=colors[i])
        ax.plot(t[good], lcs[colname.replace("flux","bkgd")][good], ".",
                color="Grey")
        ax.set_ylabel(colname)

    ax.set_xlabel("Time (d)")
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    plt.savefig("plot_outputs/{}_lcs.png".format(outfile))

if __name__=="__main__":
    lcs("test_lc.csv")
