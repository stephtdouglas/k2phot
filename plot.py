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

    ax.plot(init[0], init[1], "ko", mec="w", ms=10)

    if coords is not None:
        ax.plot(coords[0],coords[1], "r*", ms=13)

    if sources is not None:
        for i, source in enumerate(sources):
            ax.plot(source["xcentroid"], source["ycentroid"], "gD", ms=9,
                    mew=1.5)

def lcs(lc_filename):
    """Plot lightcurves from a file."""
    lcs = at.read(lc_filename)

    # output file same as input, but change ending
    outfile = lc_filename[:-4]

    # count how many apertures were used
    num_aps = (len(lcs[0]) - 4) / 2
    ap_cols = []
    for i in np.arange(4, (4 + num_aps * 2), 2):
        print lcs.dtype.names[i]
        ap_cols.append(lcs.dtype.names[i])

    fig = plt.figure(figsize=(8,10))
    plt.suptitle(outfile)

    colors = np.array(["r", "k", "c", "m", "b", "g"])

    # Plot every lightcurve and the background level
    for i, colname in enumerate(ap_cols):
        ax = plt.subplot(num_aps, 1, i+1)

        ax.plot(lcs["t"], lcs[colname], ".", color=colors[i])
        ax.plot(lcs["t"], lcs[colname.replace("flux","bkgd")], ".",
                color="Grey")
        ax.set_ylabel(colname)

    ax.set_xlabel("Time (d)")
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    plt.savefig(outfile+".png")

if __name__=="__main__":
    lcs("test_lc.csv")
