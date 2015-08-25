"""Plot lightcurve outputs, etc."""

import logging, os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
import astropy.io.ascii as at
from astropy.io import fits
import photutils
import K2fov.projection as proj
import K2fov.fov as fov
from K2fov.K2onSilicon import angSepVincenty,getRaDecRollFromFieldnum

import k2spin.plot

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

def ellipses(ax, ap_center, a, b, theta, ap_radii, color="w"):
    """Plot apertures onto an image.

    inputs
    ------
    ax: matplotlib.Axes instance with pixel stamp already plotted

    ap_centers: array-like
        ra and dec pixel coordinates

    ap_radii: array-like
        radii of apertures in pixel coordinates
    """

    #plot_center = np.array([ap_center[1], ap_center[0]])

    for rad in ap_radii:
        logging.debug("rad %f", rad)
        ap = photutils.EllipticalAperture(ap_center, a*rad, b*rad, theta=theta)
        ap.plot(ax=ax, color=color, linewidth=2)#kwargs={"color":color,"linewidth":2})

def lcs(lc_filename, epic=None):
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

    # Plot every lightcurve and the background level
    for i, colname in enumerate(ap_cols):
        ax = plt.subplot(num_aps, 1, i+1)

        good = np.where((t>2065) & (np.isfinite(lcs[colname])==True))[0]

        median = np.median(lcs[colname][good])
        stdev = np.std(lcs[colname][good])
        logging.info("%s %f %f", colname, median, stdev)

        three_sig = np.where(abs(median - lcs[colname])<=(3*stdev))[0]
        this_good = np.intersect1d(good, three_sig)

        ax.plot(t[this_good], lcs[colname][this_good], ".", color=colors[i])
#        ax.plot(t[good], lcs[colname.replace("flux","bkgd")][good], ".",
#                color="Grey")
        ax.set_ylabel(colname)

        if (epic is not None) and ("3.0" in colname):
            lc_compare(ax, epic, colname="Flux5")
        elif (epic is not None) and ("5.0" in colname):
            lc_compare(ax, epic, colname="Flux3")

    ax.set_xlabel("Time (d)")
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    plt.savefig("plot_outputs/{}_lcs.png".format(outfile))

def lc_compare(ax, epic, colname="Flux3"):
    """Overplot lightcurves from other authors."""

#    vfile = "vanderburg/hlsp_k2sff_k2_lightcurve_{}-c02_kepler_v1_llc.fits".format(epic)
#    if os.path.exists(vfile)==True:
#        vanderburg = fits.open("vanderburg/"+vfile)
#        vd = np.asarray([np.asarray(vanderburg[1].data[i]) 
#                         for i in range(len(vanderburg[1].data))])
#        vanderburg.close()
#        ax.plot(vd[:,0], vd[:,1], "k-")
#        ax.plot(vd[:,0], vd[:,1], "-", color="grey")

    cfile= "cody/EPIC_{}_xy_ap5.0_3.0_fixbox.dat".format(epic)
    if os.path.exists(cfile)==True:
        cody = at.read(cfile)
        ax.plot(cody["Dates"][cody["Dates"]>2065], 
                cody[colname][cody["Dates"]>2065], ".", color="g",
                alpha=0.5)

def plot_xy(lc_filename, epic=None):

    lcs = at.read(lc_filename)

    # output file same as input, but change ending
    outfilename = lc_filename.split("/")[-1][:-4]
    k2spin.plot.plot_xy(lcs["x"], lcs["y"], lcs["t"],
                        lcs["flux_3.0"], "Flux 3.0")
    plt.suptitle(outfilename, fontsize="large")
    plt.savefig("plot_outputs/{}_f3pos.png".format(outfilename))

    k2spin.plot.plot_xy(lcs["x"], lcs["y"], lcs["t"],
                        lcs["flux_5.0"], "Flux 5.0")
    plt.suptitle(outfilename, fontsize="large")
    plt.savefig("plot_outputs/{}_f5pos.png".format(outfilename))

    if epic is not None:
        cfile= "cody/EPIC_{}_xy_ap5.0_3.0_fixbox.dat".format(epic)
        if os.path.exists(cfile)==True:
            cody = at.read(cfile)
            k2spin.plot.plot_xy(cody["Xpos"], cody["Ypos"],
                                cody["Dates"], cody["Flux5"], "AMC Flux 5.0")
            plt.suptitle("{} AMC Position".format(epic))
            plt.savefig("plot_outputs/{}_AMCpos.png".format(outfilename))


def plot_chips(ax,fieldnum):
    """Plot the outline of the Kepler chips."""
    ra_deg, dec_deg, scRoll_deg = getRaDecRollFromFieldnum(fieldnum)
    ## convert from SC roll to FOV coordinates
    ## do not use the fovRoll coords anywhere else
    ## they are internal to this script only
    fovRoll_deg = fov.getFovAngleFromSpacecraftRoll(scRoll_deg)

    ## initialize class
    k = fov.KeplerFov(ra_deg, dec_deg, fovRoll_deg)

    raDec = k.getCoordsOfChannelCorners()

    light_grey = np.array([float(248)/float(255)]*3)
    #ph = proj.Gnomic(ra_deg, dec_deg)
    ph = proj.PlateCaree()
    k.plotPointing(ph,showOuts=False,plot_degrees=False,colour="k",mod3="None",
        lw=1.5)


def setup_k2_axes(ax,extents=None):
    """Set up figure axes."""

    ax.set_xlabel('R.A.',fontsize=16)
    ax.set_ylabel('Dec',fontsize=16)

    if extents is not None:
        ax.set_xlim(extents[:2])
        ax.set_ylim(extents[2:])

    ax.invert_xaxis()
    ax.minorticks_on()



def plot_four(epic, coadd, maskmap, maskheader, init, coords, sources, 
              campaign=4):

    logging.info("Plot four %s", epic)

    fig = plt.figure(figsize=(8,8))

    # First plot coadded image
    ax1 = plt.subplot(221)
    stamp(coadd, maskmap, ax=ax1, cmap="gray")
    centroids(ax1, init, coords, sources)

    # Then plot DSS/SDSS image (empty for now)
#    ax2 = plt.subplot(222)

    # Then the pixel motion across the CCD
    ax3 = plt.subplot(223)
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)

    stamp(coadd, maskmap, ax=ax3, cmap="gray")

    lcs = at.read("lcs/ktwo{}-c0{}.csv".format(epic, campaign))

    ax3.set_ylim(min(lcs["x"])*0.9,max(lcs["x"])*1.1)
    ax3.set_xlim(min(lcs["y"])*0.9,max(lcs["y"])*1.1)

    xyt = ax3.scatter(lcs["y"], lcs["x"], c=lcs["t"],  
                      edgecolor="none", alpha=0.5,  
                      vmin=np.percentile(lcs["t"], 5), 
                      vmax=np.percentile(lcs["t"], 95),
                      cmap="gnuplot")
    cbar_ticks = np.asarray(np.percentile(lcs["t"],np.arange(10,100,20)),int)
    cbar1 = fig.colorbar(xyt, cax=cax3, ticks=cbar_ticks)
    cbar1.set_label("Time (d)")

    # Then sky coordinates with the object position overlaid
    ax4 = plt.subplot(224)
    plot_chips(ax4, 4)
    setup_k2_axes(ax4)
    plt.plot(maskheader["RA_OBJ"], maskheader["DEC_OBJ"], '*', 
             color="Purple", ms=25, alpha=0.8)

    plt.suptitle("EPIC {}".format(epic), fontsize="large")
    plt.tight_layout()

    plt.savefig("plot_outputs/ktwo{}-c0{}_fourby.png".format(epic, campaign))
