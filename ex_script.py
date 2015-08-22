
import logging, os

import astropy.io.ascii as at
import matplotlib.pyplot as plt
import numpy as np

from k2phot import centroid
from k2phot import tpf_io
from k2phot import phot
from k2phot import plot

alphas = "ABCDEFG"

def run_one(filename):

    outfilename = filename.split("/")[-1][:-17]
    logging.info(outfilename)
    table, times, pixels, maskmap, maskheader, kpmag = tpf_io.get_data(filename)

    init = centroid.init_pos(maskheader)
    logging.debug("init %f %f", init[0], init[1])

    coadd = np.sum(pixels,axis=0)
    coords = centroid.flux_weighted_centroid(coadd, 7, init=init)
    coadd_bkgd = phot.calc_bkgd(coadd, maskmap, coords[:2], 2)
    if np.sqrt((coords[0] - init[0])**2 + (coords[1] - init[1])**2)>1:
        logging.warning("Centroid (%f %f) far from init (%f %f)", 
                        coords[0], coords[1], init[0], init[1])
    else:
        logging.debug("coords %f %f", coords[0], coords[1])

    sources, n_sources = centroid.daofind_centroid(coadd, None, 
                                                   {"fwhm":2.5, 
                                                    "threshold":coadd_bkgd,  
                                                    "sharplo":0.01,  
                                                    "sharphi":5.0})
    if n_sources==0:
        logging.info("bkgd: %f max: %f", coadd_bkgd, max(coadd.flatten()))
    logging.debug("sources")
    logging.debug(sources)

    radii = np.array([1,2,3,4,5])

    ax = plot.stamp(coadd, maskmap)
    plot.centroids(ax, init, coords, sources)
    plot.apertures(ax, coords, radii)
    plt.savefig("plot_outputs/{}_stamp.png".format(outfilename))
    plt.title(outfilename)
#    plt.show()
    plt.close("all")

#    phot.make_lc(pixels,maskmap, times, init, radii,
#                 "lcs/{}.csv".format(outfilename))

    epic = outfilename.split("-")[0][4:]
#    plot.lcs("lcs/{}.csv".format(outfilename), epic=epic)

#    plot.plot_xy("lcs/{}.csv".format(outfilename), epic=epic)
"""
    for i, source in enumerate(sources):
        sletter = alphas[i]

        radii = np.array([1,2,3])
        ax = plot.stamp(coadd, maskmap)
        plot.centroids(ax, init, coords, sources)

        new_init = np.array([source["xcentroid"],source["ycentroid"]])
        plot.apertures(ax, new_init[::-1], radii)
        plt.savefig("plot_outputs/{}{}_stamp.png".format(outfilename,
                                                         sletter))
        plt.title(outfilename+sletter)
        plt.close("all")

        phot.make_lc(pixels, maskmap, times, new_init, radii,
                     "lcs/{}{}.csv".format(outfilename, sletter))
        plot.lcs("lcs/{}{}.csv".format(outfilename, sletter), epic=epic)
"""

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)

#    filename = "tpf/ktwo202521690-c02_lpd-targ.fits.gz"
#    filename = "tpf/ktwo202533810-c02_lpd-targ.fits.gz"
#    filename = "tpf/ktwo202539362-c02_lpd-targ.fits.gz"
    filename = "tpf/ktwo203033414-c02_lpd-targ.fits.gz"
#    run_one(filename)

#"""
#    tpfs = at.read("tpf_test.lst")
    tpfs = at.read("c4_tpfs.lst")
    for fname in tpfs["filename"]:
        if os.path.exists(fname)==True:
            run_one(fname)
        else:
            logging.warning("skipping %s", fname)
#"""
