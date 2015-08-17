
import logging, os

import matplotlib.pyplot as plt
import numpy as np

import k2phot
from k2phot import centroid
from k2phot import tpf_io
from k2phot import phot
from k2phot import plot

if __name__=="__main__":
    logging.basicConfig(level=logging.DEBUG)

    filename = "tpf/ktwo202521690-c02_lpd-targ.fits"
    outfilename = filename.split("/")[-1][:-14]

    table, times, pixels, maskmap, maskheader = tpf_io.get_data(filename)

    init = centroid.init_pos(maskheader)

    coadd = np.sum(pixels,axis=0)
    coords = centroid.quadratic_centroid(coadd, init=init)
    sources, n_sources = centroid.daofind_centroid(coadd, None, 
                                                   {"fwhm":2.5, 
                                                    "threshold":3e4})

    ax = plot.stamp(coadd, maskmap)
    plot.centroids(ax, init, coords, sources)
    plt.savefig("plot_outputs/{}_stamp.png".format(outfilename))
    plt.show()
    
#    phot.make_lc(pixels,maskmap, times, init, np.array([1,2,3,4,5]),
#                 "lcs/{}.csv".format(outfilename))

    plot.lcs("lcs/{}.csv".format(outfilename))
