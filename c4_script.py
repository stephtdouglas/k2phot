
import logging, os
from datetime import date

import astropy.io.ascii as at
import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import sigma_clipped_stats

from k2phot.config import *
from k2phot import centroid
from k2phot import tpf_io
from k2phot import phot
from k2phot import plot

alphas = "BCDEFGHIJKLMN"
today = date.today().isoformat()

def run_one(filename, output_f=None, extract_companions=False):
    """Generate light curves from one TPF 

    Inputs
    ------
    filename: string
        FULL PATH to TPF

    output_f: file object, optional
        if provided, various diagnostic/informational data will be written out.
    
    extract_companions: bool, optional, default=False
        if true, light curves and plots will also be generated for any other
        DAOfind sources in the coadded image

    """

    outfilename = filename.split("/")[-1][:-14]
    logging.info(outfilename)
    table, times, pixels, maskmap, maskheader, kpmag = tpf_io.get_data(filename)

    init = centroid.init_pos(maskheader)
    logging.info("init %f %f", init[0], init[1])

    min_ax = np.argmin(np.shape(maskmap))
    min_box = np.shape(maskmap)[min_ax]
    if min_box>=9:
       min_box = 9
       logging.info("min_box set to 9")
    elif min_ax==0:
        for row in maskmap:
            row_len = len(np.where(row>0)[0])
            if row_len < min_box:
                min_box = row_len
                logging.debug("new min_box %d", min_box)
    else:
        for i in range(np.shape(maskmap)[1]):
            col = maskmap[:, i]
            col_len = len(np.where(col>0)[0])
            if col_len < min_box:
                min_box = row_len
                logging.debug("new min_box %d", min_box)

    fw_box = (min_box / 2) * 2 + 1
    logging.info("fw box %d",fw_box)

    coadd = np.sum(pixels,axis=0)
    coords = centroid.flux_weighted_centroid(coadd, fw_box, init=init)
    mean, median, std = sigma_clipped_stats(coadd, mask=(maskmap==0),
                                            sigma=3.0, iters=3)
    coadd_bkgd = median

    if np.sqrt((coords[0] - init[0])**2 + (coords[1] - init[1])**2)>1:
        logging.warning("Centroid (%f %f) far from init (%f %f)", 
                        coords[0], coords[1], init[0], init[1])
    else:
        logging.debug("coords %f %f", coords[0], coords[1])

    dkwargs = {"fwhm":2.5, "threshold":coadd_bkgd,  
               "sharplo":0.01,"sharphi":5.0}
    sources, n_sources = centroid.daofind_centroid(coadd, None, dkwargs)
    if n_sources==0:
        logging.info("bkgd: %f max: %f", coadd_bkgd, max(coadd.flatten()))
    logging.debug("sources")
    logging.debug(sources)

    ap_min, ap_max, ap_step = 2.5, 7, 0.5
    radii = np.arange(ap_min, ap_max, ap_step)

    ap_type = "circ"
#    phot.make_circ_lc(pixels, maskmap, times, init, radii,
#                      "base_path+lcs/{}.csv".format(outfilename), fw_box)

    epic = outfilename.split("-")[0][4:]
    logging.info(epic)
#    plot.lcs(base_path+"lcs/{}.csv".format(outfilename), epic=epic)

    plot.plot_four(epic, coadd, maskmap, maskheader, init, coords, sources,
                   campaign=4)

#    plot.plot_xy(base_path+"lcs/{}.csv".format(outfilename), epic=epic)

    if output_f is not None:
        output_f.write("\n{},{}".format(outfilename,epic))
        output_f.write(",{0:.6f},{1:.6f}".format(maskheader["RA_OBJ"],
                                                 maskheader["DEC_OBJ"]))
        print type(init[0]), type(init[1])
        print init[0].dtype, init[1].dtype
        print init[0], init[1]
        output_f.write(",{0:.6f},{1:.6f}".format(np.float64(init[0]),
                                                 np.float64(init[1])))
        output_f.write(",{0:.2f},{1}".format(kpmag, n_sources))
        output_f.write(",{0:.2f},{1:.6f}".format(dkwargs["fwhm"], 
                                                 dkwargs["threshold"]))
        output_f.write(",{0:.2f},{1:.2f}".format(dkwargs["sharplo"],  
                                                 dkwargs["sharphi"]))
        output_f.write(",{0:.2f},{1}".format(fw_box,ap_type))
        output_f.write(",{0:.2f},{1:.2f},{2:.2f}".format(ap_min, ap_max,   
                                                         ap_step))

    if extract_companions:
        for i, source in enumerate(sources):
            # Calculate the separation between this source and the target
            sep = np.sqrt((init[0] - source["xcentroid"])**2 +
                          (init[1] - source["ycentroid"])**2)

            # If the source is close to the center, just skip (it's the target)
            if sep<3:
                continue
            
            # If the source is further away, compute a lightcurve for it.
            sletter = alphas[i]

            init2 = np.array([source["xcentroid"], source["ycentroid"]])
            coords2 = centroid.flux_weighted_centroid(coadd, 3, init=init2)
            radii = np.arange(0.5, (sep / 2.0) + 0.1 ,0.5)

            ax = plot.stamp(coadd, maskmap)
            plot.centroids(ax, init2, coords2, sources)
            plot.apertures(ax, init2, radii)
            plt.savefig(base_path+"plot_outputs/{}{}_stamp.png".format(outfilename,sletter))
            plt.title(outfilename+sletter)

            phot.make_circ_lc(pixels, maskmap, times, init2[::-1], radii,
                              base_path+"lcs/{}{}.csv".format(outfilename, 
                                                              sletter))
            plot.lcs(base_path+"lcs/{}{}.csv".format(outfilename, sletter), 
                     epic=epic)
    
            plt.close("all")

def run_list(listname, save_output=False):

    # Need to change base_path for tpfs too before running on Yeti
    # although base_path is in run_one, it expects the full path to tpfs
 
    tpfs = at.read(listname)

    output_f = None
    
    if save_output:
        tfile = base_path+"tables/{0}_phot_{1}.csv".format(listname[:-4], today)
        output_f = open(, "w")
        output_f.write("output_file, EPIC, RA_OBJ, DEC_OBJ, pix_ra, pix_dec")
        output_f.write(", kpmag, n_src, daofind_fwhm, daofind_threshold")
        output_f.write(", daofind_sharplo, daofind_sharphi, centroid_box")
        output_f.write(", ap_type, ap_min, ap_max, ap_step")

    for fname in tpfs["filename"]:
        if os.path.exists(fname)==True:
            logging.warning(fname)
            run_one(fname)
#            run_one(fname, output_f)
        else:
            logging.warning("skipping %s", fname)

    if save_output: 
        output_f.close()

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO, 
                        format="%(asctime)s - %(name) - %(message)s")
    
    # on Yeti use c4_tpfs_yeti.lst

#    filename = "/home/stephanie/Dropbox/c4_tpf/ktwo210408563-c04_lpd-targ.fits"
#    run_one(filename)


#    run_list("c4_tpfs_centroidproblems.lst", save_output=True)
#    run_list("c4_spd_now_lpd.lst", save_output=True)
    run_list("c4_tpfs.lst", save_output=True)

    
