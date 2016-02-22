
import logging, os, sys
from datetime import date

import astropy.io.ascii as at
import numpy as np
from astropy.stats import sigma_clipped_stats

logging.basicConfig(level=logging.DEBUG)

import k2phot.config
from k2phot import centroid
from k2phot import tpf_io
from k2phot import phot
from k2phot import plot

today = date.today().isoformat()

def run_one(filename, output_f=None, fw_box=None):
    """ 
    Extract a light curve for one star. 

    Inputs
    ------
    filename: string
        filename for target pixel file

    output_f: file object (optional)
        if provided, statistics from the extraction will be printed to this file

    fw_box: odd integer (optional)
        size of the box within which to calculate the flux-weighted centroid.
        Must be odd. Defaults to the shortest dimension of the pixel stamp,
        or 9 if larger than 9. 

    """

    # Clip part of the filename for use in saving outputs
    outfilename = filename.split("/")[-1][:-14]
    logging.info(outfilename)
    # Retrieve the data
    table, times, pixels, maskmap, maskheader, kpmag = tpf_io.get_data(filename)

    # Use the RA/Dec from the header as the initial position for calculation
    init = centroid.init_pos(maskheader)
    logging.info("init %f %f", init[0], init[1])

    # If not provided, calculate the maximum possible centroid box size.
    # Maximum possible box is 9, unless set larger by the user with fw_box
    if fw_box is None:
        min_ax = np.argmin(np.shape(maskmap))
        min_box = np.shape(maskmap)[min_ax]
        if min_box>=9:
           min_box = 9
           logging.debug("min_box set to 9")
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

    # Co-add all the sub-images and calculate the flux-weighted centroid
    coadd = np.sum(pixels,axis=0)
    coords = centroid.flux_weighted_centroid(coadd, fw_box, init=init)

    # Background of the co-added stampl is just the sigma-clipped median
    mean, median, std = sigma_clipped_stats(coadd, mask=(maskmap==0),
                                            sigma=3.0, iters=3)
    coadd_bkgd = median

    # Warn the user if the calculated initial coordinates are too far
    # from the initial coordinates (indicting that there's a brighter nearby 
    # star pulling the centroid off of the target)
    if np.sqrt((coords[0] - init[0])**2 + (coords[1] - init[1])**2)>1:
        logging.warning("Centroid (%f %f) far from init (%f %f)", 
                        coords[0], coords[1], init[0], init[1])
    else:
        logging.debug("coords %f %f", coords[0], coords[1])

    # Set keyword arguments for DAOFIND - trying to find anything other 
    # real sources in the image
    dkwargs = {"fwhm":2.5, "threshold":coadd_bkgd,  
               "sharplo":0.01,"sharphi":5.0}
    sources, n_sources = centroid.daofind_centroid(coadd, None, dkwargs)
    if n_sources==0:
        logging.info("bkgd: %f max: %f", coadd_bkgd, max(coadd.flatten()))
    logging.debug("sources")
    logging.debug(sources)

    # Set the minimum and maximum aperture sizes for the extraction
    # and the step in aperture sizes
    ap_min, ap_max, ap_step = 2, 7, 1
    radii = np.arange(ap_min, ap_max, ap_step)

    # Always use circular apertures, I think ap_type is now defunct actually...
    ap_type = "circ"
    phot.make_circ_lc(pixels, maskmap, times, init, radii,
                 "lcs/{}.csv".format(outfilename), fw_box)

    # get the EPIC ID
    epic = outfilename.split("-")[0][4:]
    logging.info(epic)

    # If an output filename is provided, save the output
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

def run_list(filenames, save_output=False, output_filename="outtable", data_dir=""):
    """ Extract light curves for a list of target pixel files.

    Inputs
    ------
    filenames: list-like of strings
        list of target pixel filenames

    save_output: boolean, default=False
        if True, will save various statistics about sources and extraction
        to a comma-separated file. The output filename will have
        today's date appended.

    """

    # Need to change base_path for tpfs too before running on Yeti
    # although base_path is in run_one, it expects the full path to tpfs
 
    output_f = None
    
    if save_output:
        tfile = "tables/{0}_phot_{1}.csv".format(output_filename, today)
        output_f = open(tfile, "w")
        output_f.write("output_file, EPIC, RA_OBJ, DEC_OBJ, pix_ra, pix_dec")
        output_f.write(", kpmag, n_src, daofind_fwhm, daofind_threshold")
        output_f.write(", daofind_sharplo, daofind_sharphi, centroid_box")
        output_f.write(", ap_type, ap_min, ap_max, ap_step")

    for i, fname in enumerate(filenames):
        fname = data_dir+fname.split("/")[-1]
        if os.path.exists(fname)==True:
            logging.warning(fname)
            run_one(fname, output_f)
        else:
            logging.warning("skipping %s", fname)
#        if i==4:
#            break
    if save_output: 
        output_f.close()

if __name__=="__main__":
    logging.basicConfig(level=logging.DEBUG, 
                        format="%(asctime)s - %(name) - %(message)s")
    
    arrayid = int(os.getenv("PBS_ARRAYID",0))

    mini = (arrayid-1)*100
    maxi = arrayid * 100

    print(arrayid, mini, maxi)

    input_file = "k2_c4_kp9-16_stars_tpfs.csv"
    input_list = at.read(input_file)
    print(input_list.dtype)
    print(input_list["filename"])
    subset = input_list["filename"][mini:maxi]

    outfile = "instrumental_signals_{0}".format(arrayid)

    run_list(subset, True, outfile,
             data_dir="/vega/astro/users/sd2706/k2/data/")

    
    """
    if len(sys.argv)==3:
        epic = sys.argv[1]
        base_path = sys.argv[2]
    elif len(sys.argv)==2:
        epic = sys.argv[1]
        base_path = "/vega/astro/users/sd2706/k2/data/"
    else:
        epic = 210963067
        base_path = "/vega/astro/users/sd2706/k2/data/"

    logging.debug(epic)

    output_f = None

    filename = "{0}ktwo{1}-c04_lpd-targ.fits.gz".format(base_path,epic)

    logging.debug(filename)
    run_one(filename, None, False, None)
    """
