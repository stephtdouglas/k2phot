
import logging, os
from datetime import date

import astropy.io.ascii as at
import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import sigma_clipped_stats

from k2phot import centroid
from k2phot import tpf_io
from k2phot import phot
from k2phot import plot

alphas = "BCDEFGHIJKLMN"
today = date.today().isoformat()

def plot_one(filename, ap=None, fw_box=None):

    outfilename = filename.split("/")[-1][:-14]
    logging.info(outfilename)
    table, times, pixels, maskmap, maskheader, kpmag = tpf_io.get_data(filename)

    init = centroid.init_pos(maskheader)
    logging.info("init %f %f", init[0], init[1])

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

    ap_min, ap_max, ap_step = 2, 7, 0.5
    radii = np.arange(ap_min, ap_max, ap_step)


    epic = outfilename.split("-")[0][4:]
    logging.info(epic)
#    plot.lcs("lcs/{}.csv".format(outfilename), epic=epic)

    plot.plot_four(epic, filename, coadd, maskmap, maskheader, init, coords, 
                   sources, ap=ap, campaign=4)
    plt.savefig("plot_outputs/ktwo{}-c04_fourby.png".format(epic))


#    plot.plot_xy("lcs/{}.csv".format(outfilename), epic=epic)



def plot_list(listname, resname):

    tpfs = at.read(listname)
    res = at.read(resname)

    res_filenames = res["filename"].data
    f = open("/home/stephanie/my_papers/hyadesk2/figure_sets/f5.tbl","w")

    count = 1
    for i, fname in enumerate(tpfs["filename"]):
        epic = fname.split("/")[-1].split("_")[0].split("-")[0][4:]
        resi = np.where(np.int64(epic)==res["EPIC"])[0]
        res_fname = res_filenames[resi]
        if len(resi)>0:
            res_fname = res_filenames[resi][0]
        if ((os.path.exists(fname)==True) and 
            (os.path.exists(res_fname)==True)):
            logging.warning(fname)
            plot_one(fname, res["ap"][resi], tpfs["fw_box"][i])

            # Just use lc from 5x5 centroiding box for 210736105
            # LC extraction for 210963067 is shown in Figure 7 (fig:neighbor)
            # but I guess I'll show the diagnostic plot here for completness
            # So skip the daofind results
            # 210675409 is actually too bright but still in my list somehow
            # note I never ran 211037886
            if ((epic=="2107361051") or (epic=="2107361050") or 
                (epic=="2109630671") or (epic=="2109630670") or 
            (epic==210675409)):
                continue
            else:
                save_epic = epic
            figsetname = "f5_{0}.eps".format(count)
            f.write("{0} & EPIC {1}\n".format(figsetname,save_epic))
            figsetname = "/home/stephanie/my_papers/hyadesk2/figure_sets/"+figsetname
            plt.savefig(figsetname,bbox_inches="tight")
            plt.close("all")
            count += 1

        else:
            logging.warning("skipping %s", fname)
        
    f.close()


if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)

    output_f = None

#    filename = "/home/stephanie/Dropbox/c4_tpf/ktwo211041649-c04_lpd-targ.fits"
#    run_one(filename, None, False, None)

    plot_list("c4_tpfs_box.csv", "/home/stephanie/code/python/k2spin/tables/c4_lcs_aps_results_2015-12-18_comments.csv")

    
