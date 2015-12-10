"""I/O Operations on target pixel files."""

import logging

from astropy.io import fits
import numpy as np

def get_data(filename):
    """Open target pixel file and extract relevant information."""
    hdu = fits.open(filename,mode='readonly',memmap=True)
#    logging.debug(hdu.info())
    # Extension 0 is just the header
    # Extension 1 is the data
    # Extension 2 is the aperture mask

    table = hdu[1].data
    # Thruster fires and other issues are flagged in the headers
    bad_frames = np.where(table["QUALITY"]>0)[0]

    # Delete bad time points from times array
    times = np.delete(table['TIME'], bad_frames)
    # C3 and later were background subtracted, but local calculation 
    # will be better than the global background they used
    # (see data release notes for C4)
    if "FLUX_BKG" in table.dtype.names:
        pixels0 = table['FLUX'] + table["FLUX_BKG"]
    else:
        logging.info("No pipeline background found")
        pixels0 = table["FLUX"]

    # Delete NaNs and bad time points
    pixels0[np.isnan(pixels0)] = 0
    pixels = np.delete(pixels0, bad_frames, axis=0)

    # The mask indicates where pixels were actually saved vs.
    # padding to make a rectangular array
    # The header includes target information
    maskmap = hdu[2].data
    maskheader = hdu[2].header
    kpmag = hdu[0].header["KEPMAG"]

    hdu.close()

    return table, times, pixels, maskmap, maskheader, kpmag


if __name__=="__main__":
    pass
