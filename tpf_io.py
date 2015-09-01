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

    table = hdu[1].data[:]
    times = table['TIME']
    # C4 and later were background subtracted, but local will be
    # better than the global background they used
    if "c04" in filename:
        pixels0 = table['FLUX'] + table["FLUX_BKG"]
    else:
        pixels0 = table["FLUX"]
    pixels0[np.isnan(pixels0)] = 0

    # Thruster fires and other issues are flagged in the headers
    bad_frames = np.where(table["QUALITY"]>0)[0]
    pixels = np.delete(pixels0, bad_frames, axis=0)

    maskmap = hdu[2].data
    maskheader = hdu[2].header
    kpmag = hdu[0].header["KEPMAG"]

    hdu.close()

    return table, times, pixels, maskmap, maskheader, kpmag


if __name__=="__main__":
    pass
