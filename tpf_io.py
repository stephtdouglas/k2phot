"""I/O Operations on target pixel files."""

import logging

from astropy.io import fits

def get_data(filename):
    """Open target pixel file and extract relevant information."""
    hdu = fits.open(filename,mode='readonly',memmap=True)
#    logging.debug(hdu.info())
    # Extension 0 is just the header
    # Extension 1 is the data
    # Extension 2 is the aperture mask

    table = hdu[1].data[:]
    times = table['TIME']
    pixels = table['FLUX']
    maskmap = hdu[2].data
    maskheader = hdu[2].header
    kpmag = hdu[0].header["KEPMAG"]

    hdu.close()

    return table, times, pixels, maskmap, maskheader, kpmag


if __name__=="__main__":
    pass
