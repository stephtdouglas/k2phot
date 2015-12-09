"""Calculate fluxes and produce a lightcurve."""

import logging

import numpy as np
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
import photutils

from k2phot import centroid

def circ_flux(image, maskmap, ap_center, ap_radii, max_rad=None):
    """
    Calculate flux in a circular aperture at the given set of aperture radii.
    """
    
    mean, median, std = sigma_clipped_stats(image, mask=(maskmap==0),
                                            sigma=3.0, iters=3)
    # Calculate the background level
    bkgd_flux = median

    ap_fluxes = np.zeros(len(ap_radii))*np.nan
    bkgd_fluxes = np.zeros(len(ap_radii))*np.nan

    if max_rad is None:
        max_rad = max(np.shape(maskmap)) / 2.0
    
    # Compute fluxes and background levels for every given radius
    for i, rad in enumerate(ap_radii):
        # If the radius is bigger than half the image, the photometry
        # will fail (because the aperture will fall off the image)
        if rad > max_rad:
            continue

        # Just take bkgd as median of whole image
        # Otherwise no background left with big apertures
        bkgd_fluxes[i] = bkgd_flux
        bkgd_subtracted = image - bkgd_fluxes[i]
        
        # Now do the aperture photometry itself
        aperture = photutils.CircularAperture(ap_center, r=rad)
        phot_table = photutils.aperture_photometry(bkgd_subtracted, aperture)
        ap_fluxes[i] = phot_table["aperture_sum"][0]
        
    return ap_fluxes, bkgd_fluxes


def ellip_flux(image, maskmap, ap_center, ap_radii, a, b, theta, background,
    max_rad=None):
    """
    Calculate flux in an elliptical aperture at the given set of aperture radii.
    """
    
    ap_fluxes = np.zeros(len(ap_radii))*np.nan

    if max_rad is None:
        mshape = np.shape(maskmap)
        max_rad = np.sqrt(mshape[0]**2 + mshape[1]**2)
    
    # Compute fluxes and background levels for every given radius
    for i, rad in enumerate(ap_radii):
        # If the radius is bigger than half the image, the photometry
        # will fail (because the aperture will fall off the image)
        if a*rad > max_rad:
            continue

        bkgd_subtracted = image - background
        
        # Now do the aperture photometry itself
        aperture = photutils.EllipticalAperture(ap_center, rad*a, rad*b, 
                                                theta=theta)
        phot_table = photutils.aperture_photometry(bkgd_subtracted, aperture)
        ap_fluxes[i] = phot_table["aperture_sum"][0]
        
    return ap_fluxes

def make_circ_lc(image_list, maskmap, times, start_center, ap_radii,
                 output_filename, fw_box=9, ap_type="circular", 
                 ellipse_kwargs=None):
    """
    Make a lightcurve by computing aperture photometry for 
    all images in a target pixel file.

    inputs:
    -------

    image_list: array-like
        fluxes at every epoch from TPF

    maskmap: array-like

    times: array-like

    start_center: array-like
        initial pixel coordinates for star

    ap_radii: array-like
        aperture radii (for circular apertures), or
        multiplicative factors (for elliptical apertures)

    output_filename: string, ending in .csv

    fw_box: odd integer, default=9
        box size for flux-weighted centroid

    ap_type: string
        "circular" (default) or "elliptical"

    ellipse_kwargs: dict, required for ap_type=="elliptical"
        a, b, and theta for the elliptical aperture
        (might be better to fit for theta every time?)
    """
    
    # Open the output file and write the header line
    f = open(output_filename,"w")
    f.write("i,t,x,y")
    for r in ap_radii:
        f.write(",flux_{0:.1f},bkgd_{0:.1f}".format(r))
        
    # Do aperture photometry at every step, and save the results
    for i, time in enumerate(times):
        f.write("\n{0},{1:.6f}".format(i,time))
        
        # Find the actual centroid in this image, using start_center as a guess
        # (I should make a flag if the centroid has moved more than a pixel or two)
        #logging.debug(time)
        coords = centroid.flux_weighted_centroid(image_list[i], fw_box,
                                                 init=start_center,
                                                 to_plot=False)

        # Write out the centroid pixel coordinates
        f.write(",{0:.6f},{1:.6f}".format(coords[0],coords[1]))
        
        if (np.any(np.isfinite(coords[:2])==False) or 
            (coords[0]<0) or (coords[1]<0) or 
            (coords[0]>100) or (coords[1]>100)
            ):
            #logging.debug(coords[:2])
            f.write(",NaN,NaN"*len(ap_radii)) 
        else:

            # Now run the aperture photometry on the image
            if ap_type=="circular":
                ap_fluxes, bkgd_fluxes = circ_flux(image_list[i], maskmap, 
                                                   coords[::-1], ap_radii)
            elif ap_type=="elliptical":
                ap_fluxes, bkgd_fluxes = ellip_flux(image_list[i], maskmap,
                                                    coords, ap_radii,
                                                    **ellipse_kwargs)

            # Write out the fluxes and background level for each aperture
            for i, r in enumerate(ap_radii):
                f.write(",{0:.6f},{1:.6f}".format(ap_fluxes[i],
                                                  bkgd_fluxes[i]))
            
    f.close()


def make_ellip_lc(image_list, maskmap, times, start_center, a, b, ap_radii,
                  output_filename):
    """
    Make a lightcurve by computing aperture photometry for 
    all images in a target pixel file.

    inputs:
    -------

    image_list: array-like
        fluxes at every epoch from TPF

    maskmap: array-like

    times: array-like

    start_center: array-like
        initial pixel coordinates for star

    a, b: float
        axes of ellipse

    ap_radii: array-like
        multiplicative factors (for elliptical apertures)

    output_filename: string, ending in .csv
    """

    #rlen = len(ap_radii)    

    # Open the output file and write the header line
    f = open(output_filename,"w")
    f.write("i,t,x,y")
    for r in ap_radii:
        f.write(",flux_{0:.1f},bkgd_{0:.1f}".format(r))
        
    # Do aperture photometry at every step, and save the results
    for i, time in enumerate(times):
        f.write("\n{0},{1:.6f}".format(i,time))
        
        mean, median, std = sigma_clipped_stats(image_list[i], 
                                                mask=(maskmap==0), sigma=3.0, 
                                                iters=3)
        # Calculate the background level
        bkgd_flux = median

        # Subtract the background level from every pixel in the image
        coords, ajunk, bjunk, theta = centroid.find_ellipse(image_list[i],  
                                                            maskmap,
                                                            bkgd_flux)

        # Write out the centroid pixel coordinates
        f.write(",{0:.6f},{1:.6f}".format(coords[0],coords[1]))
        
        if (np.any(np.isfinite(coords[:2])==False) or 
            (coords[0]<0) or (coords[1]<0) or 
            (coords[0]>100) or (coords[1]>100)
            ):
            #logging.debug(coords[:2])
            f.write(",NaN,NaN"*len(ap_radii)) 
        else:

            # Now run the aperture photometry on the image
            ap_fluxes = ellip_flux(image_list[i], maskmap, coords, ap_radii,
                                   a=a, b=b, theta=theta, background=bkgd_flux)

            # Write out the fluxes and background level for each aperture
            for i, r in enumerate(ap_radii):
                f.write(",{0:.6f},{1:.6f}".format(ap_fluxes[i],bkgd_flux))
            
    f.close()
