"""Calculate fluxes and produce a lightcurve."""

import logging

import numpy as np
from astropy.wcs import WCS
import photutils

from k2phot import centroid

def calc_bkgd(image, maskmap, ap_center, bkgd_radius, 
              iterations=3, clip_at = 3):
    """Calculate background flux level for the image. 
    """
    
    # Calculate the distance of each pixel from the star's centroid
    image_shape = np.shape(image)
    center_dist = np.zeros_like(image)
    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            sq_sub = [(ap_center[0] - i)**2, 
                      (ap_center[1] - j)**2]
            center_dist[i,j] = np.sqrt(np.sum(sq_sub))
            
    # Mask out the area near the star
    bkgd_mask = np.zeros_like(image)
    bkgd_mask[center_dist>bkgd_radius] = 1
    # Mask out the areas that weren't observed
    bkgd_mask[maskmap==0] = 0
    
    background = image[bkgd_mask==1]
    bkgd_points = background.flatten()
    # Iterativelyi compute the median background level,
    # Removing 3-sigma outliers (or whatever level specified by clip_at) 
    for i in range(iterations):
#        logging.debug("%d %f",i,len(bkgd_points))
        med, stdev = np.median(bkgd_points), np.std(bkgd_points)
        bad_points = np.where(abs(bkgd_points - med) >= (clip_at * stdev))[0]
        bkgd_points = np.delete(bkgd_points, bad_points)
    
    # The background flux is the median of remaining background points
    bkgd_flux = np.median(bkgd_points)
    return bkgd_flux

def get_flux(image, maskmap, ap_center, ap_radii, max_rad=None):
    """
    Calculate flux in a circular aperture at the given set of aperture radii.
    """
    
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

        # The background radius must be greater than the aperture radius 
        # by at least one pixel including diagonals (hence the sqrt(2))
        bkgd_radius = np.ceil(rad + np.sqrt(2))
        # Calculate the background level
        bkgd_fluxes[i] = calc_bkgd(image, maskmap, ap_center, bkgd_radius)
        # Subtract the background level from every pixel in the image
        bkgd_subtracted = image - bkgd_fluxes[i]
        
        # Now do the aperture photometry itself
        aperture = photutils.CircularAperture(ap_center, r=rad)
        phot_table = photutils.aperture_photometry(bkgd_subtracted, aperture)
        ap_fluxes[i] = phot_table["aperture_sum"][0]
        
    return ap_fluxes, bkgd_fluxes

def make_lc(image_list, maskmap, times, start_center, ap_radii,
            output_filename):
    """
    Make a lightcurve by computing aperture photometry for 
    all images in a target pixel file.
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
        coords = centroid.flux_weighted_centroid(image_list[i], 7, 
                                                 init=start_center,
                                                 to_plot=False)

        # Write out the centroid pixel coordinates
        f.write(",{0:.6f},{1:.6f}".format(coords[0],coords[1]))
        
        if (np.any(np.isfinite(coords[:2])==False) or 
            (coords[0]<0) or (coords[1]<0) or 
            (coords[0]>100) or (coords[1]>100)
            ):
            logging.debug(coords[:2])
            f.write(",NaN,NaN"*len(ap_radii)) 
        else:
            # Now run the aperture photometry on the image
            ap_fluxes, bkgd_fluxes = get_flux(image_list[i], maskmap, 
                                              coords[::-1], ap_radii)
        
            # Write out the fluxes and background level for each aperture
            for i, r in enumerate(ap_radii):
                f.write(",{0:.6f},{1:.6f}".format(ap_fluxes[i],
                                                  bkgd_fluxes[i]))
            
    f.close()
