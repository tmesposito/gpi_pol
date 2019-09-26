#!/usr/bin/env python
#
# $Id$
#
# Tom Esposito (tesposito@berkeley.edu) 2019-05-15
#
# Supplemental tools for GPI pol-mode data reduction.
#

import os
import warnings
# warnings.simplefilter("ignore", ImportWarning)
# warnings.simplefilter("ignore", RuntimeWarning)
import pdb
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


def remove_quadrupole_rstokes(path_fn, dtheta0=0., C0=2., do_fit=True,
                              theta_bounds=(-np.pi, np.pi), rin=30, rout=100,
                              octo=False, scale_by_r=False, save=False, figNum=80):
    """
    Subtract instrumental quadrupole signal from GPI radial Stokes cubes.
    
    Inputs:
        path_fn: str, relative path to FITS file being reduced.
        dtheta0: flt, angle by which to rotate quadrupole (+ is CW) [radians].
        C0: flt, log10 amplitude multiplicative scale parameter.
        do_fit: if True, fit for the quadrupole; otherwise, use input dtheta0 and C0.
        theta_bounds: tuple, min and max range of quadrupole angle to allow [radians].
        rin: inner radius of fit region; inside of this the image is masked [pix].
        rout: outer radius of fit region; outside of this the image is masked [pix].
        octo: bool, True to subtract an octopole instead of a quadrupole.
        scale_by_r: bool, True to scale quadrupole by 1/radius instead of total intensity.
        save: bool, True to write the quadrupole-subtracted FITS and a summary figure.
        figNum: int, number of matplotlib figure.
    
    Outputs:
        If save is True, writes a new FITS file containing the quadrupole-subtracted
        radial Stokes cube, and saves a .png of the best-fit figure.
    """
    from matplotlib.colors import SymLogNorm
    
    def quad(phi, dtheta, pole=2):
        return np.sin(pole*(phi + dtheta))
    
    def residuals(pl, im, pole, quad_scaling, phi, theta_bounds):
        # Apply prior penalty to restrict dtheta.
        if not ((pl[0] >= theta_bounds[0]) and (pl[0] <= theta_bounds[1])):
            pen = 1e3
        else:
            pen = 1.
        return pen*np.nansum((im - (10**pl[1]*quad(phi, pl[0], pole)*quad_scaling))**2)
    
    star = np.array([140, 140])
    
    fit_fn = path_fn.split('/')[-1]
    
    hdu = fits.open(os.path.expanduser(path_fn))
    data = hdu[1].data
    
    try:
        obj = hdu[0].header['OBJECT']
    except:
        obj = "Unknown"
    
    I = data[0]
    Qr = data[1]
    Ur = data[2]
    # Replace the star center pixel value so it is not NaN in Qr.
    if np.isnan(Qr[star[0], star[1]]):
        Qr[star[0], star[1]] = np.nanmean(Qr[star[0]-1:star[0]+2, star[1]-1:star[1]+2])
    radii = make_radii(Qr, star)
    phi = make_phi(Qr, star)
    # Make an azimuthal median map of the I signal for quadrupole scaling map.
    I_radmedian = get_ann_stdmap(I, star, radii, r_max=None, mask_edges=False, use_median=True)
    # Set the (primarily radial) scaling of the quadrupole.
    if scale_by_r:
        quad_scaling = 1/radii
    else:
        quad_scaling = I_radmedian
    
    # Quadrupole or octopole?
    if octo:
        pole = 4
        # Angle offset of Qr octopole from Ur octopole.
        dtheta_Qr = np.radians(-22.5) # [radians]
    else:
        pole = 2
        # Angle offset of Qr quadrupole from Ur quadrupole.
        dtheta_Qr = np.radians(-45.) # [radians]
    
    
    mask_fit = np.ones(Ur.shape)
    mask_fit[(radii <= rin) | (radii >= rout)] = np.nan
    
    # Clip extreme outliers in Ur (outside of -3sigma and +4sigma)
    Ur_clipped = Ur.copy()
    Ur_clipped[(Ur <= np.percentile(np.nan_to_num(Ur), 0.003)) | (Ur >= np.percentile(np.nan_to_num(Ur), 99.994))] = np.nan
    # Clip extreme outliers in Qr (outside of -3sigma and +4sigma).
    # This may also clip out real disk signal.
    Qr_clipped = Qr.copy()
    Qr_clipped[(Qr <= np.percentile(np.nan_to_num(Qr), 0.003)) | (Qr >= np.percentile(np.nan_to_num(Ur), 99.994))] = np.nan

    
    # Mean background in masked Ur.
    Ur_bg_mean = np.nanmean(Ur_clipped*mask_fit)
    
    vmin_Ur = 4*np.percentile(np.nan_to_num(Ur[100:180, 100:180]), 1.) #np.nanmin(Ur)
    vmax_Ur = 4*np.percentile(np.nan_to_num(Ur[100:180, 100:180]), 99.) # np.nanmax(Ur)
    vmin_Qr = 4*np.percentile(np.nan_to_num(Qr[100:180, 100:180]), 1.) #np.nanmin(Ur)
    vmax_Qr = 4*np.percentile(np.nan_to_num(Qr[100:180, 100:180]), 99.) # np.nanmax(Ur)
    linthresh = 0.1 #2e-9
    linscale = 1.
    fontSize = plt.rcParams.get('font.size')
    
    # Perform the fit.
    p0 = np.array([dtheta0, C0])
    
    if do_fit:
        from scipy.optimize import fmin
        output = fmin(residuals, p0, (mask_fit*Ur_clipped - Ur_bg_mean, pole, quad_scaling, phi, theta_bounds), full_output=1, disp=1)
        pf = output[0]
    else:
        pf = p0
    
    quad_bf = 10**pf[1]*quad(phi, pf[0], pole)*quad_scaling
    res_bf = Ur_clipped - quad_bf
    red_chi2_Ur = np.nansum((mask_fit*res_bf - Ur_bg_mean)**2)/(np.where(~np.isnan(mask_fit*res_bf))[0].shape[0] - 2)
    print("Reduced Chi2 Ur = %.3e" % red_chi2_Ur)
    quad_bf_rotQr = 10**pf[1]*quad(phi, pf[0] + dtheta_Qr, pole)*quad_scaling
    quad_bf_rotQr[radii <= 8.6] = 0.
    Qr_sub = Qr - quad_bf_rotQr
    Qr_sub_res = Qr_clipped - quad_bf_rotQr
    red_chi2_Qr = np.nansum((mask_fit*Qr_sub_res)**2)/(np.where(~np.isnan(mask_fit*Qr_sub_res))[0].shape[0] - 2)
    print("Reduced Chi2 Qr = %.3e" % red_chi2_Qr)
    
    fig0 = plt.figure(figNum, figsize=(8, 6))
    fig0.clf()
    fig0.suptitle("%s\n%s: $\\Delta\\theta$=%.3f deg , amp=%.3e" % (fit_fn, obj, np.degrees(pf[0]), 10**pf[1]), fontsize=14)
    ax0 = fig0.add_subplot(231)
    ax0.imshow(Ur, vmin=vmin_Ur, vmax=vmax_Ur,
               norm=SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin_Ur, vmax=vmax_Ur))
    ax0.set_ylabel("U$_\\phi$", rotation=0)
    ax0.text(0.05, 0.95, "Data", fontsize=fontSize-2, transform=ax0.transAxes, verticalalignment='top')
    ax1 = fig0.add_subplot(232)
    ax1.imshow(mask_fit*res_bf, vmin=vmin_Ur, vmax=vmax_Ur,
               norm=SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin_Ur, vmax=vmax_Ur))
    ax1.text(0.05, 0.95, "Data$-$Quad", fontsize=fontSize-2, transform=ax1.transAxes, verticalalignment='top')
    ax2 = fig0.add_subplot(233)
    ax2.imshow(mask_fit*quad_bf, vmin=vmin_Ur, vmax=vmax_Ur,
               norm=SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin_Ur, vmax=vmax_Ur))
    ax2.text(0.05, 0.95, "Quad", fontsize=fontSize-2, transform=ax2.transAxes, verticalalignment='top')
    ax3 = fig0.add_subplot(234)
    ax3.imshow(Qr, vmin=vmin_Qr, vmax=vmax_Qr,
               norm=SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin_Qr, vmax=vmax_Qr))
    ax3.set_ylabel("Q$_\\phi$", rotation=0)
    ax4 = fig0.add_subplot(235)
    ax4.imshow(Qr_sub, vmin=vmin_Qr, vmax=vmax_Qr,
               norm=SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin_Qr, vmax=vmax_Qr))
    ax5 = fig0.add_subplot(236)
    ax5.imshow(quad_bf_rotQr, vmin=vmin_Qr, vmax=vmax_Qr,
               norm=SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin_Qr, vmax=vmax_Qr))
    plt.subplots_adjust(bottom=0.07, right=0.98, top=0.87, left=0.11, hspace=0.1, wspace=0.05)
    for ax in [ax0, ax1, ax2, ax3, ax4, ax5]:
        ax.set_xticks(range(star[1] - 100, star[1] + 101, 100))
        ax.set_yticks(range(star[0] - 100, star[0] + 101, 100))
        ax.set_xticklabels(range(-100, 101, 100))
        ax.set_yticklabels(range(-100, 101, 100))
    for ax in [ax0, ax1, ax2]:
        ax.set_xticklabels("")
    for ax in [ax1, ax2, ax4, ax5]:
        ax.set_yticklabels("")
    # plt.tight_layout()
    plt.draw()
    
    # Optionally write a new FITS cube and figure for the subtracted images.
    if save:
        new_hdu = hdu
        new_data = data.copy()
        new_data[1] = Qr_sub
        new_data[2] = res_bf
        new_hdu[1].data = new_data.astype('float32')
        if octo:
            new_hdu[1].header.add_history("Subtracted instrumental octopole * I radial prof.")
        else:
            new_hdu[1].header.add_history("Subtracted instrumental quadrupole * I radial prof.")
        new_hdu[1].header['QUADTHET'] = ("%.3f" % np.degrees(pf[0]), "Rotation angle of instr. quadrupole [degrees]")
        new_hdu[1].header['QUADAMP'] = ("%.3e" % 10**pf[1], "Amplitude of instr. quadrupole")
        
        try:
            new_hdu.writeto(os.path.expanduser(path_fn.split('.fits')[0] + "_quadsub.fits"))
        except fits.verify.VerifyError:
            new_hdu.writeto(os.path.expanduser(path_fn.split('.fits')[0] + "_quadsub.fits"), output_verify='fix+warn')
        
        fig0.savefig(os.path.expanduser("/".join(path_fn.split('/')[:-1]) + '/quadsub_' + fit_fn.split('.fits')[0] + '.png'), dpi=300, transparent=True, format='png')
    
    # pdb.set_trace()
    
    return


def remove_quadrupole_batch(path_list=None, path_dir=None, do_fit=True,
                            dtheta0=0., C0=1., save=False, figNum=80):
    """
    Subtract quadrupole signal from a bunch of rstokes cubes. Just run
    remove_quadrupole_rstokes in a loop.
    """
    from glob import glob
    
    # Search for rstokes FITS in path_dir if no path_list is given.
    if path_list is None:
        paths = np.array(glob(path_dir + '*rstokes*.fits'))
        # Reject any already-quad-subtracted cubes.
        wh_quad = ['quadsub' in ff for ff in paths]
        path_list = paths[~np.array(wh_quad)]
        path_list = np.sort(path_list)
    
    theta_bounds = (0., np.pi)
    
    for ii, path_fn in enumerate(path_list):
        try:
            remove_quadrupole_rstokes(path_fn, dtheta0=dtheta0, C0=C0, do_fit=do_fit,
                              theta_bounds=theta_bounds, save=save, figNum=figNum)
        except:
            print("Failed on %s" % path_fn)
    
    return


def remove_persistence_raw_pol(pref_fn, fl, dark_fn, dark_scale=1., pref_dark_fn=None,
                               C0=0., forcescale=1., do_fit=True,
                               sciref_fn=None, plot=False, save=False):
    """
    Subtract detector persistence from raw pol-mode frames. This only partially
    works and is no replacement for having a clean (persistence-free) dataset.
    If do_fit=True fails to produce good results, try adjusting C0 manually
    with do_fit=False.
    
    Inputs:
        pref_fn: str, path to persistence reference image (like a dark) to use.
        fl: list-like, str filepaths to raw science frames being corrected.
        dark_fn: str, path to dark frame to use for general bias subtraction.
            This dark should match the science frames, ideally.
        dark_scale: multiplicative scale factor to make DARK integration time match
            the science and persistence frames (if needed).
        C0: initial guess for persistence reference scale factor.
        forcescale: forced factor by which to multiply best-fit scale factor.
        do_fit: if False, skip the fit and use the input args only.
        sciref_fn: NOT IMPLEMENTED
            (str, optional science frame to use instead of a persistence reference)
        plot: bool, True to show multiple diagnostic figures.
        save: True to write the persistence-corrected data to new file.
    """
    from scipy.optimize import fmin, minimize
    from scipy.ndimage import gaussian_filter, median_filter, interpolation
    
    def residual_persistence(pl, sci, ref, mask):
        # Linear scaling of reference.
        res = (sci - (10**pl[0])*ref)*mask
        # # Exponential scaling of reference.
        # res = (sci - np.log(pl[0]*np.exp(ref)))*mask
        # res = (sci - (10**pl[0])*interpolation.shift(ref, np.array([pl[1], pl[2]]), order=1, mode='constant', cval=np.nan))*mask
        # res -= np.nanmean(res.copy())
        # chi2 = np.nansum(res**2)
        # chi2_poisson = np.nansum((res/np.sqrt(np.abs(sci)))**2)
        # L1norm = np.nansum(np.abs(res))
        phot_noise = np.sqrt(sci)
        phot_noise[phot_noise == 0.] = np.abs(res[phot_noise == 0.])
        L1norm_poisson = np.nansum(np.abs(res)/phot_noise)
        # std = np.nanstd(res)
        # print pl, chi2, L1norm, std
        # print chi2_poisson, L1norm_poisson
        # print pl, np.log(chi2)
        # plt.figure(6)
        # plt.clf()
        # plt.imshow(mask*res)
        # plt.clim(-5,5)
        # plt.draw()
        # pdb.set_trace()
        # return chi2
        # return chi2_poisson
        # return L1norm
        return L1norm_poisson
        # return std
    
    
    dark_hdu = fits.open(os.path.expanduser(dark_fn))
    dark = dark_hdu[1].data
    dark *= dark_scale
    
    pref_hdu = fits.open(pref_fn)
    if pref_dark_fn is not None:
        pref_dark = fits.getdata(os.path.expanduser(pref_dark_fn))
    else:
        pref_dark = dark
    pref = pref_hdu[1].data
    pref -= pref_dark
    med_pref = np.nanmedian(pref)
    # pref_highpass = pref - gaussian_filter(pref.copy(), 50)
    # # pref_highpass = pref - median_filter(pref, 10)
    # pref_highpass[pref_highpass < 0] = 0.
    
    if sciref_fn is not None:
        sciref_hdu = fits.open(sciref_fn)
        sciref = sciref_hdu[1].data
        sciref -= dark
    
    radii = make_radii(pref, (1046, 1157))
    # Mask edges of frames b/c often extra noisy.
    mask = np.ones(pref.shape)
    # mask[0:800] = np.nan
    # mask[-400:] = np.nan
    # mask[:, 0:600] = np.nan
    # mask[:, -1000:] = np.nan
    mask[radii < 100] = np.nan
    mask[radii > 500] = np.nan
    
    # Actual reference fed into the fitting function.
    ref_input = pref #/med_pref
    # ref_input = pref_highpass
    
    # Use log of scale factor to keep it positive.
    p0 = [C0]
    # p0 = [0., 0., 0.]
    
    plt.figure(2)
    plt.clf()
    plt.imshow(ref_input)
    # plt.imshow(gaussian_filter(pref, 1)/med_pref)
    plt.clim(-5, 5)
    plt.title("Persistence reference")
    plt.draw()
    
    plt.figure(4)
    plt.clf()
    plt.imshow(dark)
    plt.clim(-5, 5)
    plt.title("Dark master")
    plt.draw()
    
    scalefacs = []
    
    for ii, fn in enumerate(fl):
        hdu = fits.open(fn)
        data = hdu[1].data
        data_high = np.percentile(data - dark, 99.95)
        data_dsub = data - dark
        # data_dsub = data - dark - (sciref/(np.percentile(sciref, 99.95)/data_high))
        med_data = np.nanmedian(data_dsub)
        # Prune outliers (beyond -3sigma and +5sigma).
        data_dsub[(data_dsub <= np.percentile(data_dsub, 0.003)) | (data_dsub >= np.percentile(data_dsub, 99.9999))] = med_data
        # data_highpass = data_dsub - gaussian_filter(data_dsub.copy(), 50)
        # data_dsub[data_dsub > 50] = np.nan
        # data_dsub[data_dsub > 80] = np.nan
        # data_dsub -= np.mean(data_dsub.copy())
        
 # TEMP!!! Try masking out the pol spots to focus on spaces in between.
        # test = np.ones(pref.shape)
        # test[data > 5*med_data] = np.nan
        # 
        # plt.figure(8)
        # plt.clf()
        # plt.imshow(mask*test*data_dsub)
        # plt.clim(0, 20)
        # plt.draw()
        
        # pdb.set_trace()
        if plot:
            plt.figure(1)
            plt.clf()
            plt.imshow(data_dsub) # - med_data)
            plt.clim(0, 40)
            plt.title("Raw data")
            plt.draw()
            
            plt.figure(10)
            plt.clf()
            plt.imshow(mask*data_dsub)
            plt.clim(0, 40)
            plt.draw()
        
        if do_fit:
            # Iteratively scale and subtract the dark to minimize residuals in data.
            pf = fmin(residual_persistence, p0, args=(data_dsub, ref_input, mask), disp=False, full_output=True, retall=False)
            # pf = fmin(residual_persistence, p0, args=(data_dsub/med_data, gaussian_filter(pref/med_pref, 1), mask), disp=False, full_output=True, retall=False)
            # pf = fmin(residual_persistence, p0, args=(data_dsub - med_data, pref - med_pref, mask), disp=False, full_output=True, retall=False)
            # pf = fmin(residual_persistence, p0, args=(data_dsub, pref_highpass, mask), disp=False, full_output=True, retall=False)
            scalefac = pf[0][0]
            # shifts = np.array([pf[0][1], pf[0][2]])
        else:
            scalefac = C0
        
        # pf = minimize(residual_persistence, p0, args=(data_dsub, pref_highpass, mask), method='Nelder-Mead') #, full_output=True, retall=False)
        # scalefac = pf.x[0]
        # print(scalefac, pf.fun)
        
        # Best-fit scaled reference.
        ref_bf = (10**(scalefac*forcescale))*ref_input
        # ref_bf = np.log(scalefac*np.exp(pref))
        # Persistence-corrected data.
        corr_data = data - ref_bf
        # corr_data = data - (10**scalefac)*interpolation.shift(pref, shifts, order=1, mode='constant', cval=0.)
        
        res = (data_dsub - ref_bf)*mask
        phot_noise = np.sqrt(data_dsub)
        phot_noise[phot_noise == 0.] = np.abs(res[phot_noise == 0.])
        # res -= np.nanmean(res.copy())
        # chi2 = np.nansum(res**2)
        # chi2_poisson = np.nansum((res/np.sqrt(np.abs(sci)))**2)
        # L1norm = np.nansum(np.abs(res))
        # L1norm_poisson = np.nansum(np.abs(res)/np.sqrt(np.abs(sci)))
        L1norm_poisson = np.nansum(np.abs(res)/phot_noise)
        
        print(scalefac, scalefac*forcescale, "%.4e" % L1norm_poisson)
        
        if plot:
            plt.figure(3)
            plt.clf()
            plt.imshow(corr_data - dark) # - med_data)
            plt.clim(0, 40)
            plt.title("Corrected data")
            plt.draw()
        
        scalefacs.append(scalefac)
        
        if save:
            # Write corrected data to a new cube.
            corr_hdu = hdu
            corr_hdu[1].data = corr_data
            corr_hdu[1].header.add_comment("Persistence corrected with reference image %s scaled by %.5f" % (pref_fn.split('/')[-1], 10**scalefac))
            if fn.split('.')[-1] == 'gz':
                save_path = fn.split('.')[-3] + '_persistcorr.fits.gz'
            else:
                save_path = fn.split('.')[-2] + '_persistcorr.fits'
            
            corr_hdu.writeto(save_path, overwrite=True)
    
    # pdb.set_trace()
    
    return


def remove_skypol_podc(fl, C0=0., do_fit=True, bg_ref_fp=None, plot=False, save=False):
    """
    Subtract sky polarization from podc frames. This function picks the brighter
    slice of the podc (by median) and subtracts a constant from it to minimize the
    difference with the other slice. This assumes the sky polarization only
    contaminates the data by making one polarization state brighter than the other.
    This only partially works and is no replacement for having a clean dataset.
    
    If do_fit=True fails to produce good results, try adjusting C0 manually
    with do_fit=False.
    
    Inputs:
        fl: list-like, str filepaths to podc frames being corrected.
        C0: initial guess for persistence reference scale factor.
        do_fit: if False, skip the fit and use the input args only.
        plot: bool, True to show multiple diagnostic figures.
        save: True to write the sky polarization-corrected data to new file.
    """
    from scipy.optimize import fmin, minimize
    from scipy.ndimage import gaussian_filter, median_filter
    
    def residual_skypol(pl, hi, lo, mask):
        # Residuals ~ brighter slice - constant - fainter slice.
        res = (hi - 10**pl[0]) - lo*mask
        # chi2 = np.nansum(res**2)
        # chi2_poisson = np.nansum((res/np.sqrt(np.abs(lo)))**2)
        phot_noise = np.sqrt(lo)
        phot_noise[phot_noise == 0.] = np.abs(res[phot_noise == 0.])
        L1norm_poisson = np.nansum(np.abs(res)/phot_noise)
        # return chi2
        # return chi2_poisson
        return L1norm_poisson
    
    # If given, get median of background in the mean of the two pol slices.
    if bg_ref_fp is not None:
        bg_ref_hdu = fits.open(os.path.expanduser(bg_ref_fp))
        bg_ref = bg_ref_hdu[1].data
        radii_bg = make_radii(bg_ref[0],
                                np.array([bg_ref_hdu[1].header['PSFCENTY'],
                                          bg_ref_hdu[1].header['PSFCENTX']]))
        bg_ref_med = np.nanmedian(np.nanmean(bg_ref, axis=0)[radii_bg > 80])
    else:
        bg_ref_med = np.nan
    
    # Use log of scale factor to keep it positive.
    p0 = [C0]
    
    scalefacs = []
    
    for ii, fn in enumerate(fl):
        hdu = fits.open(os.path.expanduser(fn))
        d0 = hdu[1].data[0] # zeroth pol state
        d1 = hdu[1].data[1] # first pol state
        star = np.array([hdu[1].header['PSFCENTY'], hdu[1].header['PSFCENTX']])
        med_d0 = np.nanmedian(d0)
        med_d1 = np.nanmedian(d1)
        # Prune outliers (beyond -3sigma and +5sigma).
        d0[(d0 <= np.percentile(d0, 0.003)) | (d0 >= np.percentile(d0, 99.9999))] = med_d0
        d1[(d1 <= np.percentile(d1, 0.003)) | (d1 >= np.percentile(d1, 99.9999))] = med_d1
        
        # Pick out the slice with the higher median value.
        # Assume that's where most of the sky polarization has shown up.
        if med_d0 > med_d1:
            hi = d0
            lo = d1
            ind_hi = 0
            ind_lo = 1
        else:
            hi = d1
            lo = d0
            ind_hi = 1
            ind_lo = 0
        
        data_high = np.percentile(hi, 99.95)
        
        radii = make_radii(d0, star)
        # Mask edges of frames b/c often extra noisy.
        # mask = np.ones(hi.shape)
        mask = gaussian_filter(hi, 2)
        mask[~np.isnan(mask)] = 1.
        mask[radii < 20] = np.nan
        # mask[radii > 90] = np.nan
        
        # What's the median of the background in the fainter image?
        lo_med = np.nanmedian(lo[radii > 80])
        # Subtract off excess sky background compared to reference image, if given.
        if not np.isnan(bg_ref_med):
            lo -= lo_med - bg_ref_med
        
        if plot:
            plt.figure(1)
            plt.clf()
            plt.imshow(hi - lo)
            plt.clim(-100, 100)
            plt.title("Original Difference: hi - lo")
            plt.draw()
        
        if do_fit:
            # Subtract a global scalar from the brighter slice.
            pf = fmin(residual_skypol, p0, args=(hi, lo, mask), disp=False, full_output=True, retall=False)
            scalefac = pf[0][0]
        else:
            scalefac = C0
        
        # Sky polarization-corrected data.
        corr_hi = hi - 10**scalefac
        
        # What's the median of the background in the corrected bright image?
        corr_hi_med = np.nanmedian(corr_hi[radii > 80])
        # Subtract off excess sky background compared to reference image, if given.
        if not np.isnan(bg_ref_med):
            corr_hi -= corr_hi_med - bg_ref_med
        
        res = corr_hi - lo
        phot_noise = np.sqrt(lo)
        phot_noise[phot_noise == 0.] = np.abs(res[phot_noise == 0.])
        # chi2 = np.nansum(res**2)
        # chi2_poisson = np.nansum((res/np.sqrt(np.abs(lo)))**2)
        # L1norm = np.nansum(np.abs(res))
        L1norm_poisson = np.nansum(np.abs(res*mask)/phot_noise)
        
        print(scalefac, "%.4e" % L1norm_poisson)
        
        if plot:
            plt.figure(3)
            plt.clf()
            plt.imshow(corr_hi)
            plt.clim(0, data_high)
            plt.title("Corrected brighter slice")
            plt.draw()
            
            plt.figure(4)
            plt.clf()
            plt.imshow(res)
            plt.clim(-100, 100)
            plt.title("Residuals: Corrected brighter slice - fainter slice")
            plt.draw()
            
# # TESTING!!! Plot the average of the two slices after highpass filtering.
#             avg = np.nanmean([corr_hi, lo], axis=0)
#             avg[np.isnan(avg)] = np.nanmean(avg[radii >= 120])
#             # hp_avg = avg - gaussian_filter(avg, 30)
#             hp_avg = avg - median_filter(avg, 50)
#             plt.figure(2)
#             plt.clf()
#             plt.imshow(hp_avg, vmin=-200, vmax=200)
#             plt.draw()
        
        scalefacs.append(scalefac)
        
        if save:
            # Write corrected data to a new cube.
            corr_hdu = hdu
            corr_hdu[1].data[ind_hi] = corr_hi
            corr_hdu[1].data[ind_lo] = lo
            corr_hdu[1].header.add_comment("Sky Pol corrected by scaling slice %d by %.5f" % (ind_hi, 10**scalefac))
            if not np.isnan(bg_ref_med):
                corr_hdu[1].header.add_comment("Background subtracted to match %s" % (bg_ref_fp.split('/')[-1]))
            else:
                corr_hdu[1].header.add_comment("No background reference used.")
            save_path = fn.split('.')[-2] + '_skypolcorr.fits'
            
            corr_hdu.writeto(os.path.expanduser(save_path), overwrite=True)
    
    # pdb.set_trace()
    
    return


def make_radii(arr, cen):
    """
    Make array of radial distances from centerpoint cen in 2-D array arr.
    """
    # Make array of indices matching arr indices.
    grid = np.indices(arr.shape, dtype=float)
    # Shift indices so that origin is located at cen coordinates.
    grid[0] -= cen[0]
    grid[1] -= cen[1]
    
    return np.sqrt((grid[0])**2 + (grid[1])**2) # [pix]


def make_phi(arr, cen):
    """
    Make array of phi angles [radians] from phi=0 at +x axis increasing CCW to 2pi,
    around centerpoint cen in 2-D array arr.
    """
    # Make array of indices matching arr indices.
    grid = np.indices(arr.shape, dtype=float)
    # Shift indices so that origin is located at cen coordinates.
    grid[0] -= cen[0]
    grid[1] -= cen[1]
    
    phi = np.arctan2(grid[0], grid[1]) # [rad]
    # Shift phi=0 to lie along +x axis.
    phi[np.where(phi < 0)] += 2*np.pi # [rad]

    return phi


def get_ann_stdmap(im, cen, radii, r_max=None, mask_edges=False,
				   use_mean=False, use_median=False, rprof_out=False):
	"""
	Get standard deviation map from image im, measured in concentric annuli 
	around cen. NaN's in im will be ignored (use for masking).
	
	Inputs:
		im: image from which to calculate standard deviation map.
		radii: array of radial distances for pixels in im from cen.
		r_max: maximum radius to measure std dev.
		mask_edges: False or int N; mask out N pixels on all edges to mitigate edge effects
			biasing the standard devation at large radii.
		use_mean: if True, use mean instead of standard deviation.
		use_median: if True, use median instead of standard deviation.
		rprof_out: if True, also output the radial profile.
	"""
	
	if r_max==None:
		r_max = radii.max()
	
	if mask_edges:
		from scipy.ndimage import gaussian_filter #median_filter
		cen = np.array(im.shape)/2
		mask = np.ma.masked_invalid(gaussian_filter(im, mask_edges)).mask
		mask[cen[0]-mask_edges*5:cen[0]+mask_edges*5, cen[1]-mask_edges*5:cen[1]+mask_edges*5] = False
		im = np.ma.masked_array(im, mask=mask).filled(np.nan)
	
	stdmap = np.zeros(im.shape, dtype=float)
	rprof = []
	for rr in np.arange(0, r_max, 1):
		# Central pixel often has std=0 b/c single element. Avoid this by giving
		# it same std dev as 2 pix-wide annulus from r=0 to 2.
		if rr==0:
			wr = np.nonzero((radii >= 0) & (radii < 2))
			if use_mean:
				val = np.nanmean(im[wr])
				stdmap[cen[0], cen[1]] = val
			elif use_median:
				val = np.nanmedian(im[wr])
				stdmap[cen[0], cen[1]] = val
			else:
				val = np.nanstd(im[wr])
				stdmap[cen[0], cen[1]] = val
		else:
			wr = np.nonzero((radii >= rr-0.5) & (radii < rr+0.5))
			#stdmap[wr] = np.std(im[wr])
			if use_mean:
				val = np.nanmean(im[wr])
				stdmap[wr] = val
			elif use_median:
				val = np.nanmedian(im[wr])
				stdmap[wr] = val
			else:
				val = np.nanstd(im[wr])
				stdmap[wr] = val
		rprof.append(val)
	
	if rprof_out:
		return stdmap, [np.arange(0, r_max, 1), rprof]
	else:
		return stdmap