#!/usr/bin/env python
#
# $Id$
#
# Tom Esposito (tesposito@berkeley.edu) 2019-05-15
#
# Supplemental tools for GPI pol-mode data reduction.
#

import os
import shutil
import warnings
import time
# warnings.simplefilter("ignore", ImportWarning)
# warnings.simplefilter("ignore", RuntimeWarning)
try:
    import ipdb
    pdb = ipdb
except:
    import pdb
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from scipy.stats import binned_statistic
from xml.dom import minidom
from glob import glob



def remove_quadrupole_rstokes(path_fn, dtheta0=0., C0=2., do_fit=True,
                              theta_bounds=(-np.pi, np.pi), rin=30, rout=100,
                              octo=False, scale_by_r=False, save=False, figNum=80,
                              show_region=True, path_fn_stokes=None,
                              quad_scale=None, pos_pole=False):
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
        save: bool or str; str path to specify exact output path of the resulting
            FITS and summary figure; True to write that output to the same directory
            as the input file; False to not save any output.
        figNum: int, number of matplotlib figure.
        show_region: bool, True to outline the region used for the fit.
        path_fn_stokes: str, relative path to stokesdc FITS file if you want to
            subtract instrumental polarization from Q and U as well.
        quad_scale: Choice of radial scaling:
            None: med(I)
            I: I
            U_abs: absolute value of U after subtracting median
            U_div: divided by pole to fit radial profile, then refit
            U_pos: Only keep positive of pole, then fit radial profile and refit pole
        pos_pole: choice of pole function (1: range from 0 to 1, 2: negatives = 0,
            else: -1 to 1)
    
    Outputs:
        If save is True, writes a new FITS file containing the quadrupole-subtracted
        radial Stokes cube, and saves a .png of the best-fit figure.
    """
    from matplotlib.colors import SymLogNorm
    
    def quad(phi, dtheta, pole=2, pos_pole=False):
        if pos_pole == 1:
            return (np.sin(pole*(phi + dtheta))+1)/2
        elif pos_pole == 2:
            sine = np.sin(pole*(phi + dtheta))
            sine[sine < 0] = 0
            return sine
        else:
            return np.sin(pole*(phi + dtheta))
    
    def residuals(pl, im, pole, quad_scaling, phi, theta_bounds):
        # Apply prior penalty to restrict dtheta.
        if not ((pl[0] >= theta_bounds[0]) and (pl[0] <= theta_bounds[1])):
            pen = 1e3
        else:
            pen = 1.
        return pen*np.nansum((im - (10**pl[1]*quad(phi, pl[0], pole, pos_pole)*quad_scaling))**2)

    def UI_residuals(pl, im, pole, quad_scaling, phi, theta_bounds, I, U_radprof):
        # Fits both I image and U radial profile as rad scaling.
        # Apply prior penalty to restrict dtheta.
        if not ((pl[0] >= theta_bounds[0]) and (pl[0] <= theta_bounds[1])):
            pen = 1e3
        else:
            pen = 1.
        base_pole = 10**pl[1]*quad(phi, pl[0], pole, pos_pole)
        mixed_pole = base_pole * pl[2] * I + base_pole * pl[3] * U_radprof
        return pen*np.nansum((im - mixed_pole)**2)
    
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
        Ur[star[0], star[1]] = np.nanmean(Ur[star[0]-1:star[0]+2, star[1]-1:star[1]+2])
    radii = make_radii(Qr, star)
    phi = make_phi(Qr, star)
    
    # Make an azimuthal median map of the I signal for quadrupole scaling map.
    # Also get radial and azimuthal profile of I for plotting
    I_radmedian, rprof_I, az_prof_I = get_ann_stdmap(I, star, radii, phi=phi, r_max=None, mask_edges=False, use_median=True, rprof_out=True)
    #plot_profiles(rprof_I, az_prof_I, 'Total Intensity Profiles')
    
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

    # Get Ur radial and azimuthal profiles and plot
    Ur_med, Ur_rprof, Ur_aprof = get_ann_stdmap(Ur_clipped, star, radii, phi=phi, r_max=None, mask_edges=False, use_median=True, rprof_out=True)
    #plot_profiles(Ur_rprof, Ur_aprof, 'Ur Profiles')
    
    # Mean background in masked Ur.
    Ur_bg_mean = np.nanmean(Ur_clipped*mask_fit)
    
    vmin_Ur = 4*np.percentile(np.nan_to_num(Ur[100:180, 100:180]), 1.) #np.nanmin(Ur)
    vmax_Ur = 4*np.percentile(np.nan_to_num(Ur[100:180, 100:180]), 99.) # np.nanmax(Ur)
    vmin_Qr = 4*np.percentile(np.nan_to_num(Qr[100:180, 100:180]), 1.) #np.nanmin(Ur)
    vmax_Qr = 4*np.percentile(np.nan_to_num(Qr[100:180, 100:180]), 99.) # np.nanmax(Ur)
    linthresh = np.abs(vmin_Ur)/50. # 0.1 #2e-9
    linscale = 1.
    fontSize = plt.rcParams.get('font.size')
    
    # Beginning of NEW Quad Scale options
    # Just I im, rather than median profile of I
    if quad_scale == 'I':
        quad_scaling = I
    # Radial profile of U
    elif quad_scale == 'U_med':
        quad_scaling = Ur_med
    # Try to find radial profile of pole using absolute value of Ur
    elif quad_scale == 'U_abs':
        Ur_absolute = np.absolute(Ur_clipped-Ur_bg_mean)
        plt.imshow(Ur_absolute,vmin=np.nanpercentile(Ur_absolute,5), vmax=np.nanpercentile(Ur_absolute,95))
        plt.colorbar()
        plt.show()
        quad_scaling, rprof = get_ann_stdmap(Ur_absolute, star, radii, r_max=None, mask_edges=False, use_median=True, rprof_out=True)
        plt.imshow(quad_scaling,vmin=np.nanpercentile(quad_scaling,5), vmax=np.nanpercentile(quad_scaling,95))
        plt.colorbar()
        plt.show()
    
    # Perform the fit. (First fit if it being done twice)
    if quad_scale != 'UI_mix':
        p0 = np.array([dtheta0, C0])
    else:
        p0 = np.array([dtheta0, C0, 0.5, 0.5]) # half-half U, I mix as guess
    
    if do_fit:
        from scipy.optimize import fmin
        if quad_scale != 'UI_mix':
            output = fmin(residuals, p0, (mask_fit*Ur_clipped - Ur_bg_mean, pole, quad_scaling, phi, theta_bounds), full_output=1, disp=1)
        else:
            output = fmin(UI_residuals, p0, (mask_fit * Ur_clipped - Ur_bg_mean, pole, quad_scaling, phi, theta_bounds, I, Ur_med),
                          full_output=1, disp=1)
        pf = output[0]
    else:
        pf = p0
    
    # MORE Quad Scale options where pole fitting is done again
    # Find radial profile by dividing the first fit, clipping extremes
    if quad_scale == 'U_div':
        quad_bf = 10**pf[1]*quad(phi, pf[0], pole, pos_pole) 
        quad_bf = quad_bf / np.nanmax(quad_bf)  
        plt.imshow(quad_bf)
        plt.colorbar()
        plt.show()
        quad_Ur_scalefit = Ur_clipped / quad_bf
        quad_Ur_scalefit[quad_Ur_scalefit > np.nanpercentile(quad_Ur_scalefit, 98)] = np.nan # Clip divide by zeros
        quad_Ur_scalefit[quad_Ur_scalefit < 0] = np.nan  # Negatives aren't helpful
        plt.imshow(quad_Ur_scalefit, vmin=np.nanpercentile(quad_Ur_scalefit,5), vmax=np.nanpercentile(quad_Ur_scalefit,95))
        plt.colorbar()
        plt.show()
        quad_scaling, rprof = get_ann_stdmap(quad_Ur_scalefit, star, radii, r_max=None, mask_edges=False, use_median=True, rprof_out=True)
        plt.imshow(quad_scaling)
        plt.colorbar()
        plt.show()
        
        if do_fit:
            from scipy.optimize import fmin
            output = fmin(residuals, p0, (mask_fit * Ur_clipped - Ur_bg_mean, pole, quad_scaling, phi, theta_bounds),
                          full_output=1, disp=1)
            pf = output[0]
        else:
            pf = p0
    # Find radial profile by only keeping positive of previous pole fit, then fit again
    elif quad_scale == 'U_pos':
        quad_bf = 10**pf[1]*quad(phi, pf[0], pole, pos_pole) # Raw pole shape
        quad_bf[quad_bf < np.nanpercentile(quad_bf,90)] = np.nan  # Mask negatives
        quad_Ur_scalefit = Ur_clipped * quad_bf
        plt.imshow(quad_Ur_scalefit, vmin=np.nanpercentile(quad_Ur_scalefit,5), vmax=np.nanpercentile(quad_Ur_scalefit,95))
        plt.colorbar()
        plt.show()
        quad_scaling, rprof = get_ann_stdmap(quad_Ur_scalefit, star, radii, r_max=None, mask_edges=False, use_median=True, rprof_out=True)
        plt.imshow(quad_scaling)
        plt.colorbar()
        plt.show()
        

        if do_fit:
            from scipy.optimize import fmin
            output = fmin(residuals, p0, (mask_fit * Ur_clipped - Ur_bg_mean, pole, quad_scaling, phi, theta_bounds),
                          full_output=1, disp=1)
            pf = output[0]
        else:
            pf = p0

    if quad_scale == 'UI_mix':
        quad_scaling = pf[2]*I + pf[3]*Ur_med
    quad_bf = 10**pf[1]*quad(phi, pf[0], pole, pos_pole)*quad_scaling
    
    # Get and plot the fit profiles to compare with I and Ur profiles
    scaling_med, scaling_rprof = get_ann_stdmap(quad_scaling, star, radii, r_max=None, mask_edges=False, use_median=True, rprof_out=True)
    quad_med, quad_rprof, quad_aprof = get_ann_stdmap(quad_bf, star, radii, phi=phi, r_max=None, mask_edges=False, use_median=True, rprof_out=True)
    #plot_profiles(scaling_rprof, quad_aprof, 'Pole Profiles')
    
    res_bf = Ur_clipped - quad_bf
    red_chi2_Ur = np.nansum((mask_fit*res_bf - Ur_bg_mean)**2)/(np.where(~np.isnan(mask_fit*res_bf))[0].shape[0] - 2)
    print("Reduced Chi2 Ur = %.3e" % red_chi2_Ur)
    quad_bf_rotQr = 10**pf[1]*quad(phi, pf[0] + dtheta_Qr, pole, pos_pole)*quad_scaling
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
    ax1.imshow(res_bf, vmin=vmin_Ur, vmax=vmax_Ur,
               norm=SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin_Ur, vmax=vmax_Ur))
    ax1.text(0.05, 0.95, "Data$-$Quad", fontsize=fontSize-2, transform=ax1.transAxes, verticalalignment='top')
    ax2 = fig0.add_subplot(233)
    ax2.imshow(quad_bf, vmin=vmin_Ur, vmax=vmax_Ur,
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
    if show_region:
        for ax in [ax1, ax2]:
            in_circle = plt.Circle(star, rin, fill=False, facecolor='None', edgecolor='0.8', alpha=0.7, linestyle='--', linewidth=2)
            out_circle = plt.Circle(star, rout, fill=False, facecolor='None', edgecolor='0.8', alpha=0.7, linestyle='--', linewidth=2)
            ax.add_artist(in_circle)
            ax.add_artist(out_circle)
    for ax in [ax0, ax1, ax2, ax3, ax4, ax5]:
        fpm_circle = plt.Circle(star, 8.6, fill=False, facecolor='None', edgecolor='r', alpha=0.5, linestyle='-', linewidth=1)
        ax.add_artist(fpm_circle)
        ax.set_xticks(range(star[1] - 150, star[1] + 151, 100))
        ax.set_yticks(range(star[0] - 150, star[0] + 151, 100))
        ax.set_xticklabels(range(-150, 151, 100))
        ax.set_yticklabels(range(-150, 151, 100))
    for ax in [ax0, ax1, ax2]:
        ax.set_xticklabels("")
    for ax in [ax1, ax2, ax4, ax5]:
        ax.set_yticklabels("")
    # plt.tight_layout()
    plt.draw()
    
    # Optionally write a new FITS cube and figure for the subtracted images.
    if save:
        if save is True:
            output_path = os.path.expanduser(os.path.splitext(path_fn)[0]) + "_quadsub.fits"
        else:
            output_path = save
        new_hdu = hdu
        new_data = data.copy()
        new_data[1] = Qr_sub
        new_data[2] = res_bf
        new_hdu[1].data = new_data.astype('float32')
        if octo:
            new_hdu[1].header.add_history("Subtracted instrumental octopole")
        else:
            new_hdu[1].header.add_history("Subtracted instrumental quadrupole")
        if not quad_scale:
            new_hdu[1].header.add_history("Pole multiplied by median I profile")
        elif quad_scale == 'I':
            new_hdu[1].header.add_history("Pole multiplied by scaled I image")
        elif quad_scale in ['U_abs', 'U_div', 'U_pos']:
            new_hdu[1].header.add_history("Pole multiplied by median U profile")
        new_hdu[1].header['QUADTHET'] = ("%.3f" % np.degrees(pf[0]), "Rotation angle of instr. quadrupole [degrees]")
        new_hdu[1].header['QUADAMP'] = ("%.3e" % 10**pf[1], "Amplitude of instr. quadrupole")

        if not quad_scale:
            scale_name = ''
        else:
            scale_name = quad_scale
        
        try:
            new_hdu.writeto(output_path, overwrite=True)
        except fits.verify.VerifyError:
            new_hdu.writeto(output_path, output_verify='fix+warn', overwrite=True)
        except OSError as ee:
            print(ee)
            print("WARNING: Not saving quadrupole-subtracted FITS file (see error above).")
        
        # pdb.set_trace()
        # if len(path_fn.split('/')) == 1:
        #     fig0.savefig('quadsub_' + fit_fn.split('.fits')[0] + '.png', dpi=300, transparent=True, format='png')
        # else:
        #     fig0.savefig(os.path.split(os.path.expanduser(path_fn))[0] + , dpi=300, transparent=True, format='png')
        #     fig0.savefig(os.path.expanduser("/".join(path_fn.split('/')[:-1]) + '/quadsub_' + fit_fn.split('.fits')[0] + '.png'), dpi=300, transparent=True, format='png')
        fig0.savefig(os.path.splitext(output_path)[0] + '.png', dpi=300, transparent=False, format='png')
    
# TEMP!!! Subtract instrumental pol from Q and U.
    if path_fn_stokes is not None:
        hdu_st = fits.open(os.path.expanduser(path_fn_stokes))
        data_st = hdu_st[1].data
        Q = data_st[1]
        U = data_st[2]
        # Replace the star center pixel value so it is not NaN in Qr.
        if np.isnan(Q[star[0], star[1]]):
            Q[star[0], star[1]] = np.nanmean(Q[star[0]-1:star[0]+2, star[1]-1:star[1]+2])
            U[star[0], star[1]] = np.nanmean(U[star[0]-1:star[0]+2, star[1]-1:star[1]+2])
        
        rin_st = 50 #65
        rout_st = 80 #75
        I_scale_st = I_radmedian/I_radmedian[int(np.median(range(rin_st, rout_st+1)))]
        mask_st = (radii > rin_st) & (radii < rout_st)
        Q_sig_lo = np.percentile(np.nan_to_num(Q[mask_st]), 15.87)
        Q_sig_up = np.percentile(np.nan_to_num(Q[mask_st]), 84.14)
        Q_mask = np.ones(Q.shape)
        Q_mask[mask_st] = 0
        Q_mask[Q < Q_sig_lo] = 1
        Q_mask[Q > Q_sig_up] = 1
        Q_masked = np.ma.masked_array(Q.copy(), mask=Q_mask)
        U_sig_lo = np.percentile(np.nan_to_num(U[mask_st]), 15.87)
        U_sig_up = np.percentile(np.nan_to_num(U[mask_st]), 84.14)
        U_mask = np.ones(U.shape)
        U_mask[mask_st] = 0
        U_mask[U < U_sig_lo] = 1
        U_mask[U > U_sig_up] = 1
        U_masked = np.ma.masked_array(U.copy(), mask=U_mask)
        
        Q_med = np.nanmedian(Q_masked.compressed())
        U_med = np.nanmedian(U_masked.compressed())
        
        def residuals_st(pl, im, st_scaling, phi):
            return np.nansum((im - (10**pl[0]*st_scaling))**2)
        
        p0_st = [-4.]
        if do_fit:
            from scipy.optimize import fmin
            output = fmin(residuals_st, p0_st, (U_masked, quad_scaling, phi), full_output=1, disp=1)
            pf_st = output[0]
        else:
            pf_st = p0_st
        
        Q_sub = Q - Q_med*I_scale_st
        # U_sub = U - U_med*I_scale_st
        U_sub = U - (10**pf_st[0])*quad_scaling
        res_U_bf = U_sub
        
        Qr_medsub, Ur_medsub = get_radial_stokes(Q_sub, U_sub, phi)
        
        
    # TEMP!!! Plot Q and U
        vmin_U = 4*np.percentile(np.nan_to_num(U[100:180, 100:180]), 1.) #np.nanmin(U)
        vmax_U = 4*np.percentile(np.nan_to_num(U[100:180, 100:180]), 99.) # np.nanmax(U)
        vmin_Q = 4*np.percentile(np.nan_to_num(Q[100:180, 100:180]), 1.) #np.nanmin(U)
        vmax_Q = 4*np.percentile(np.nan_to_num(Q[100:180, 100:180]), 99.) # np.nanmax(U)
        linthresh = 0.1 #2e-9
        linscale = 1.
        
        fig1 = plt.figure(figNum+1, figsize=(8, 6))
        fig1.clf()
        fig1.suptitle("%s\n%s: $\\Delta\\theta$=%.3f deg , amp=%.3e" % (fit_fn, obj, np.degrees(pf[0]), 10**pf[1]), fontsize=14)
        ax0 = fig1.add_subplot(231)
        ax0.imshow(U, vmin=vmin_U, vmax=vmax_U,
                   norm=SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin_U, vmax=vmax_U))
        ax0.set_ylabel("U", rotation=0)
        ax0.text(0.05, 0.95, "Data", fontsize=fontSize-2, transform=ax0.transAxes, verticalalignment='top')
        ax1 = fig1.add_subplot(232)
        ax1.imshow(res_U_bf, vmin=vmin_U, vmax=vmax_U,
                   norm=SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin_U, vmax=vmax_U))
        ax1.text(0.05, 0.95, "Data$-$Median", fontsize=fontSize-2, transform=ax1.transAxes, verticalalignment='top')
        ax2 = fig1.add_subplot(233)
        ax2.imshow(U_masked, vmin=vmin_U, vmax=vmax_U,
                   norm=SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin_U, vmax=vmax_U))
        ax2.text(0.05, 0.95, "ROI", fontsize=fontSize-2, transform=ax2.transAxes, verticalalignment='top')
        ax3 = fig1.add_subplot(234)
        ax3.imshow(Q, vmin=vmin_Q, vmax=vmax_Q,
                   norm=SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin_Q, vmax=vmax_Q))
        ax3.set_ylabel("Q", rotation=0)
        ax4 = fig1.add_subplot(235)
        ax4.imshow(Q_sub, vmin=vmin_Q, vmax=vmax_Q,
                   norm=SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin_Q, vmax=vmax_Q))
        ax5 = fig1.add_subplot(236)
        ax5.imshow(Q_masked, vmin=vmin_Q, vmax=vmax_Q,
                   norm=SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin_Q, vmax=vmax_Q))
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
        
        fig2 = plt.figure(figNum+2, figsize=(6., 6))
        fig2.clf()
        # fig2.suptitle("%s\n%s: $\\Delta\\theta$=%.3f deg , amp=%.3e" % (fit_fn, obj, np.degrees(pf[0]), 10**pf[1]), fontsize=14)
        ax0 = fig2.add_subplot(221)
        ax0.imshow(Ur_medsub, vmin=vmin_Ur, vmax=vmax_Ur,
                   norm=SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin_Ur, vmax=vmax_Ur))
        ax0.set_ylabel("U$_\\phi$", rotation=0)
        ax0.text(0.05, 0.95, "Stokes Median sub", fontsize=fontSize-2, transform=ax0.transAxes, verticalalignment='top')
        ax1 = fig2.add_subplot(222)
        ax1.imshow(res_bf, vmin=vmin_Ur, vmax=vmax_Ur,
                   norm=SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin_Ur, vmax=vmax_Ur))
        ax1.text(0.05, 0.95, "$\phi$ Quad sub", fontsize=fontSize-2, transform=ax1.transAxes, verticalalignment='top')
        ax2 = fig2.add_subplot(223)
        ax2.imshow(Qr_medsub, vmin=vmin_Qr, vmax=vmax_Qr,
                   norm=SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin_Qr, vmax=vmax_Qr))
        ax2.set_ylabel("Q$_\\phi$", rotation=0)
        # ax2.text(0.05, 0.95, "$\phi$ sub", fontsize=fontSize-2, transform=ax2.transAxes, verticalalignment='top')
        ax3 = fig2.add_subplot(224)
        ax3.imshow(Qr_sub, vmin=vmin_Qr, vmax=vmax_Qr,
                   norm=SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin_Qr, vmax=vmax_Qr))
        plt.subplots_adjust(bottom=0.07, right=0.98, top=0.87, left=0.15, hspace=0.05, wspace=0.05)
        for ax in [ax0, ax1, ax2, ax3]:
            ax.set_xticks(range(star[1] - 100, star[1] + 101, 100))
            ax.set_yticks(range(star[0] - 100, star[0] + 101, 100))
            ax.set_xticklabels(range(-100, 101, 100))
            ax.set_yticklabels(range(-100, 101, 100))
        for ax in [ax0, ax1]:
            ax.set_xticklabels("")
        for ax in [ax1, ax3]:
            ax.set_yticklabels("")
        # plt.tight_layout()
        plt.draw()
        
        # Plot azimuthal profiles to check instr. pol subtraction performance.
        fig3 = plt.figure(figNum+3)
        fig3.clf()
        ax0 = fig3.add_subplot(111)
        import matplotlib.cm as cm
        from scipy.ndimage.filters import gaussian_filter1d
        cmap = cm.plasma
        sm_sigma = 20
        for ii, rr in enumerate(range(rin, rout+1, 45)):
            cc = cmap(float(ii)/len(range(rin, rout+1, 45)))
            r_cond = (radii > rr-0.5) & (radii < rr+0.5)
            # pdb.set_trace()
            phi_sort = np.argsort(phi[r_cond])
            ax0.plot(gaussian_filter1d(phi[r_cond][phi_sort], sm_sigma), gaussian_filter1d(res_bf[r_cond][phi_sort], sm_sigma), c=cc, linestyle='-', label='r=' + str(rr))
            ax0.plot(gaussian_filter1d(phi[r_cond][phi_sort], sm_sigma), gaussian_filter1d(Ur_medsub[r_cond][phi_sort], sm_sigma), c=cc, linestyle='--')
            ax0.plot(gaussian_filter1d(phi[r_cond][phi_sort], sm_sigma), gaussian_filter1d(Ur_clipped[r_cond][phi_sort], sm_sigma), c=cc, linestyle=':')
            ax0.axhline(y=np.nanmedian(Ur_clipped[r_cond]), color=cc, linestyle='-.')
        ax0.set_xlabel('theta')
        ax0.legend(numpoints=1)
        plt.draw()
        
        
        # Optionally write a new FITS cube and figure for the subtracted images.
        if save:
            new_st_hdu = hdu_st
            new_st_data = data_st.copy()
            new_st_data[1] = Q_sub
            new_st_data[2] = U_sub
            new_st_hdu[1].data = new_st_data.astype('float32')
            new_st_hdu[1].header.add_history("Subtracted instrumental polarization * I radial prof.")
            new_phi_hdu = hdu
            new_phi_data = data.copy()
            new_phi_data[1] = Qr_medsub
            new_phi_data[2] = Ur_medsub
            new_phi_hdu[1].data = new_phi_data.astype('float32')
            new_phi_hdu[1].header.add_history("Subtracted median instr. polarization * I radial prof.")
            new_phi_hdu[1].header['QUADTHET'] = ("None", "Rotation angle of instr. quadrupole [degrees]")
            new_phi_hdu[1].header['QUADAMP'] = ("None", "Amplitude of instr. quadrupole")
    
            try:
                new_st_hdu.writeto(os.path.expanduser(path_fn_stokes.split('.fits')[0] + "_ipsub.fits"))
                new_phi_hdu.writeto(os.path.expanduser(path_fn.split('.fits')[0] + "_med-ipsub.fits"))
            except fits.verify.VerifyError:
                new_st_hdu.writeto(os.path.expanduser(path_fn_stokes.split('.fits')[0] + "_ipsub.fits"), output_verify='fix+warn')
                new_phi_hdu.writeto(os.path.expanduser(path_fn.split('.fits')[0] + "_med-ipsub.fits"), output_verify='fix+warn')
            
            # if len(path_fn.split('/')) == 1:
            #     fig0.savefig('quadsub_' + fit_fn.split('.fits')[0] + '.png', dpi=300, transparent=True, format='png')
            # else:
            #     fig0.savefig(os.path.expanduser("/".join(path_fn.split('/')[:-1]) + '/quadsub_' + fit_fn.split('.fits')[0] + '.png'), dpi=300, transparent=True, format='png')
    
    # pdb.set_trace()
    return


def remove_quadrupole_batch(path_list=None, path_dir=None, dtheta0=0., C0=2., do_fit=True,
                              theta_bounds=(-np.pi, np.pi), rin=30, rout=100,
                              octo=False, scale_by_r=False, save=False, figNum=80, quad_scale=None, pos_pole=False):
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
    
    for ii, path_fn in enumerate(path_list):
        try:
            remove_quadrupole_rstokes(path_fn, dtheta0=dtheta0, C0=C0, do_fit=do_fit,
                                      theta_bounds=theta_bounds, rin=rin, rout=rout, octo=octo, scale_by_r=scale_by_r, 
                                      save=save, figNum=figNum, quad_scale=quad_scale, pos_pole=pos_pole)
        except:
            print("Failed on %s" % path_fn)
    
    return


def remove_quadrupole_podc_group(path_fn, recipe_temp, queue_path, path_list=None, dtheta0=0., C0=2., do_fit=True,
                              theta_bounds=(-np.pi, np.pi), rin=30, rout=100,
                              octo=False, scale_by_r=False, save=False, figNum=80, quad_scale=None, pos_pole=False):
    """
    Performs remove_quadrupole_rstokes on multiple rstokes files constructed from the minimum number of podc files (4), 
    then combines the results
    
    !!! gpi-pipeline must be running for this function to work !!!
    
    Inputs:
        path_fn: str, relative path to folder with podc files.
        recipe_temp: str, relative path to recipe template.
        queue_path: str, relative path to gpi-pipeline queue folder.
        The rest of the inputs are the same as remove_quadrupole_rstokes.
    
    Outputs:
        If save is True, writes a new FITS file containing the final quadrupole-subtracted
        radial Stokes cube. 
    """
    
    recipe_dir = path_fn+'/recipe_tmp/'
    out_dir = path_fn+'/stokes_output/'

    if path_list:
        podc_files = path_list
    else:
        podc_files = sorted(glob(path_fn+'*.fits'))
    
    try:
        os.mkdir(out_dir)
        overwrite = 0
    except:
        overwrite = input('Converted rstokes cubes may already exist. \nEnter 1 to use, 2 to remove and overwrite, or nothing to quit.')
        if not overwrite:
            exit()
    try:
        os.mkdir(recipe_dir)
    except:
        pass

    if overwrite == 2:
        existing_files = glob(out_dir+'*')
        print('The following files will be removed:')
        print(existing_files[i] for i in range(len(existing_files)))
        confirm = input('Confirm remove these files? (y/n)')
        os.remove(existing_files) if confirm == 'y' else exit()

    if overwrite == 0 or overwrite == 2:
        # Get podc groups of 4 to convert to rstokes
        num_g = int(np.floor((len(podc_files)/4)))
        im_groups = [podc_files[i*4:i*4+4] for i in range(num_g)]

        # Recipe editing
        name_base = recipe_temp.strip('/').split('/')[-1].replace('template_recipe','').replace('.xml','')

        for group in im_groups:
            # Load recipe template
            recipe_xml = minidom.parse(recipe_temp)

            # Set input/output directory
            dataset = recipe_xml.getElementsByTagName('dataset')[0]
            dataset.setAttribute('InputDir', path_fn)
            dataset.setAttribute('OutputDir', out_dir)

            # Add file names
            for file in group:
                fits_att = recipe_xml.createElement('fits')
                fname = file.strip('/').split('/')[-1]
                fits_att.setAttribute('FileName', fname)
                recipe_xml.getElementsByTagName('dataset')[0].appendChild(fits_att)

            # Get naming for new template and write
            first_fname = group[0].strip('/').split('/')[-1].replace('.fits','')
            last_fname = group[-1].strip('/').split('/')[-1].replace('.fits','')
            pre_name = first_fname +'-' + last_fname
            xml_out = pre_name+name_base+'.waiting.xml'
            recipe_xml.writexml(open(recipe_dir+xml_out, 'w'))
            recipe_xml.unlink()

            shutil.move(recipe_dir+xml_out, queue_path+xml_out)

        # Allow time for recipes to finish
        time.sleep(num_g*3)

    if overwrite == 1:
        print('Attempting pole subtraction on existing rstokes cubes...')

    # Remove quadrupoles
    remove_quadrupole_batch(path_list=None, path_dir=out_dir, dtheta0=dtheta0, C0=C0, do_fit=do_fit,
                            theta_bounds=theta_bounds, rin=rin, rout=rout, octo=octo, scale_by_r=scale_by_r,
                            save=save, figNum=figNum, quad_scale=quad_scale, pos_pole=pos_pole)

    print('\n\tPole subtraction complete, combining subtracted rstokes cubes.')
    # Gather subtracted rstokes and combine
    sub_rstokes = glob(out_dir+'*rstokesdc_quadsub.fits')

    hdu_master = fits.open(sub_rstokes[0])
    data = hdu_master[1].data

    I_images = np.empty((len(sub_rstokes), data[0].shape[0], data[0].shape[1]))
    Qrsub_images = np.empty((len(sub_rstokes), data[1].shape[0], data[1].shape[1]))
    Ursub_images = np.empty((len(sub_rstokes), data[2].shape[0], data[2].shape[1]))

    for i, file in enumerate(sub_rstokes):
        hdu = fits.open(file)
        I_images[i] = hdu[1].data[0]
        Qrsub_images[i] = hdu[1].data[1]
        Ursub_images[i] = hdu[1].data[2]

    I_mn_comb = np.mean(I_images, axis=0)
    Qr_mn_comb = np.mean(Qrsub_images, axis=0)
    Ur_mn_comb = np.mean(Ursub_images, axis=0)

    new_hdu = hdu_master
    new_data = data.copy()
    new_data[0] = I_mn_comb
    new_data[1] = Qr_mn_comb
    new_data[2] = Ur_mn_comb
    new_hdu[1].data = new_data.astype('float32')
    new_hdu[1].header.add_history("Mean combined all subtracted frames")

    if not quad_scale:
        scale_name = ''
    else:
        scale_name = quad_scale

    try:
        new_hdu.writeto(out_dir+'mean_combined_rstokes_' + scale_name + '_quadsub.fits')
    except:
        confirm = input('Overwite existing combined cube? (y/n)')
        new_hdu.writeto(out_dir + 'mean_combined_rstokes_' + scale_name + '_quadsub.fits', overwrite=True) if confirm == 'y' else exit()

    print('Done.')

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


def get_radial_stokes(Q, U, phi):
    """
    Take normal Stokes parameters Q and U and convert them to
    their radial counterparts Q_phi and U_phi. Math from Schmid et al. 2006
    but adapted so +Q_phi represents vectors perpendicular to a ray from
    the pixel to the star.
    Conversion matches that of GPItv and pipeline primitive.
    
    Inputs:
        Q: Stokes Q image.
        U: Stokes U image.
        phi: polar angle of a given position in Q or U image, calculated as
                np.arctan2(yy - star_y, xx - star_x)
    """
    
    Q_phi = Q*np.cos(2*phi) + U*np.sin(2*phi)
    U_phi = -Q*np.sin(2*phi) + U*np.cos(2*phi)
    
    return Q_phi, U_phi


def plot_profiles(rprof, az_prof, title):
    fig, ax = plt.subplots(2,2, figsize=(8,8))
    fig.suptitle(title, fontsize=18)
    
    ax[0,0].plot(rprof[0], rprof[1])
    ax[0,0].set_title('Radial Profile')
    ax[0,0].set_xlabel('Radius (px)')
    ax[0,0].set_ylabel('Counts')
    
    ax[0,1].plot(az_prof[0][1], az_prof[0][2])
    ax[0,1].set_title('Azimuthal Profile at Radius '+str(az_prof[0][0]))
    ax[0,1].set_xlabel('Azimuth (deg)')
    ax[0,1].set_ylabel('Counts')
    
    ax[1,0].plot(az_prof[1][1], az_prof[1][2])
    ax[1,0].set_title('Azimuthal Profile at Radius '+str(az_prof[1][0]))
    ax[1,0].set_xlabel('Azimuth (deg)')
    ax[1,0].set_ylabel('Counts')
    
    ax[1,1].plot(az_prof[2][1], az_prof[2][2])
    ax[1,1].set_title('Azimuthal Profile at Radius '+str(az_prof[2][0]))
    ax[1,1].set_xlabel('Azimuth (deg)')
    ax[1,1].set_ylabel('Counts')
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    return
            

def get_ann_stdmap(im, cen, radii, phi=None, r_max=None, mask_edges=False,
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
		phi: if phi is not None, then azimuthal profiles will also be generated.      
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

	if phi is not None:
		az_rad = np.linspace(10, r_max-40,4)
		az_prof = []
		for rr in az_rad:
			wr = np.nonzero((radii >= int(rr)-2.5) & (radii < int(rr)+2.5))
			bin_med, bin_edges, bin_num = binned_statistic(np.degrees(phi[wr]), im[wr], statistic='median', bins=36)
			bin_width = (bin_edges[1] - bin_edges[0])
			bin_centers = bin_edges[1:] - bin_width/2
			#az_prof.append([int(rr),np.degrees(phi[wr]),im[wr]])
			az_prof.append([int(rr),bin_centers, bin_med])
	
	
	if rprof_out and phi is None:
		return stdmap, [np.arange(0, r_max, 1), rprof]
	elif rprof_out and phi is not None:
		return stdmap, [np.arange(0, r_max, 1), rprof], az_prof
	else:
		return stdmap
