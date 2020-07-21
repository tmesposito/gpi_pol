#!/usr/bin/env python
#
# $Id$
#
# Tom Esposito (tesposito@berkeley.edu) 2020-05-05
#
# Supplemental tools for GPI pol-mode data reduction.
#

import os
import warnings
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
from scipy.ndimage import filters


def measure_azimuthal_profile(data, star, add_noise=False, model=False):

	radii_data = make_radii(data, star)
	
	# Mask region outside data's PSF subtraction zone and inside IWA.
	r_mask = 9
	data[radii_data >= 135] = np.nan
	data[radii_data < r_mask] = np.nan
	
	# # Scale data brightness by sb_scale.
	# data *= sb_scale
	
# OFF! Model flux calibration.
	# data *= conv_WtoJy*sb_scale*data # [mJy arcsec^-2]

	# Add Gaussian noise to image if desired.
	if add_noise:
		data += np.random.normal(scale=0.1, size=data.shape)
	
	# Convolve model with Gaussian simulating instrument PSF.
	if model:
		data = filters.gaussian_filter(data.copy(), 1.2) #1.31)
	
	fontSize = 16
	
	# Clean up the edges of the data, which often have artifacts.
	# Do this by smoothing the data to expand the NaN's already present outside
	# the edges, ignoring the inner mask, and creating a new mask from this.
	mask_edges = 2.
	if mask_edges:
		mask = np.ma.masked_invalid(filters.gaussian_filter(data, mask_edges)).mask
		mask[star[0]-r_mask*4:star[0]+r_mask*4, star[1]-r_mask*4:star[1]+r_mask*4] = False
		data = np.ma.masked_array(data, mask=mask).filled(np.nan)
	
	# Brightness cuts.
	cut_colors = ['C4', 'C9', 'C2', 'C8', 'C4', 'C9', 'C2', 'C8']
	bw = 3 # width of slice [pix]
	fs = 6 # filter size [pix]
	data_rot45 = rot_array(data, star, -45*np.pi/180) # data rotated -45 deg
	sl_pa0 = np.sum(data[:,star[1]-(bw//2):star[1]+(bw//2)+1], axis=1)[::-1] # data PA=0 slice
	sl_pa45 = np.sum(data_rot45[:,star[1]-(bw//2):star[1]+(bw//2)+1], axis=1)[::-1] # data PA=45 slice
	sl_pa90 = np.sum(data[star[0]-(bw//2):star[1]+(bw//2)+1], axis=0) # data PA=90 slice
	sl_pa135 = np.sum(data_rot45[star[0]-(bw//2):star[1]+(bw//2)+1], axis=0) # data PA=135 slice
	
	sl_list = np.array([sl_pa0, sl_pa45, sl_pa90, sl_pa135])
	pa_sls = [0, 45, 90, 135]
	
	# Estimate 'noise' as median of all slices.
	sl_noise = np.nanmedian(sl_list, axis=0)
	sl_rms = np.sqrt(np.nanmedian(sl_list**2, axis=0)) # 'root median square'
	sl_med = np.sqrt(np.nanmedian(sl_list**2))
	# Find S/N ratio.
	# sl_snr = sl_list/filters.median_filter(sl_rms, fs)
	sl_snr = np.array([filters.median_filter(sl_list[ii], fs) for ii in range(len(sl_list))])/sl_med
	
	# Plot the image with lines marking PA angles measured.
	fig0 = plt.figure(3)
	plt.clf()
	plt.subplot(111)
	plt.subplots_adjust(bottom=0.1, right=0.99, top=0.91, left=0.05)
	im = plt.imshow(data, extent=[-140, 140, -140, 140])
	plt.clim(np.percentile(np.nan_to_num(data), 1.5),
			 np.percentile(np.nan_to_num(data), 99.99))
	# Plot lines tracing PA=0, 45, 90, 135.
	ext = im.get_extent() # [-x, +x, -y, +y]
	plt.plot([0, 0], [ext[2], ext[3]], 'r--', alpha=0.5, linewidth=1)
	plt.plot([ext[3], ext[2]], [ext[0], ext[1]], 'r--', alpha=0.5, linewidth=1)
	plt.plot([ext[2], ext[3]], [0, 0], 'r--', alpha=0.5, linewidth=1)
	plt.plot([ext[1], ext[0]], [ext[3], ext[2]], 'r--', alpha=0.5, linewidth=1)
	plt.text(5, ext[3] - 15, '-0', fontsize=12)
	plt.text(ext[0] + 15, ext[3] - 15, '-45', fontsize=12)
	plt.text(ext[0] + 5, 5, '-90', fontsize=12)
	plt.text(ext[0] + 15, ext[2] + 5, '-135', fontsize=12)
	plt.title(fn + "\nPA = 0 is north up", fontsize=10)
	plt.draw()
	
	# Plot surface brightness v. projected separation.
	fig1 = plt.figure(5)
	plt.clf()
	plt.subplot(111)
	plt.subplots_adjust(bottom=0.13, right=0.98, top=0.91, left=0.15)
	plt.axvspan(xmin=-r_mask, xmax=r_mask, color='k', alpha=0.1)
	plt.axvline(x=0, c='0.4', linestyle='--', linewidth=1.)
	plt.axhline(y=0, c='0.4', linestyle='--', linewidth=1.)
	label_sls = 4*['data'] #4*['mod'] + 4*['data']
	for ii, sl in enumerate(sl_list):
		plt.plot(range(0,data.shape[0])-star[0], sl, label='%s %d' % (label_sls[ii], pa_sls[ii]), c=cut_colors[ii], marker='.', linestyle='None', alpha=0.5)
	plt.plot(range(0,data.shape[0])-star[0], filters.median_filter(sl_noise, fs), label='med', c='k', linestyle=':', linewidth=2.)
	plt.plot(range(0,data.shape[0])-star[0], sl_rms, label='rms', c='m', linestyle='-', linewidth=1.5)
	plt.ylabel("Surface Brightness")
	plt.xlabel("Projected Separation (px)")
	plt.title('%s\nsl %d px, filt %d px' % (fn, bw, fs), fontsize=10)
	plt.legend(ncol=2, fontsize=fontSize-4, labelspacing=0.2, columnspacing=1, borderpad=0.4, handletextpad=0.3, handlelength=1, framealpha=0.5)
	plt.draw()
	
	# Plot S/N-like things.
	fig2 = plt.figure(6)
	plt.clf()
	plt.subplot(111)
	plt.subplots_adjust(bottom=0.13, right=0.98, top=0.91, left=0.15)
	plt.axvspan(xmin=-r_mask, xmax=r_mask, color='k', alpha=0.1)
	plt.axvline(x=0, c='0.4', linestyle='--', linewidth=1.)
	plt.axhline(y=0, c='0.4', linestyle='--', linewidth=1.)
	# for ii, sl in enumerate(sl_list[4:]):
	#     plt.plot(range(0,data_Qr.shape[0])-star[0], sl_dQr_snr[ii], label='%d' % pa_sls[ii], c=cut_colors[ii], linestyle='-', linewidth=1.5)
	for ii, sl in enumerate(sl_list):
		plt.plot(range(0,data.shape[0])-star[0], sl_snr[ii], label='%d' % pa_sls[ii], c=cut_colors[ii], linestyle='-', linewidth=1.5)
	plt.axhline(y=3, c='0.2', linestyle='--', linewidth=1.5, label='"3$\sigma$"', zorder=0)
	plt.ylim(-5, 20)
	plt.ylabel("Pseudo S/N")
	plt.xlabel("Projected Separation (px)")
	plt.title('%s\nsl %d px, filt %d px' % (fn, bw, fs), fontsize=10)
	plt.legend(ncol=2, fontsize=fontSize-4, labelspacing=0.2, columnspacing=1, borderpad=0.4, handletextpad=0.3, handlelength=1, framealpha=0.5)
	plt.draw()
	
	# Plot folded S/N-like things.
	fig3 = plt.figure(7)
	plt.clf()
	plt.subplot(111)
	plt.subplots_adjust(bottom=0.13, right=0.98, top=0.91, left=0.15)
	plt.axvspan(xmin=0, xmax=r_mask, color='k', alpha=0.1)
	plt.axvline(x=0, c='0.4', linestyle='--', linewidth=1.)
	plt.axhline(y=0, c='0.4', linestyle='--', linewidth=1.)
	# for ii, sl in enumerate(sl_list[4:]):
	#     plt.plot(range(0,data_Qr.shape[0])-star[0], sl_dQr_snr[ii], label='%d' % pa_sls[ii], c=cut_colors[ii], linestyle='-', linewidth=1.5)
	folded_snr = np.array([sl_snr[ii][0:sl_snr[ii].shape[0]//2][::-1] + sl_snr[ii][1+sl_snr[ii].shape[0]//2:] for ii in range(len(sl_list))])
	folded_mean = np.nanmean(folded_snr, axis=0)
	for ii, sl in enumerate(sl_list[:4]):
		plt.plot(range(0,data.shape[0]//2), folded_snr[ii], label='%d' % pa_sls[ii], c=cut_colors[ii], linestyle='-', linewidth=1.5)
	plt.plot(range(0,data.shape[0]//2), folded_mean, label='Mean', c='k', linestyle='-', linewidth=2.5)
	plt.axhline(y=3, c='0.2', linestyle='--', linewidth=1.5, label='"3$\sigma$"', zorder=0)
	plt.ylim(-5, 20)
	plt.ylabel("Folded Pseudo S/N")
	plt.xlabel("Projected Separation (px)")
	plt.title('%s\nsl %d px, filt %d px' % (fn, bw, fs), fontsize=10)
	plt.legend(ncol=2, fontsize=fontSize-4, labelspacing=0.2, columnspacing=1, borderpad=0.4, handletextpad=0.3, handlelength=1, framealpha=0.5)
	plt.draw()
	
	
	# Compile list of hits.
	snr_hits = np.where(sl_snr >= 3)
	folded_snr_hits = np.where(folded_snr >= 3)
	folded_mean_hits = np.where(folded_mean >= 3)
	
	N_snr_hits = snr_hits[0].shape[0]
	N_folded_snr_hits = folded_snr_hits[0].shape[0]
	N_folded_mean_hits = folded_mean_hits[0].shape[0]
	
	print("\nS/N > 3: %d (%d%%)" % (N_snr_hits, 100*N_snr_hits/sl_snr.size))
	print("Folded S/N > 3: %d (%d%%)" % (N_folded_snr_hits, 100*N_folded_snr_hits/folded_snr.size))
	print("Folded Mean S/N > 3: %d (%d%%)" % (N_folded_mean_hits, 100*N_folded_mean_hits/folded_mean.size))
	print("Max slice S/N: %.1f" % np.nanmax(sl_snr))
	
	# Plot flux contours in 3D.
	# data3d = data.copy()
	# Smooth slightly and trim negative values for better contouring.
	data3d = filters.gaussian_filter(np.nan_to_num(data.copy()), 2.)
	data3d[data3d < 0.] = np.nan
	fig3d = plt.figure(8)
	plt.clf()
	ax = fig3d.add_subplot(111, projection='3d')
	yy, xx = np.mgrid[:data3d.shape[0], :data3d.shape[1]]
	yy -= star[0]
	xx -= star[1]
	ax.plot([0, 0], [0, 0], [0, np.nanmax(data3d)], 'k--')
	nlevels = 20
	c3d = ax.contour(xx, yy, data3d, levels=nlevels)
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Flux')
	ax.set_title('%s\n%d contours' % (fn, nlevels), fontsize=10, pad=40)
	plt.draw()
	
	# fig3 = plt.figure(33)
	# ax1 = plt.subplot(111)
	# # ax1.imshow((radii**2)*conv_WtoJy*I_scale*mod_hdu[0].data[0,0,0]/1000., vmin=vmin, vmax=vmax, cmap=colmap_devs) #viridis)
	# # ax1.imshow(np.log10(conv_WtoJy*I_scale*mod_hdu[0].data[0,0,0]), vmin=vmin*2., vmax=vmax*2., cmap=colmap_devs) #viridis)
	# from matplotlib.colors import SymLogNorm
	# ax1.imshow(conv_WtoJy*I_scale*mod_hdu[0].data[0,0,i_inc], vmin=0., vmax=vmax*10., cmap=colmap,
	#            norm=SymLogNorm(linthresh=0.05, linscale=1.0, vmin=0., vmax=vmax*10.))
	# ax1.set_xlim(star[1]-wdw[1], star[1]+wdw[1])
	# ax1.set_ylim(star[0]-wdw[0], star[0]+wdw[0])
	# fpm_circle = plt.Circle(star[::-1], 8.68, color='k', fill=False)
	# ax1.add_artist(fpm_circle)
	# # plt.title('%.2f x Raw Model * r^2' % I_scale)
	# plt.title('%.2f x Raw Model I' % I_scale)
	# ax1.tick_params(which='both', direction='in', color=ctick)
	# ax1.yaxis.set_ticks(np.arange(90, 191, 50))
	# ax1.xaxis.set_ticks(np.arange(90, 191, 50))
	# # ax1.yaxis.set_minor_locator(minor_locator)
	# # ax1.xaxis.set_minor_locator(minor_locator)
	# ax1.grid(axis='both', which='both', c='0.6', alpha=0.5)
	# 
	# plt.draw()
	
	# if save:
	# 	fig0.savefig(os.path.expanduser('~/Research/data/gpi/auto_disk/%s_autodisk_imagecuts' % fn) + '.png', dpi=200, transparent=False, format='png')
	# 	fig2.savefig(os.path.expanduser('~/Research/data/gpi/auto_disk/%s_autodisk_slsnr' % fn) + '.png', dpi=200, transparent=False, format='png')
	# 	fig3.savefig(os.path.expanduser('~/Research/data/gpi/auto_disk/%s_autodisk_foldedsnr' % fn) + '.png', dpi=200, transparent=False, format='png')
	# 
	# # Compute the disk brightness; feed in units of ADU/s.
	# # Assume data are already in ADU/coadd (default GPI pipeline output).
	# itime = data_hdu[hdu_ind].header['ITIME'] # [s] integration time per coadd
	# auto_disk_brightness(data/itime, data_Ur/itime, star)
	
	# pdb.set_trace()
	
	if output:
		return [np.nanmax(sl_snr), N_snr_hits, 100*N_snr_hits/sl_snr.size, N_folded_snr_hits, 100*N_folded_snr_hits/folded_snr.size, N_folded_mean_hits, 100*N_folded_mean_hits/folded_mean.size]
	else:
		return


def azimuthal_disk_brightness(data, noise_map, star, r_list=None, yrange=None):
	"""
	Measure and plot image brightness as a function of azimuthal angle in
	radial annuli of consistent width (set by dr parameter below).
	
	data: image array from which to measure brightness.
	noise_map: image array from which to measure noise.
	star: [y,x] array or list of pixel on which annuli are centered.
	r_list: list of radius on which to center annuli.
	yrange: list of ymin and ymax to enforce for plots.
	
	"""
	from ttools import make_radii, make_phi, get_ann_stdmap
	from astropy.convolution import Gaussian1DKernel, Box1DKernel, convolve

	radii = make_radii(data, star)
	# Compute PA angles with 0 at +y (assuming things are north up).
	phi = make_phi(data, star) - np.pi/2
	phi[phi < 0] += 2*np.pi

# # TEMP!!!
#    data = filters.gaussian_filter(data.copy(), 2)
#    noise_map = filters.gaussian_filter(noise_map.copy(), 2)

	noise_std = get_ann_stdmap(noise_map, star, radii,
					r_max=145, mask_edges=2)

	snr_map = data/noise_std

	# Annulus radial width.
	dr = 2 # [pix]

	annuli_flux = []
	annuli_snr = []
	annuli_phi = []
	annuli_r = []
	if r_list is None:
		r_list = np.arange(10, 100, dr)

	# For smoothing
	kernel = Gaussian1DKernel(stddev=1)

	for rr in r_list:
		r_cond = (radii >= rr) & (radii < rr+dr)
		sortinds = np.argsort(phi[r_cond])
		# annuli_flux.append(data[r_cond][sortinds])
		annuli_flux.append(convolve(data[r_cond][sortinds], kernel))
		annuli_snr.append(snr_map[r_cond][sortinds])
		# annuli_phi.append(phi[r_cond][sortinds])
		annuli_phi.append(convolve(phi[r_cond][sortinds], kernel))
		annuli_r.append(np.array(len(annuli_flux[-1])*[rr]))
		# rr += dr

	plt.figure(12)
	plt.clf()
	plt.subplot(111)
	plt.subplots_adjust(bottom=0.15, right=0.98, top=0.98, left=0.16)
	for ii in range(len(annuli_flux)):
		plt.plot(np.degrees(annuli_phi[ii]), annuli_flux[ii], '-', alpha=0.3, label="r={} pix".format(annuli_r[ii][0]))
	plt.ylabel('Flux')
	plt.xlabel('PA (0 is +y) [deg]')
	plt.legend(numpoints=1, ncol=3, loc=1)
	if yrange is not None:
		plt.ylim(yrange[0], yrange[1])
	plt.draw()

	plt.figure(13)
	plt.clf()
	plt.subplot(111)
	plt.subplots_adjust(bottom=0.15, right=0.98, top=0.98, left=0.16)
	for ii in range(len(annuli_snr)):
		plt.plot(np.degrees(annuli_phi[ii]), annuli_snr[ii], '.', alpha=0.3)
	plt.ylabel('SNR')
	plt.xlabel('PA (0 is +y) [deg]')
	plt.ylim(-20, 20)
	plt.draw()

	# Do some gymnastics to pad out all annuli arrays to same length.
	annuli_lens = [len(ann) for ann in annuli_flux]
	ann_longest = annuli_flux[np.where(annuli_lens==np.nanmax(annuli_lens))[0][0]]
	template = np.full_like(ann_longest, np.nan)
	annuli_flux_pad = np.tile(template, (len(annuli_flux), 1))
	annuli_snr_pad = annuli_flux_pad.copy()
	annuli_phi_pad = annuli_flux_pad.copy()
	annuli_r_pad = annuli_flux_pad.copy()
	for ii, ann in enumerate(annuli_flux):
		annuli_flux_pad[ii][:len(ann)] = ann
		annuli_snr_pad[ii][:len(ann)] = annuli_snr[ii]
		annuli_phi_pad[ii][:len(ann)] = annuli_phi[ii]
		annuli_r_pad[ii][:len(ann)] = annuli_r[ii]

	# flux_max = np.nanmax([max(ann) for ann in annuli_flux])
	wh_flux_max = np.where(annuli_flux_pad == np.nanmax(annuli_flux_pad))
	# Find the five highest fluxes by (flattened) index.
	wh_flux_top5 = np.argpartition(np.nan_to_num(annuli_flux_pad.flatten()), -5)[-5:]
	flux_top5 = annuli_flux_pad.flatten()[wh_flux_top5]
	snr_top5 = annuli_snr_pad.flatten()[wh_flux_top5]
	phi_top5 = annuli_phi_pad.flatten()[wh_flux_top5]
	r_top5 = annuli_r_pad.flatten()[wh_flux_top5]

	# Discard any points that have low SNR.
	if np.all(snr_top5 < 3):
		print("\nWARNING: S/N < 3 at all max flux points - are you sure there's a disk here?")
	else:
		for ii, sn in enumerate(snr_top5):
			if sn < 3:
				flux_top5[ii] = -1
			# elif flux_top5[ii] > 1.5:
			#     flux_top5[ii] = -1

	# Afterwards, take the max remaining flux as our value.
	wh_max_final = np.where(flux_top5==np.nanmax(flux_top5))
	max_flux = flux_top5[wh_max_final]
	max_snr = snr_top5[wh_max_final]
	max_phi = phi_top5[wh_max_final]
	max_r = r_top5[wh_max_final]

	print("Max Flux = %.2f ADU/s +/- %.2f  (assumed units)" % (max_flux, max_flux/max_snr))
	print("at r = %d px , phi = %.1f deg , SNR = %.1f" % (max_r, np.degrees(max_phi), max_snr))
	print("y x:", np.where(data==max_flux)[0][0]-star[0], np.where(data==max_flux)[1][0]-star[1])

	return


def aperture_disk_brightness(data, noisemap, regionFile, radius=3):
	"""
	
	
	"""
	from regions import read_ds9, write_ds9
	from ttools import make_radii
	
	if 'randomseed' in regionFile:
		seed = int(regionFile.split('randomseed')[-1])
		np.random.seed(seed=seed)
		cens = np.array([70, 70]) + np.array([160, 160])*np.random.rand(20,2)
		# TEMP to output randomly generated positions
		# for cen in cens:
		# 	print('circle({},{},3.5967998)'.format(cen[1]+1, cen[0]+1))
	else:
		regions = read_ds9(regionFile)
		# Get centers in numpy coordinates.
		cens = [np.array([reg.center.y, reg.center.x]) for reg in regions]
	
	apMeans = []
	apSums = []
	noiseApMeans = []
	noiseApSums = []
	noiseApStds = []
	nPixAps = []
	for ii, cen in enumerate(cens):
		radii = make_radii(data, cen)
		apVals = data[radii <= radius]
		apMeans.append(np.nanmean(apVals))
		apSums.append(np.nansum(apVals))
		noiseApVals = noisemap[radii <= radius]
		noiseApMeans.append(np.nanmean(noiseApVals))
		noiseApSums.append(np.nansum(noiseApVals))
		noiseApStds.append(np.nanstd(noiseApVals))
		nPixAps.append(apVals.size)
	
	apMeans = np.array(apMeans)
	apSums = np.array(apSums)
	noiseApMeans = np.array(noiseApMeans)
	noiseApSums = np.array(noiseApSums)
	noiseApStds = np.array(noiseApStds)
	nPixAps = np.array(nPixAps)
	
	return apMeans, apSums, noiseApMeans, noiseApSums, noiseApStds, nPixAps, cens


def plot_aperture_brightness(data, labels=None):
	
	colors = ['#4C7F00', 'C1', '0.5', 'm', 'k', 'c']
	symbols = ['^', 'v', '^', 'v', '^', 'v']
	
	fig = plt.figure(20, figsize=(10.5,5))
	fig.clf()
	ax = plt.subplot(121)
	ax2 = plt.subplot(122)
	plt.subplots_adjust(top=0.95, bottom=0.1, left=0.09, right=0.99, wspace=0.18)
	for ii, da in enumerate(data):
		ax.plot(da[1], markersize=6, marker=symbols[ii], color=colors[ii], label=labels[ii])
	# ax.set_ylabel("Aperture Mean")
	ax.set_ylabel("Aperture Sum", fontsize=14)
	ax.legend(numpoints=1, loc=1, fontsize=12)
	
	# plt.subplots_adjust(top=0.95, bottom=0.1, left=0.14, right=0.97)
	for ii, da in enumerate(data):
		# ax2.plot(da[1]/np.abs(da[3]), marker='.', markersize=8, color=colors[ii], label=labels[ii])
		# Propagated error per pixel.
		errSum = np.sqrt(np.sum(da[5]*da[4]))
		snrSum = da[1]/errSum
		if ii == 0:
			refSnrSum = snrSum.copy()
		ax2.plot(snrSum/refSnrSum, markersize=6, marker=symbols[ii], color=colors[ii], label=labels[ii])

	# ax.set_ylabel("Aperture Mean")
	ax2.set_ylabel("SNR Ratio w/ 'No Sub' SNR", fontsize=14)
	
	for aa in [ax, ax2]:
		aa.tick_params(labelsize=14)
	plt.draw()
	
	return fig, ax


def robust_std(im, method='biweight'):
	# get robust stdev 
	# Method can be biweight or mad, otherwise just normal stdev
	# im must be numpy array
	from astropy.stats import biweight_midvariance
	from scipy.stats import median_absolute_deviation
	if method == 'biweight':
		var = biweight_midvariance(im, axis=None, ignore_nan=True)
		std = np.sqrt(var)
	elif method == 'mad':
		std = median_absolute_deviation(im, axis=None, nan_policy='omit')
	else:
		std = np.nanstd(im, axis=None)

	# From Mike's code:
	"""
	m = np.nanmedian(im) # median value
	d = im - m       # deviation
	ad = np.abs(d)      # absolute deviation
	mad = np.nanmedian(ad) # median absolute deviation
	if mad == 0: std = 0. # no deviation -> zero stdev
	else:
		wt = biweight(d/1.483/mad) # weights
		sum_wt = wt.sum()
		sum_wt2 = (wt**2).sum()
		m = (im*wt).sum() / sum_wt # weighted mean
		d = im-m # deviation from weighted mean
		var = (d**2 * wt).sum() / (sum_wt-sum_wt2/sum_wt) # weighted var
		std = n.sqrt(var) # weighted stdev
	"""
	return std

