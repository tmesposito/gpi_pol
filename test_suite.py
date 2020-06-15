#!/usr/bin/env python
#
# $Id$
#
# Tom Esposito (tesposito@berkeley.edu), Aidan Gibbs, Mike Fitzgerald
# 2020-04-20
#
# Test suite for instrumental polarization subtraction from GPI data
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
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pol_tools
from pol_analysis import robust_std
from astropy.stats import sigma_clip


base_dir = os.path.join(os.path.expandvars("$TEST_DATA_PATH"), '')
output_dir = base_dir + "output/"
recipe_temp = base_dir + "template_recipe_stokesdc_from_podc-tme.xml"
queue_path = '/home/aidan/pipelines/gpi_pipeline_1.4.0_r4405_source/data/queue/'


class Suite:
    
    def __init__(self, test_funcs, func_kwargs, datasets, input_files,
                 test_id='noid'):
        """
        Initialization code for Suite object.
        
        Inputs:
            test_funcs: dict
            func_kwargs: dict
            datasets: list
            input_files: dict
            test_id: str name ID for the test you are running.
            
        """
        
        self.test_funcs = test_funcs
        self.func_kwargs = func_kwargs
        self.test_id = test_id
        self.base_dir = base_dir
        self.output_dir = output_dir
        self.datasets = datasets
        self.input_files = input_files
    
    # Set up output directories, etc.
    def setup_dirs(self):
        
        for ds in self.datasets:
            ds_output_dir = self.output_dir + ds
            if not os.path.exists(ds_output_dir):
                os.mkdir(ds_output_dir)
                print("Created empty output directory at {}".format(ds_output_dir))
        
        return
    
    # Measure some metrics for comparison...
    def analyze_output(self):
        
        # Noise amplitude at range of radii
        # Calculated in high pass, low pass, and no filter images
        star = np.array([140,140])
        rin=[10,15,20,25,30,35] # Inner/outer radii of annulus
        rout=[15,20,25,30,35,40]
        sigma=3 # Low pass filter width
        std_method = 'mad' # Method for calculating noise (biweight or mad, else stdev)
        self.noise = {}
        for ii, ds in enumerate(self.datasets):
            self.noise[ds] = {}
            ims = [fits.getdata(self.input_files[ds]['rstokesdc_nosub'])]
            for key in sorted(self.test_funcs.keys()):
                try:
                    if key == "remove_quadrupole_rstokes":
                        fp = glob.glob(self.output_dir + ds + '/*_quadsub.fits')[0]
                    elif key == "remove_quadrupole_podc_I":
                        fp = glob.glob(self.base_dir + ds + '/stokes_output/mean_combined_rstokes_I_quadsub.fits')[0]
                    elif key == "remove_quadrupole_podc_Udiv":
                        fp = glob.glob(self.base_dir + ds + '/stokes_output/mean_combined_rstokes_U_div_quadsub.fits')[0]
                    elif key == "remove_quadrupole_podc_UImix":
                        fp = glob.glob(self.base_dir + ds + '/stokes_output/mean_combined_rstokes_UI_mix_quadsub.fits')[0]
                    ims.append(fits.getdata(fp))

                except:
                    ims.append(np.nan*np.ones(ims[0].shape))

            # Dictionary for normal, low pass, and high pass
            self.noise[ds]['nom'] = {}
            self.noise[ds]['lp'] = {}
            self.noise[ds]['hp'] = {}
            self.noise[ds]['nom']['no_sub'] = []
            self.noise[ds]['lp']['no_sub'] = []
            self.noise[ds]['hp']['no_sub'] = []
            for kk, im in enumerate(ims[1:]):
                    key = sorted(self.test_funcs.keys())[kk]
                    self.noise[ds]['nom'][key] = []
                    self.noise[ds]['lp'][key] = []
                    self.noise[ds]['hp'][key] = []

            for jj in range(6):
                # No Subtraction
                radii = pol_tools.make_radii(ims[0][2], star)
                mask_fit = np.ones(ims[0][2].shape)
                mask_fit[(radii <= rin[jj]) | (radii >= rout[jj])] = np.nan
                ims[0][2] = sigma_clip(ims[0][2], sigma=3, maxiters=1)
                lp_filtered = pol_tools.gauss_filter(ims[0][2],sigma=sigma)
                hp_filtered = ims[0][2] - lp_filtered
                nomn = robust_std(ims[0][2]*mask_fit, method=std_method)
                lpn = robust_std(lp_filtered*mask_fit, method=std_method)
                hpn = robust_std(hp_filtered*mask_fit, method=std_method)
                self.noise[ds]['nom']['no_sub'].append(nomn)
                self.noise[ds]['lp']['no_sub'].append(lpn)
                self.noise[ds]['hp']['no_sub'].append(hpn)

                # Different subtraction methods
                for kk, im in enumerate(ims[1:]):
                    key = sorted(self.test_funcs.keys())[kk]
                    radii = pol_tools.make_radii(im[2], star)
                    mask_fit = np.ones(im[2].shape)
                    mask_fit[(radii <= rin[jj]) | (radii >= rout[jj])] = np.nan
                    im[2] = sigma_clip(im[2], sigma=3, maxiters=1)
                    lp_filtered = pol_tools.gauss_filter(im[2],sigma=sigma)
                    hp_filtered = im[2] - lp_filtered
                    nomn = robust_std(im[2]*mask_fit, method=std_method)
                    lpn = robust_std(lp_filtered*mask_fit, method=std_method)
                    hpn = robust_std(hp_filtered*mask_fit, method=std_method)
                    self.noise[ds]['nom'][key].append(nomn)
                    self.noise[ds]['lp'][key].append(lpn)
                    self.noise[ds]['hp'][key].append(hpn)

            # Create a figure showing noise as a function of radius in low and high pass filtered images
            vmax = np.percentile(ims[0][2][~np.isnan(ims[0][2])], 99.)
            vmin = np.percentile(ims[0][2][~np.isnan(ims[0][2])], 5.)
            fig, axes = plt.subplots(2,3, figsize=(10,6))
            plt.suptitle('Noise Comparison ' + ds + ' - Method = '+std_method, fontsize=16)

            # Top reference image, no filter
            axes[0,0].imshow(ims[0][2], vmin=vmin, vmax=vmax)
            axes[0,0].set_title('No Subtraction - No Filter')

            # Noise plot without filter
            ax0 = axes[1,0]
            ax0.plot(rin, self.noise[ds]['nom']['no_sub'], label='no_sub')
            ax0.scatter(rin, self.noise[ds]['nom']['no_sub'])
            for kk, im in enumerate(ims[1:]):
                key = sorted(self.test_funcs.keys())[kk]
                ax0.plot(rin, self.noise[ds]['nom'][key],label=(key.strip('_').split('_')[-1]))
                ax0.scatter(rin, self.noise[ds]['nom'][key])
            ax0.set_title('No Filter')
            ax0.set_ylabel('Noise')
            ax0.set_xlabel('Inner radius + 5 px')
            ax0.legend()

            # Top reference image, high pass
            axes[0,1].imshow(ims[0][2] - pol_tools.gauss_filter(ims[0][2], sigma=sigma), vmin=vmin, vmax=vmax)
            axes[0,1].set_title('No Subtraction - High Pass')

            # Noise plot, high pass
            ax1 = axes[1,1]
            ax1.plot(rin, self.noise[ds]['hp']['no_sub'], label='no_sub')
            ax1.scatter(rin, self.noise[ds]['hp']['no_sub'])
            for kk, im in enumerate(ims[1:]):
                key = sorted(self.test_funcs.keys())[kk]
                ax1.plot(rin, self.noise[ds]['hp'][key],label=(key.strip('_').split('_')[-1]))
                ax1.scatter(rin, self.noise[ds]['hp'][key])
            ax1.set_title('High Pass')
            ax1.set_ylabel('Noise')
            ax1.set_xlabel('Inner radius + 5 px')
            ax1.legend()

            # Top reference image, low pass
            axes[0,2].imshow(pol_tools.gauss_filter(ims[0][2], sigma=sigma), vmin=vmin, vmax=vmax)
            axes[0,2].set_title('No Subtraction - Low Pass')

            # Noise plot, low pass
            ax2 = axes[1,2]
            ax2.plot(rin, self.noise[ds]['lp']['no_sub'], label='no_sub')
            ax2.scatter(rin, self.noise[ds]['lp']['no_sub'])
            for kk, im in enumerate(ims[1:]):
                key = sorted(self.test_funcs.keys())[kk]
                ax2.plot(rin, self.noise[ds]['lp'][key],label=(key.strip('_').split('_')[-1]))
                ax2.scatter(rin, self.noise[ds]['lp'][key])
            ax2.set_title('Low Pass')
            ax2.set_ylabel('Noise')
            ax2.set_xlabel('Inner radius + 5 px')
            ax2.legend()

            plt.savefig(self.output_dir+ds+'/'+'subtraction_noise_comparison_'+std_method+'.png', dpi=300)



        # SNR gain at specific locations?
        
        
        return
    
    # Plot Qphi and Uphi output for the control data and various test functions.
    def plot_rstokes(self):
        fontSize = 14
        for ii, ds in enumerate(self.datasets):
            ims = [fits.getdata(self.input_files[ds]['rstokesdc_nosub'])]
            for key in sorted(self.test_funcs.keys()):
                try:
                    if key == "remove_quadrupole_rstokes":
                        fp = glob.glob(self.output_dir + ds + '/*_quadsub.fits')[0]
                    elif key == "remove_quadrupole_podc_I":
                        fp = glob.glob(self.base_dir + ds + '/stokes_output/mean_combined_rstokes_I_quadsub.fits')[0]
                    elif key == "remove_quadrupole_podc_Udiv":
                        fp = glob.glob(self.base_dir + ds + '/stokes_output/mean_combined_rstokes_U_div_quadsub.fits')[0]
                    elif key == "remove_quadrupole_podc_UImix":
                        fp = glob.glob(self.base_dir + ds + '/stokes_output/mean_combined_rstokes_UI_mix_quadsub.fits')[0]
                    ims.append(fits.getdata(fp))

                except:
                    ims.append(np.nan*np.ones(ims[0].shape))

            vmax = np.percentile(ims[0][2][~np.isnan(ims[0][2])], 99.)
            # plt.figure(ii)
            # plt.clf()
            # plt.suptitle(self.input_files[ds]['rstokesdc_nosub'].split('/')[-1])

            # Subtraction visual comparison figure
            fig, ax_list = plt.subplots(2, len(ims), sharex='col',
                                        gridspec_kw={'hspace': 0, 'wspace': 0}, figsize=(8,6))
            plt.suptitle(ds, fontsize=fontSize+2)
            N_axes = ax_list.size
            # Qphi
            ax_list[0][0].imshow(ims[0][1], vmin=0, vmax=vmax)
            ax_list[0][0].set_title('No Sub', fontsize=fontSize)
            # Uphi
            ax_list[1][0].imshow(ims[0][2], vmin=0, vmax=vmax)
            
            for kk, im in enumerate(ims[1:]):
                key = sorted(self.test_funcs.keys())[kk]
                ax_list[0][kk+1].imshow(im[1], vmin=0, vmax=vmax)
                ax_list[0][kk+1].set_title(key.strip('_').split('_')[-1], fontsize=fontSize)
                ax_list[1][kk+1].imshow(im[2], vmin=0, vmax=vmax)
            
            plt.draw()
            
            # ax_list = [ax0, ax1]
            for jj, ax in enumerate(ax_list.flatten()):
                ax.yaxis.set_visible(False)
                ax.xaxis.set_visible(False)
                if jj == 0:
                    ax.text(-0.5, 0.5, 'Q_phi', transform=ax.transAxes)
                if jj == N_axes//2:
                    ax.text(-0.5, 0.5, 'U_phi', transform=ax.transAxes)

            plt.savefig(self.output_dir+ds+'/'+'subtraction_visual_comparison.png', dpi=300)

            # Figure with grid of U images subtract in high and low pass filters
            vmin = np.percentile(ims[0][2][~np.isnan(ims[0][2])], 5.)
            sigma=3

            fig, ax_list = plt.subplots(5,3, figsize=(10,10))
            plt.suptitle(ds+' - Uphi', fontsize=fontSize+2)

            ax_list[0][0].imshow(ims[0][2], vmin=vmin, vmax=vmax)
            ax_list[0][0].set_title('No Filter', fontsize=fontSize)
            ax_list[0][0].set_ylabel('No Sub', fontsize=fontSize)
            ax_list[0][1].imshow(ims[0][2] - pol_tools.gauss_filter(ims[0][2], sigma=sigma), vmin=vmin, vmax=vmax)
            ax_list[0][1].set_title('High Pass', fontsize=fontSize)
            ax_list[0][2].imshow(pol_tools.gauss_filter(ims[0][2]), vmin=vmin, vmax=vmax)
            ax_list[0][2].set_title('Low Pass', fontsize=fontSize)

            for kk, im in enumerate(ims[1:]):
                key = sorted(self.test_funcs.keys())[kk]
                ax_list[kk+1][0].imshow(im[2], vmin=vmin, vmax=vmax)
                ax_list[kk+1][0].set_ylabel(key.strip('_').split('_')[-1], fontsize=fontSize)
                ax_list[kk+1][1].imshow(im[2] - pol_tools.gauss_filter(im[2], sigma=sigma), vmin=vmin, vmax=vmax)
                ax_list[kk+1][2].imshow(pol_tools.gauss_filter(im[2], sigma=sigma), vmin=vmin, vmax=vmax)

            plt.subplots_adjust(wspace=None, hspace=None)
            plt.savefig(self.output_dir+ds+'/'+'filtered_image_grid.png', dpi=300)



        return
    
    # Make some summary plots of results...
    def plot_output(self):
        
        self.plot_rstokes()
        
        return
    
    # Loop through each dataset for each test function.
    def run(self):
        self.setup_dirs()
        
        for key in self.test_funcs.keys():
            func = self.test_funcs[key]
            for ds in self.datasets:
                if key == 'remove_quadrupole_rstokes':
                    self.func_kwargs[key]['path_fn'] = self.input_files[ds]["rstokesdc_nosub"]
                    self.func_kwargs[key]['save'] = self.output_dir + "{}/{}_quadsub.fits".format(ds, os.path.splitext(os.path.split(self.input_files[ds]['rstokesdc_nosub'])[-1])[0])
                else:
                    self.func_kwargs[key]['path_fn'] = self.base_dir + ds + '/'
                    self.func_kwargs[key]['path_list'] = self.input_files[ds]["podc_nosub"]
                func(**self.func_kwargs[key])
                
        
        return


if __name__ == "__main__":
    
    # The test data sets we are using.
    # HD 32297: high SNR edge-on disk with moderate instrumental noise
    # CE Ant: medium SNR face-on disk with weak instrumental noise
    # HD 191089: low SNR disk with moderate instrumental noise (octopole)
    # 73 Her: non-detection with moderate, octopole instrumental noise
    # HD 7112: non-detection with weak instrumental noise (light quadrupole)
    datasets = ["HD_32297", "HD_191089", "CE_Ant", "73_Her", "HD_7112"]


    
    # The input data files we are using, by data set.
    input_files = {}

    input_files.update(HD_32297={
                    "podc_nosub":sorted(glob.glob(base_dir + "HD_32297/*_podc_distorcorr.fits")),
                    "stokesdc_nosub":base_dir + "HD_32297/S20141218S0206_podc_distorcorr_stokesdc.fits",
                    "rstokesdc_nosub":base_dir + "HD_32297/S20141218S0206_podc_distorcorr_rstokesdc.fits"})
                    # "stokesdc_nosub":base_dir + "HD_32297/S20141218S0206_podc_distorcorr_stokesdc_sm0_stpol1-3.fits",
                    # "rstokesdc_nosub":base_dir + "HD_32297/S20141218S0206_podc_distorcorr_rstokesdc_sm0_stpol1-3.fits"})
    input_files.update(HD_191089={
                    "podc_nosub":sorted(glob.glob(base_dir + "HD_191089/*_podc_distorcorr.fits")),
                    "stokesdc_nosub":base_dir + "HD_191089/S20150901S0340_podc_distorcorr_stokesdc_sm1_stpol7-13.fits",
                    "rstokesdc_nosub":base_dir + "HD_191089/S20150901S0340_podc_distorcorr_rstokesdc_sm1_stpol7-13.fits"})
    input_files.update(CE_Ant={
                    "podc_nosub":sorted(glob.glob(base_dir + "CE_Ant/*_podc_distorcorr.fits")),
                    "stokesdc_nosub":base_dir + "CE_Ant/S20180405S0070_podc_distorcorr_stokesdc_sm1_stpol13-15.fits",
                    "rstokesdc_nosub":base_dir + "CE_Ant/S20180405S0070_podc_distorcorr_rstokesdc_sm1_stpol13-15.fits"})
    input_files.update({'73_Her':{
                    "podc_nosub":sorted(glob.glob(base_dir + "73_Her/*_podc_distorcorr.fits")),
                    "stokesdc_nosub":base_dir + "73_Her/S20170809S0103_podc_distorcorr_stokesdc.fits",
                    "rstokesdc_nosub":base_dir + "73_Her/S20170809S0103_podc_distorcorr_rstokesdc.fits"}})
    input_files.update(HD_7112={
                    "podc_nosub":sorted(glob.glob(base_dir + "HD_7112/*_podc_distorcorr.fits")),
                    "stokesdc_nosub":base_dir + "HD_7112/S20181121S0094_podc_distorcorr_stokesdc.fits",
                    "rstokesdc_nosub":base_dir + "HD_7112/S20181121S0094_podc_distorcorr_rstokesdc.fits"})
        
    # Dict of functions we are going to run on the test data.
    test_funcs = {"remove_quadrupole_rstokes":pol_tools.remove_quadrupole_rstokes,
                "remove_quadrupole_podc_I":pol_tools.remove_quadrupole_podc_group,
                "remove_quadrupole_podc_Udiv":pol_tools.remove_quadrupole_podc_group,
                "remove_quadrupole_podc_UImix":pol_tools.remove_quadrupole_podc_group}  

    theta_bounds = (0., np.pi)
    
    # Dict of keyword arguments to feed into the funcs in the test_funcs dict.
    # Make sure the dict keys match between the two dicts.
    octo=False

    func_kwargs = {}
    func_kwargs.update(remove_quadrupole_rstokes={"path_fn":None, "dtheta0":np.pi/2.,
                                "C0":-3., "do_fit":True,
                                "theta_bounds":theta_bounds, "rin":30, "rout":100,
                                "octo":octo, "scale_by_r":False, "save":True,
                                "figNum":80, "show_region":True, "path_fn_stokes":None})
    func_kwargs.update(remove_quadrupole_podc_I={"path_fn":None, "recipe_temp":recipe_temp, "queue_path":queue_path, 
                                "path_list":None, "dtheta0":np.pi/2.,
                                "C0":-3., "do_fit":True,
                                "theta_bounds":theta_bounds, "rin":30, "rout":100,
                                "octo":octo, "scale_by_r":False, "save":True,
                                "figNum":80, "quad_scale":'I', "pos_pole":False})
    func_kwargs.update(remove_quadrupole_podc_Udiv={"path_fn":None, "recipe_temp":recipe_temp, "queue_path":queue_path, 
                                "path_list":None, "dtheta0":np.pi/2.,
                                "C0":-3., "do_fit":True,
                                "theta_bounds":theta_bounds, "rin":30, "rout":100,
                                "octo":octo, "scale_by_r":False, "save":True,
                                "figNum":80, "quad_scale":'U_div', "pos_pole":False})
    func_kwargs.update(remove_quadrupole_podc_UImix={"path_fn":None, "recipe_temp":recipe_temp, "queue_path":queue_path, 
                                "path_list":None, "dtheta0":np.pi/2.,
                                "C0":-3., "do_fit":True,
                                "theta_bounds":theta_bounds, "rin":30, "rout":100,
                                "octo":octo, "scale_by_r":False, "save":True,
                                "figNum":80, "quad_scale":'UI_mix', "pos_pole":False})

    
    # Create the test suite object.
    suite = Suite(test_funcs, func_kwargs, datasets, input_files)
    
    # Run the functions on the test datasets.
    suite.run()
    # Plot the output.
    suite.analyze_output()
    suite.plot_output()
    # Eventually, do some analysis on the output....
    
    
    # Pause before exiting. Enter 'c' to end the script.
    # pdb.set_trace()
    