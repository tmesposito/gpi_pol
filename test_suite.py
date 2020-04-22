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
from gpi_pol import pol_tools


data_path = os.path.join(os.path.expandvars("$TEST_DATA_PATH"), '')


class Suite:
    
    def __init__(self, test_funcs, func_kwargs, datasets, input_files,
                 test_id='noid'):
        """
        Initialization code for Suite object.
        
        Inputs:
            
        """
        
        self.test_funcs = test_funcs
        self.func_kwargs = func_kwargs
        self.test_id = test_id
        self.data_path = data_path
        self.datasets = datasets
        self.input_files = input_files
    
    # Measure some metrics for comparison...
    def analyze_output(self):
        
        # Noise amplitude at range of radii?
        
        # SNR gain at specific locations?
        
        
        return
    
    # Plot Qphi and Uphi output for the control data and various test functions.
    def plot_rstokes(self):
        fontSize = 14
        for ii, ds in enumerate(self.datasets):
            ims = [fits.getdata(self.input_files[ds]['rstokesdc_nosub'])]
            vmax = np.percentile(ims[0][2][~np.isnan(ims[0][2])], 99.)
            # plt.figure(ii)
            # plt.clf()
            # plt.suptitle(self.input_files[ds]['rstokesdc_nosub'].split('/')[-1])
            fig, ax_list = plt.subplots(2, 2, sharex='col',
                                        gridspec_kw={'hspace': 0, 'wspace': 0})
            plt.suptitle(ds, fontsize=fontSize+2)
            N_axes = ax_list.size
            # Qphi
            ax_list[0][0].imshow(ims[0][1], vmin=0, vmax=vmax)
            ax_list[0][0].set_title('No Sub', fontsize=fontSize)
            # Uphi
            ax_list[1][0].imshow(ims[0][2], vmin=0, vmax=vmax)
            plt.draw()
            
            # ax_list = [ax0, ax1]
            for jj, ax in enumerate(ax_list.flatten()):
                ax.yaxis.set_visible(False)
                ax.xaxis.set_visible(False)
                if jj == 0:
                    ax.text(-0.5, 0.5, 'Q_phi', transform=ax.transAxes)
                if jj == N_axes//2:
                    ax.text(-0.5, 0.5, 'U_phi', transform=ax.transAxes)
        
        pdb.set_trace()
        
        return
    
    # Make some summary plots of results...
    def plot_output(self):
        
        self.plot_rstokes()
        
        return
    
    # Loop through each dataset for each test function.
    def run(self):
        for key in self.test_funcs.keys():
            func = self.test_funcs[key]
            for ds in self.datasets:
                if key == 'remove_quadrupole_rstokes':
                    self.func_kwargs[key]['path_fn'] = self.input_files[ds]["rstokesdc_nosub"]
                func(**self.func_kwargs[key])
                # pdb.set_trace()
        
        return



if __name__ == "__main__":
    
    # The test data sets we are using.
    # HD 32297: high SNR edge-on disk with moderate instrumental noise
    # CE Ant: medium SNR face-on disk with weak instrumental noise
    # NEED: low SNR disk with moderate instrumental noise
    # 73 Her: non-detection with moderate, octopole instrumental noise
    # NEED: non-detection with weak instrumental noise
    datasets = ["HD_32297", "CE_Ant", "73_Her"]
    
    # The input data files we are using, by data set.
    input_files = {}
    # HD 32297 input.
    input_files.update(HD_32297={
                    "podc_nosub":sorted(glob.glob(data_path + "HD_32297/*_podc_distorcorr.fits")),
                    "stokesdc_nosub":data_path + "HD_32297/S20141218S0206_podc_distorcorr_stokesdc.fits",
                    "rstokesdc_nosub":data_path + "HD_32297/S20141218S0206_podc_distorcorr_rstokesdc.fits"})
                    # "stokesdc_nosub":data_path + "HD_32297/S20141218S0206_podc_distorcorr_stokesdc_sm0_stpol1-3.fits",
                    # "rstokesdc_nosub":data_path + "HD_32297/S20141218S0206_podc_distorcorr_rstokesdc_sm0_stpol1-3.fits"})
    # CE Ant input
    input_files.update(CE_Ant={
                    "podc_nosub":sorted(glob.glob(data_path + "CE_Ant/*_podc_distorcorr.fits")),
                    "stokesdc_nosub":data_path + "CE_Ant/S20180405S0070_podc_distorcorr_stokesdc_sm1_stpol13-15.fits",
                    "rstokesdc_nosub":data_path + "CE_Ant/S20180405S0070_podc_distorcorr_rstokesdc_sm1_stpol13-15.fits"})
    input_files.update({'73_Her':{
                    "podc_nosub":sorted(glob.glob(data_path + "73_Her/*_podc_distorcorr.fits")),
                    "stokesdc_nosub":data_path + "73_Her/S20170809S0103_podc_distorcorr_stokesdc.fits",
                    "rstokesdc_nosub":data_path + "73_Her/S20170809S0103_podc_distorcorr_rstokesdc.fits"}})
    
    # Tuple of functions we are going to run on the test data.
    test_funcs = {"remove_quadrupole_rstokes":pol_tools.remove_quadrupole_rstokes}
    
    theta_bounds = (0., np.pi)
    
    func_kwargs = {}
    func_kwargs.update(remove_quadrupole_rstokes={"path_fn":None, "dtheta0":np.pi/2.,
                                "C0":-3., "do_fit":True,
                                "theta_bounds":theta_bounds, "rin":30, "rout":100,
                                "octo":False, "scale_by_r":False, "save":False,
                                "figNum":80, "show_region":True, "path_fn_stokes":None})
    
    
    suite = Suite(test_funcs, func_kwargs, datasets, input_files)
    
    # Run the functions on the test datasets.
    suite.run()
    suite.plot_output()
    
    pdb.set_trace()
    