# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 11:52:34 2021

@author: u300737
"""
import glob
import os
import performance
import sys

import numpy as np
import pandas as pd
import xarray as xr


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
from matplotlib import colors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import seaborn as sns
try:
    from cmcrameri import cm as cmaeri
except:
    print("Cmcrameri module cannot be loaded")
try: 
    import typhon
except:
    print("Typhon module cannot be loaded")
        
import measurement_instruments_ql
#import Measurement_Instruments
#def add_sondes_in_plots(date):

def get_sonde_times_as_quicklist(date):
    """
    This function is necessary if you want to easily add dropsondes inside your
    plots. But it justs get the sonde timestamps as lists

    Parameters
    ----------
    date : str
        this is the date of flight to be consider in format YYYYMMDD.

    Returns
    -------
    d_time : list
        list of dropsonde times to easily plot in your figure if desired.

        
    """
    
    input_path = 'C:\\Users\\u300737\\Desktop\\PhD_UHH_WIMI\\Work\\GIT_Repository\\'+\
        'hamp_processing_py\\hamp_processing_python\\Flight_Data\\'+\
            'EUREC4A\dropsonde\\'
    d_time = []
    dropsonde = sorted(glob.glob(input_path+"D"+date+"*"))
    for i in range (len(dropsonde)):
        drop_time = dropsonde[i][-13:-7]
        hour = int(drop_time[0:2])
        minute = int(drop_time[2:4])/60
        second = int(drop_time[4:6])/3600
        drop_time_dec = hour + minute + second
        d_time.append(drop_time_dec)
    return d_time

def add_sondes_in_plots(date,axis):
    """
    

    Parameters
    ----------
    date : str
        this is the date of flight to be consider in format YYYYMMDD.
        
    Returns
    -------
    None.
    
    see also @get_sonde_times_as_quicklist
    
    """
    
def calc_radar_cfad(df,
                    reflectivity_bins=np.linspace(-60,60,121),
                    ):    
    """
    Parameters
    ----------
    df : pd.DataFrame, xr.Dataset or dict
            dataframe of radar reflectivity measurements for given distance,
            ideally with height columns as provided in unified grid of HAMP.
        
    reflectivity_bins : numpy.array
            array of reflectivity bins to group data. Default is binwidth of 1
            for a Ka-Band typical reflectivity range (-60 to 50 dbZ)
            
    Returns
    -------
    cfad_hist : pd.DataFrame
            dataframe of the histogram for given settings columns are binmids

    """
    cfad_hist_dict={}
    ## Create array to assign for dataframe afterwards 
    x_dim=len(df.columns)
    y_dim=len(reflectivity_bins)-1
    # Empty array allocation
    cfad_array=np.empty((x_dim,y_dim))
        
        
    # if dataframe contain np.nans they should be replaced
    #df=df.replace(to_replace=np.nan, value=-reflectivity_bins[0]+0.1)
    cfad_hist=pd.DataFrame(data=cfad_array,index=df.columns,
                               columns=reflectivity_bins[:-1]+0.5)
    cfad_hist_absolute=cfad_hist.copy()
        
    # Start looping
    print("Calculate CFAD for HALO Radar Reflectivity")
    i=0
    perform=performance.performance()
    for height in df.columns:
        perform.updt(len(df.columns),i)
        # Start grouping by pd.cut and pd.value_counts
        bin_groups=pd.cut(df[height],reflectivity_bins)
        height_hist=pd.value_counts(bin_groups).sort_index()        
            
        #Assign counted bins to histogram dataframe
        cfad_hist.iloc[i,:]=height_hist.values/(cfad_hist.shape[0]*cfad_hist.shape[1])
        cfad_hist_absolute.iloc[i,:]=height_hist.values
        i+=1
        
    cfad_hist_dict["relative"]=cfad_hist
    cfad_hist_dict["absolute"]=cfad_hist_absolute
    # Finished, return histogram    
    return cfad_hist_dict

def replace_fill_and_missing_values_to_nan(ds,variables):
    """
    in the final versions missing values are labeled by a certain defined value 
    such as -888 similar to filled values -999.
    
    Parameters
    ----------
    ds : TYPE
        DESCRIPTION.
    variables : TYPE
        DESCRIPTION.

    Returns
    -------
    ds : TYPE
        DESCRIPTION.

    """
    for var in variables:
        ds[var]=xr.where(ds[var]!=-888,ds[var],np.nan)
        ds[var]=xr.where(ds[var]!=-999,ds[var],np.nan)
        ds[var]=xr.where(ds[var]!=-2*888,ds[var],np.nan)
    return ds

def no_pd_dt_set_xticks_and_xlabels(ax, time_extend):
    import datetime
    """    
    From https://github.com/jroettenbacher/phd_base/blob/main/src/pylim/helpers.py :
    This function sets the ticks and labels of the x-axis 
    (only when the x-axis is time in UTC).
    
    Options:
	1 days > time_extend > 12 hours:
            major ticks every 2 hours, minor ticks every 1 hour
	12 hours > time_extend:	
            major ticks every 1 hour, minor ticks every 30 minutes
    Args:
        ax: axis in which the x-ticks and labels have to be set
        time_extend: time difference of t_end - t_start 
        (format datetime.timedelta)
    Returns:
        axis with new ticks and labels
        """
    if time_extend > datetime.timedelta(days=1):
        pass    
    elif datetime.timedelta(days=1) >= time_extend >\
            datetime.timedelta(hours=12):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    elif datetime.timedelta(hours=12) >= time_extend:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0, 60, 30)))
    return ax
#######################################################################
#%% Quicklook Plotter
class Quicklook_Plotter():
    def __init__(self,cfg_dict):
        self.cfg_dict=cfg_dict
        self.base_plot_path= self.cfg_dict["campaign_path"]+"/Flight_Data/"+\
                                self.cfg_dict["campaign"]+"/plots/"
        
        # Define matplotlib Font Size
        set_font=20
        matplotlib.rcParams.update({'font.size':set_font})#,
                           # "text.usetex": True,
                           # "font.family": "sans-serif",
                           # "font.sans-serif": ["Computer Modern Sans Serif"]})
        
        
    def specify_plot_path(self):
        if self.instrument=="radiometer":
            if not hasattr(self,"radiometer_fig_path"):
                self.radiometer_fig_path=self.base_plot_path+"radiometer/"
                if not os.path.exists(self.radiometer_fig_path):
                    os.makedirs(self.radiometer_fig_path)
        elif self.instrument=="radar":
            if not hasattr(self,"radar_fig_path"):
                self.radar_fig_path=self.base_plot_path+"radar/"
                if not os.path.exists(self.radar_fig_path):
                    os.makedirs(self.radar_fig_path)
        else:
            raise Exception("Wrong instrument given.")
#%% Radiometer Quicklook
class Radiometer_Quicklook(Quicklook_Plotter):
    def __init__(self,cfg_dict):
        super(Radiometer_Quicklook,self).__init__(cfg_dict)
        self.specifier()
        
    def specifier(self):    
        self.instrument="radiometer"
        self.specify_plot_path()        
    
    def plot_HAMP_TB_calibration_coeffs_of_flight(self):
        # Load calibration coefficients
        import Measurement_Instruments
        import matplotlib.pyplot as plt
        import seaborn as sns
        HALO_Devices_cls=Measurement_Instruments.HALO_Devices(self.cfg_dict)
        HAMP_cls=Measurement_Instruments.HAMP(HALO_Devices_cls)
        HAMP_cls.get_HAMP_TB_calibration_coeffs_of_flight()
        self.flight_tb_slope_coeff_ds=HAMP_cls.flight_tb_slope_coeff_ds.copy()
        self.flight_tb_offset_coeff_ds=HAMP_cls.flight_tb_offset_coeff_ds.copy()
        
        calib_coeff_fig=plt.figure(figsize=(14,8))
        #%% Slope plots (upper row)
        ax1=calib_coeff_fig.add_subplot(241)
        ax1.scatter(self.flight_tb_slope_coeff_ds["frequency"][0:7].astype(str),
                 self.flight_tb_slope_coeff_ds.values[0:7].astype(float),
                 marker="v",s=100,color="grey",edgecolor="k",zorder=10)
        ax1.set_ylabel("Slope")
        ax1.set_xticklabels("")
        ax1.set_ylim([0.5,1.5])
        sns.despine(ax=ax1,offset=20)

        ax1.spines['bottom'].set_position(('data', 1.0)) 
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        for axis in ['bottom','left']:
            ax1.spines[axis].set_linewidth(2)
            ax1.tick_params(width=2)
        
        ax1.set_title("KV \nModule")
        
        
        ax2=calib_coeff_fig.add_subplot(242)
        ax2.scatter(self.flight_tb_slope_coeff_ds["frequency"][7:14].astype(str),
                 self.flight_tb_slope_coeff_ds.values[7:14].astype(float),
                 marker="v",s=100,color="darkred",edgecolor="k",zorder=10)
        ax2.set_xticklabels("")
        ax2.set_yticklabels("")
        ax2.set_ylim([0.5,1.5])
        sns.despine(ax=ax2,offset=10)

        ax2.set_title("11990 \nModule (A)")
        ax2.spines['bottom'].set_position(('data', 1.0)) 
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        for axis in ['bottom','left']:
            ax2.spines[axis].set_linewidth(2)
            ax2.tick_params(width=2)
        
        ax3=calib_coeff_fig.add_subplot(243)
        
        ax3.scatter(self.flight_tb_slope_coeff_ds["frequency"][14:19].astype(str),
                 self.flight_tb_slope_coeff_ds.values[14:19].astype(float),
                 marker="v",s=100,color="darkgreen",edgecolor="k",zorder=10)
        ax3.set_xticklabels("")
        ax3.set_yticklabels("")
        ax3.set_ylim([0.5,1.5])
        sns.despine(ax=ax3,offset=10)
        ax3.spines['bottom'].set_position(('data', 1.0)) 
        for axis in ['bottom','left']:
            ax3.spines[axis].set_linewidth(2)
            ax3.tick_params(width=2)
        
        ax3.set_title("11990 \nModule (B)")
        
        ax4=calib_coeff_fig.add_subplot(244)
        ax4.scatter(self.flight_tb_slope_coeff_ds["frequency"][19::].astype(str),
                 self.flight_tb_slope_coeff_ds.values[19::].astype(float),
                 marker="v",s=100,color="darkblue",edgecolor="k",zorder=10)
        
        ax4.set_ylim([0.5,1.5])
        ax4.set_yticklabels("")
        sns.despine(ax=ax4,offset=20)
        ax4.spines['bottom'].set_position(('data', 1.0)) 
        ax4.set_title("183 \nModule")
        ax4.set_xticklabels("")
        for axis in ['bottom','left']:
            ax4.spines[axis].set_linewidth(2)
            ax4.tick_params(width=2)
        ax4.set_axisbelow(True)
        
        ax5=calib_coeff_fig.add_subplot(245)
        ax5.set_ylabel("Offset (K)")
        ax5.scatter(self.flight_tb_slope_coeff_ds["frequency"][0:7].astype(str),
                 self.flight_tb_offset_coeff_ds.values[0:7].astype(float),
                 marker="v",s=100,color="grey",edgecolor="k",zorder=10)
        sns.despine(ax=ax5,offset=20)

        ax5.spines["top"].set_visible(False)
        ax5.spines["right"].set_visible(False)
        ax5.spines['bottom'].set_position(('data', 0.0)) 
        
        for label in ax5.get_xticklabels():
            label.set_rotation(55)
            label.set_ha('center')
        ax5.set_ylim([-20,100])
        for axis in ['bottom','left']:
            ax5.spines[axis].set_linewidth(2)
            ax5.tick_params(width=2)
        
        ax6=calib_coeff_fig.add_subplot(246)
        ax6.scatter(self.flight_tb_slope_coeff_ds["frequency"][7:14].astype(str),
                 self.flight_tb_offset_coeff_ds.values[7:14].astype(float),
                 marker="v",s=100,color="darkred",edgecolor="k",zorder=10)
        sns.despine(ax=ax6,offset=20)

        ax6.spines["top"].set_visible(False)
        ax6.spines["right"].set_visible(False)
        ax6.spines['bottom'].set_position(('data', 0.0)) 
        
        for label in ax6.get_xticklabels():
            label.set_rotation(55)
            label.set_ha('center')
        ax6.text(0.9,-0.2,"Frequency (GHz)", transform=ax6.transAxes)
        ax6.set_ylim([-20,100])
        ax6.set_yticklabels("")
        for axis in ['bottom','left']:
            ax6.spines[axis].set_linewidth(2)
            ax6.tick_params(width=2)
        
        ax7=calib_coeff_fig.add_subplot(247)
        ax7.scatter(self.flight_tb_slope_coeff_ds["frequency"][14:19].astype(str),
                 self.flight_tb_offset_coeff_ds.values[14:19].astype(float),
                 marker="v",s=100,color="darkgreen",edgecolor="k",zorder=10)
        sns.despine(ax=ax7,offset=20)
        
        ax7.spines["top"].set_visible(False)
        ax7.spines["right"].set_visible(False)
        ax7.spines['bottom'].set_position(('data', 0.0)) 
        
        for label in ax7.get_xticklabels():
            label.set_rotation(55)
            label.set_ha('center')
        ax7.set_ylim([-20,100])
        ax7.set_yticklabels("")
        for axis in ['bottom','left']:
            ax7.spines[axis].set_linewidth(2)
            ax7.tick_params(width=2)
        
        ax8=calib_coeff_fig.add_subplot(248)
        ax8.scatter(self.flight_tb_slope_coeff_ds["frequency"][19::].astype(str),
                 self.flight_tb_offset_coeff_ds.values[19::].astype(float),
                 marker="v",s=100,color="darkblue",edgecolor="k",zorder=10)
        sns.despine(ax=ax8,offset=20)

        ax8.spines["top"].set_visible(False)
        ax8.spines["right"].set_visible(False)
        ax8.spines['bottom'].set_position(('data', 0.0)) 
        
        for label in ax8.get_xticklabels():
            label.set_rotation(55)
            label.set_ha('center')
        ax8.set_ylim([-20,100])
        ax8.set_yticklabels("")
        for axis in ['bottom','left']:
            ax8.spines[axis].set_linewidth(2)
            ax8.tick_params(width=2)

    def unified_radiometer_plot(self,flight,plot_path, hourly=np.nan,calibrated=True):
        """
        Create quicklooks from processed HALO HAMP microwave radiometer 
        measurements. TBs for each available frequency will be plotted.
        Time axis boundaries will be according to BAHAMAS data.
        - Load MWR data
        - Load BAHAMAS data
        - Plot quicklook overview
        """
        # Plot: first option: quite combined (K, V, 11990, 183) <- four plots +1 for surface
        fig, ax = plt.subplots(5, 1, figsize=(11, 15), sharex=True,
                    gridspec_kw=dict(hspace=0.5,height_ratios=(1,1,1,1,0.1)))
        
        # ax lims:
        #time_lims = [time_var.values[0], time_var.values[-1]]
        tb_lims_k = [130, 270]
        tb_lims_v = [180, 290]
        tb_lims_wf = [180, 290]
        tb_lims_g = [220, 290]
        #try:
        #    time_var=bah.TIME[:]
        #except:
        time_var=np.array(self.radiometer_tb_dict.time[:])
        # time extent:
        time_extent = measurement_instruments_ql.\
                        HALO_Devices.numpydatetime64_to_datetime(time_var[-1]) -\
                    measurement_instruments_ql.\
                        HALO_Devices.numpydatetime64_to_datetime(time_var[0])


        # Plotting:
        # For each receiver, one plot pdf (or at least: KV = plot 1, 11990 + 183 = plot 2
             
        fs = 14
        fs_small = fs - 2
        fs_dwarf = fs - 4
        marker_size = 15
             
        dt_fmt = mdates.DateFormatter("%H:%M") # (e.g. "12:00")
        datetick_auto = True
             
        
        
        
        Tb_df=pd.DataFrame(data=np.array(self.radiometer_tb_dict["TB"]),
                               index=pd.DatetimeIndex(np.array(
                                   self.radiometer_tb_dict.time[:])),
                               columns=np.array(
                                   self.radiometer_tb_dict.uniRadiometer_freq[:]).\
                                   round(2).astype(str))
        Tb_KV_df=Tb_df.loc[:,["22.24","23.04","23.84","25.44","26.24",
                                  "27.84","31.4","50.3","51.76","52.8",
                                  "53.75","54.94","56.66","58.0"]]
        print("TBKVDF:",Tb_KV_df.columns)
        Tb_183_df=Tb_df.loc[:,["183.91","184.81","185.81",
                                    "186.81","188.31","190.81"]]
        Tb_11990_df=Tb_df.loc[:,["90.0","120.15","121.05","122.95","127.25"]]
        
        if not np.isnan(hourly):
            print("Plot for Hour ",hourly)
            Tb_11990_df=Tb_11990_df[Tb_11990_df.index.hour==hourly]
            Tb_183_df=Tb_183_df[Tb_183_df.index.hour==hourly]
            Tb_KV_df=Tb_KV_df[Tb_KV_df.index.hour==hourly]
            
        # some aux info read out of the data:
        #     n_freq_kv = len(HAMP_MWR.freq['KV'])
        #     n_freq_11990 = len(HAMP_MWR.freq['11990'])
        #     n_freq_183 = len(HAMP_MWR.freq['183'])
        #     n_freq_k = 7
        #     n_freq_v = 7
            # colors:
        #     cmap_kv = matplotlib.cm.get_cmap('tab10', n_freq_kv)
        #     cmap_11990 = matplotlib.cm.get_cmap('tab10', n_freq_11990)
        #     cmap_183 = matplotlib.cm.get_cmap('tab10', n_freq_183)
        #     cmap_k = matplotlib.cm.get_cmap('tab10', n_freq_k)
        #     cmap_v = matplotlib.cm.get_cmap('tab10', n_freq_v)
        #     
        #     #import locale
        
        #ax[0].scatter(HAMP_MWR.time_npdt['KV'],
        #                        HAMP_MWR.TB['KV'][:,freq_idx[k]], 
        #                        s=1, color=cmap_k(k), linewidth=0,
        #    					label=f"{HAMP_MWR.freq['KV'][freq_idx[k]]:.2f} GHz")
        
        # K-Band    
        k_colors=["k","darkslateblue","darkmagenta","violet","thistle","gray","silver"]
        ax[0].scatter(Tb_KV_df.index,Tb_KV_df["22.24"],
                      s=1,linewidth=0,label="22.24 GHz",color="k")
        ax[0].scatter(Tb_KV_df.index,Tb_KV_df["23.04"],
                      s=1,linewidth=0,label="23.04 GHz",color="darkslateblue")
        ax[0].scatter(Tb_KV_df.index,Tb_KV_df["23.84"],
                      s=1,linewidth=0,label="23.84 GHz",color="darkmagenta")
        ax[0].scatter(Tb_KV_df.index,Tb_KV_df["25.44"],
                      s=1,linewidth=0,label="25.44 GHz",color="violet")
        ax[0].scatter(Tb_KV_df.index,Tb_KV_df["26.24"],
                      s=1,linewidth=0,label="26.24 GHz",color="thistle")
        ax[0].scatter(Tb_KV_df.index,Tb_KV_df["27.84"],
                      s=1,linewidth=0,label="27.84 GHz",color="gray")
        ax[0].scatter(Tb_KV_df.index,Tb_KV_df["31.4"],
                      s=1,linewidth=0,label="31.40 GHz",color="silver")
        ax[0].set_ylabel("T$_{b}$ in K")
        ax[0].set_ylim([150,350])
        ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        for axis in ['bottom','left']:
            ax[0].spines[axis].set_linewidth(2)
            ax[0].tick_params(width=2)
        #ax[0].legend(loc="center left",bbox_to_anchor=(1.0,0.5),fontsize=12)
        ax[0].set_xticklabels("")
        
        # add figure identifier of subplots: a), b), ...
        ax[0].text(0.02, 1.01, "a) K-Band", fontsize=fs_small, fontweight='bold',
                   ha='left', va='top', transform=ax[0].transAxes)
        ax[1].text(0.02, 1.01, "b) V-Band", fontsize=fs_small, fontweight='bold',
                   ha='left', va='top', transform=ax[1].transAxes)
        ax[2].text(0.02, 1.01, "c) W- and F-Band", fontsize=fs_small, fontweight='bold',
                   ha='left', va='top', transform=ax[2].transAxes)
        ax[3].text(0.02, 1.01, "d) G-Band", fontsize=fs_small, fontweight='bold', 
                   ha='left', va='top', transform=ax[3].transAxes)
        # V-Band
        v_colors=["maroon","red","tomato","indianred","salmon","rosybrown","grey"]
        ax[1].scatter(Tb_KV_df.index,Tb_KV_df["50.3"],
                      s=1,linewidth=0,
                      label="50.3 GHz",color="maroon")
        ax[1].scatter(Tb_KV_df.index,Tb_KV_df["51.76"],s=1,linewidth=0,label="51.76 GHz",color="red")
        ax[1].scatter(Tb_KV_df.index,Tb_KV_df["52.8"],s=1,linewidth=0,label="52.8 GHz",color="tomato")
        ax[1].scatter(Tb_KV_df.index,Tb_KV_df["53.75"],s=1,linewidth=0,label="53.75 GHz",color="indianred")
        ax[1].scatter(Tb_KV_df.index,Tb_KV_df["54.94"],s=1,linewidth=0,label="54.94 GHz",color="salmon")
        ax[1].scatter(Tb_KV_df.index,Tb_KV_df["56.66"],s=1,linewidth=0,label="56.66 GHz",color="rosybrown")
        ax[1].scatter(Tb_KV_df.index,Tb_KV_df["58.0"],s=1,linewidth=0,label="58.00 GHz",color="grey")
        ax[1].set_ylabel("T$_{b}$ in K")
        ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        for axis in ['bottom','left']:
            ax[1].spines[axis].set_linewidth(2)
            ax[1].tick_params(width=2)
        
        ax[1].set_ylim([150,350])
        ax[1].set_xticklabels("")
        #ax[1].legend(loc="center left",bbox_to_anchor=(1.0,0.5),fontsize=12)
        
        # W-Band
        w_colors=["darkgreen","green","forestgreen","mediumseagreen","darkseagreen"]
        ax[2].scatter(Tb_11990_df.index,Tb_11990_df["90.0"],s=1,linewidth=0,label="90.0 GHz",color="darkgreen")
        ax[2].scatter(Tb_11990_df.index,Tb_11990_df["120.15"],s=1,linewidth=0,label="(118.75 +/- 1.4) GHz",
                 color="green")
        ax[2].scatter(Tb_11990_df.index,Tb_11990_df["121.05"],s=1,linewidth=0,label="(118.75 +/- 2.3) GHz",
                 color="forestgreen")
        ax[2].scatter(Tb_11990_df.index,Tb_11990_df["122.95"],s=1,linewidth=0,label="(118.75 +/- 4.2) GHz",
                 color="mediumseagreen")
        ax[2].scatter(Tb_11990_df.index,Tb_11990_df["127.25"],s=1,linewidth=0,label="(118.75 +/- 8.5) GHz",
                 color="darkseagreen")
        ax[2].set_ylabel("T$_{b}$ in K")
        ax[2].set_ylim([150,350])
        ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        for axis in ['bottom','left']:
            ax[2].spines[axis].set_linewidth(2)
            ax[2].tick_params(width=2)
        ax[2].set_xticklabels("")
        #ax[2].legend(loc="center left",bbox_to_anchor=(1.0,0.5),fontsize=12)
        
        # G-Band
        g_colors=["k","midnightblue","blue","royalblue","steelblue","skyblue","grey"]
        ax[3].scatter(Tb_183_df.index,Tb_183_df["183.91"],s=1,linewidth=0,label="(183.31 +/- 0.6) GHz",
                 color="k")
        ax[3].scatter(Tb_183_df.index,Tb_183_df["184.81"],s=1,linewidth=0,label="(183.31 +/- 1.5) GHz",
                 color="midnightblue")
        ax[3].scatter(Tb_183_df.index,Tb_183_df["185.81"],s=1,linewidth=0,label="(183.31 +/- 2.5) GHz",
                 color="blue")
        ax[3].scatter(Tb_183_df.index,Tb_183_df["186.81"],s=1,linewidth=0,label="(183.31 +/- 3.5) GHz",
                 color="royalblue")
        ax[3].scatter(Tb_183_df.index,Tb_183_df["188.31"],s=1,linewidth=0,label="(183.31 +/- 5.0) GHz",
                 color="steelblue")
        ax[3].scatter(Tb_183_df.index,Tb_183_df["190.81"],s=1,linewidth=0,label="(183.31 +/- 7.5) GHz",
                 color="skyblue")
        try:
            ax[3].plot(Tb_183_df.index,Tb_183_df["195.81"],label="(183.31 +/- 12.5) GHz",
                     color="grey")
        except:
            pass
        ax[3].set_ylabel("T$_{b}$ in K")
        ax[3].set_xlabel("Time in UTC")
        ax[3].set_ylim([150,350])
        ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        #ax[3].legend(loc="center left",bbox_to_anchor=(1.0,0.5),fontsize=12)
        for axis in ['bottom','left']:
            ax[3].spines[axis].set_linewidth(2)
            ax[3].tick_params(width=2)        
        
        # legends or colorbars:
        lh, ll = ax[0].get_legend_handles_labels()
        lg = ax[0].legend(handles=lh, labels=ll, loc="lower center",
                          bbox_to_anchor=(0.5,1.05), ncol=4, handletextpad=0,
                          columnspacing=1, borderaxespad=0, handlelength=0,
                          fontsize=fs_small)
        for item in lg.legendHandles:
                item.set_visible(False)
        for k, text in enumerate(lg.get_texts()):
            text.set_color(k_colors[k])
        
        lh, ll = ax[1].get_legend_handles_labels()
        lg = ax[1].legend(handles=lh, labels=ll, loc="lower center", 
                          bbox_to_anchor=(0.5,1.05), ncol=4, handletextpad=0,
                          columnspacing=1, borderaxespad=0, handlelength=0,
                          fontsize=fs_small)
        for item in lg.legendHandles:
                item.set_visible(False)
        for k, text in enumerate(lg.get_texts()):
            text.set_color(v_colors[k])
        
        lh, ll = ax[2].get_legend_handles_labels()
        lg = ax[2].legend(handles=lh, labels=ll, loc="lower center", 
                          bbox_to_anchor=(0.5,1.05), ncol=3, handletextpad=0,
                          columnspacing=1, borderaxespad=0, handlelength=0,
                          fontsize=fs_small)
        for item in lg.legendHandles:
                item.set_visible(False)
        for k, text in enumerate(lg.get_texts()):
            text.set_color(w_colors[k])
        
        lh, ll = ax[3].get_legend_handles_labels()
        lg = ax[3].legend(handles=lh, labels=ll, loc="lower center",
                          bbox_to_anchor=(0.5,1.05), ncol=3, handletextpad=0,
                          columnspacing=1, borderaxespad=0, handlelength=0,
                          fontsize=fs_small)
        for item in lg.legendHandles:
                item.set_visible(False)
        for k, text in enumerate(lg.get_texts()):
            text.set_color(g_colors[k])
        ax[0].set_ylim(bottom=tb_lims_k[0], top=tb_lims_k[1])
        ax[1].set_ylim(bottom=tb_lims_v[0], top=tb_lims_v[1])
        ax[2].set_ylim(bottom=tb_lims_wf[0], top=tb_lims_wf[1])
        ax[3].set_ylim(bottom=tb_lims_g[0], top=tb_lims_g[1])
        #define_color_map
        for axx in ax:
            if not axx==ax[4]:
                axx.spines.right.set_visible(False)
                axx.spines.top.set_visible(False)
            #axx.set_visible({"top":False,"right":False})
            axx = no_pd_dt_set_xticks_and_xlabels(ax=axx,
                                time_extend=time_extent)
        # Add surface mask
        # plot AMSR2 sea ice concentration
        blue_colorbar=cm.get_cmap('Blues_r', 22)
        blue_cb=blue_colorbar(np.linspace(0, 1, 22))
        brown_rgb = np.array(colors.hex2color(colors.cnames['brown']))
        
        blue_cb[:2, :] = [*brown_rgb,1]
        newcmp = ListedColormap(blue_cb)
        bah_df=pd.DataFrame(data=np.nan,columns=["sea_ice"],
                            index=pd.DatetimeIndex(Tb_df.index))
        bah_df["sea_ice"]=self.radiometer_tb_dict["surface_mask"].values[:]
        im = ax[4].pcolormesh(np.array([pd.DatetimeIndex(bah_df["sea_ice"].index),
                                        pd.DatetimeIndex(bah_df["sea_ice"].index)]),
                              np.array([0, 1]),
                              np.tile(np.array(bah_df["sea_ice"].values),(2,1)),
                              cmap=newcmp, vmin=-0.1, vmax=1,
                              shading='auto')
        cax = fig.add_axes([0.7, 0.1, 0.15, ax[4].get_position().height])
        C1=fig.colorbar(im, cax=cax, orientation='horizontal')
        C1.set_label(label='Sea ice [%]',fontsize=fs_small)
        C1.ax.tick_params(labelsize=fs_small)
        ax[4].tick_params(axis='x', labelleft=False, 
                          left=False,labelsize=fs_small)
        ax[4].tick_params(axis='y', labelleft=False, left=False)
        ax[4].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        # set labels:
        ax[4].set_xlabel("Time (HH:MM)",fontsize=fs_small)#" of {str(time_var[round(len(time_var)/2)].values)[:10]}",
        #                 fontsize=fs_small)
        
        # Limit axis spacing:
        #plt.subplots_adjust(hspace=0.35)			# removes space between subplots
        box = ax[4].get_position()
        
        box.y0 = box.y0 + 0.025
        box.y1 = box.y1 + 0.025
        
        ax[4]=ax[4].set_position(box)
        
        
        radiometer_str="processed_radiometer"
        
        if calibrated: radiometer_str="calibrated and "+radiometer_str
        fig_name="unified_radiometer_tb_quicklook_"+\
            flight+"_"+self.cfg_dict["date"]+".png"
        print(fig_name)
        plot_path+="/unified_dataset/"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        fig.savefig(plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as:",plot_path+fig_name)       
        #self.define_quicklook_fig_name("TB")         
        #fig_file=self.radiometer_fig_path+self.fig_name
        #fig.savefig(fig_file,dpi=300,bbox_inches="tight")
        #print(self.instrument," Quicklook saved as:",fig_file)
        
    def plot_radiometer_TBs(self,date,raw_measurements=True,hourly=np.nan):
        matplotlib.rcParams.update({"font.size":18})
        print("Plotting ...")
        if raw_measurements:
            Tb_KV_df=self.radiometer_tb_dict["KV"]
            Tb_183_df=self.radiometer_tb_dict["183"]
            Tb_11990_df=self.radiometer_tb_dict["11990"]
        else:
            Tb_df=pd.DataFrame(data=np.array(self.radiometer_tb_dict["TB"][:]),
                               index=pd.DatetimeIndex(np.array(
                                   self.radiometer_tb_dict.time[:])),
                               columns=np.array(
                                   self.radiometer_tb_dict.uniRadiometer_freq[:]).\
                                   round(2).astype(str))
            #print("Tb_df:",Tb_df)
            Tb_KV_df=Tb_df.loc[:,["22.24","23.04","23.84","25.44","26.24",
                                  "27.84","31.4","50.3","51.76","52.8",
                                  "53.75","54.94","56.66","58.0"]]
            print("TBKVDF:",Tb_KV_df.columns)
            Tb_183_df=Tb_df.loc[:,["183.91","184.81","185.81",
                                    "186.81","188.31","190.81"]]
            Tb_11990_df=Tb_df.loc[:,["90.0","120.15","121.05","122.95","127.25"]]
        
        if not np.isnan(hourly):
            print("Plot for Hour ",hourly)
            Tb_11990_df=Tb_11990_df[Tb_11990_df.index.hour==hourly]
            Tb_183_df=Tb_183_df[Tb_183_df.index.hour==hourly]
            Tb_KV_df=Tb_KV_df[Tb_KV_df.index.hour==hourly]
            
        fig = plt.figure(figsize=(12,12))
        ax1=fig.add_subplot(411)
        ax1.plot(Tb_KV_df["22.24"],label="22.24 GHz",color="k")
        ax1.plot(Tb_KV_df["23.04"],label="23.04 GHz",color="darkslateblue")
        ax1.plot(Tb_KV_df["23.84"],label="23.84 GHz",color="darkmagenta")
        ax1.plot(Tb_KV_df["25.44"],label="25.44 GHz",color="violet")
        ax1.plot(Tb_KV_df["26.24"],label="26.24 GHz",color="thistle")
        ax1.plot(Tb_KV_df["27.84"],label="27.84 GHz",color="gray")
        ax1.plot(Tb_KV_df["31.4"],label="31.40 GHz",color="silver")
        ax1.set_ylabel("T$_{b}$ in K")
        ax1.set_ylim([150,350])
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        for axis in ['bottom','left']:
            ax1.spines[axis].set_linewidth(2)
            ax1.tick_params(width=2)
        ax1.legend(loc="center left",bbox_to_anchor=(1.0,0.5),fontsize=12)
        ax1.set_xticklabels("")
        ###########################################################################
        ax2=fig.add_subplot(412)
        ax2.plot(Tb_KV_df["50.3"],label="50.3 GHz",color="maroon")
        ax2.plot(Tb_KV_df["51.76"],label="51.76 GHz",color="red")
        ax2.plot(Tb_KV_df["52.8"],label="52.8 GHz",color="tomato")
        ax2.plot(Tb_KV_df["53.75"],label="53.75 GHz",color="indianred")
        ax2.plot(Tb_KV_df["54.94"],label="54.94 GHz",color="salmon")
        ax2.plot(Tb_KV_df["56.66"],label="56.66 GHz",color="rosybrown")
        ax2.plot(Tb_KV_df["58.0"],label="58.00 GHz",color="grey")
        ax2.set_ylabel("T$_{b}$ in K")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        for axis in ['bottom','left']:
            ax2.spines[axis].set_linewidth(2)
            ax2.tick_params(width=2)
        ax2.set_ylim([150,350])
        ax2.set_xticklabels("")
        ax2.legend(loc="center left",bbox_to_anchor=(1.0,0.5),fontsize=12)
        #######################################################################
        ax3= fig.add_subplot(413,sharex=ax1)
        ax3.plot(Tb_11990_df["90.0"],label="90.0 GHz",color="darkgreen")
        ax3.plot(Tb_11990_df["120.15"],label="(118.75 +/- 1.4) GHz",
                 color="green")
        ax3.plot(Tb_11990_df["121.05"],label="(118.75 +/- 2.3) GHz",
                 color="forestgreen")
        ax3.plot(Tb_11990_df["122.95"],label="(118.75 +/- 4.2) GHz",
                 color="mediumseagreen")
        ax3.plot(Tb_11990_df["127.25"],label="(118.75 +/- 8.5) GHz",
                 color="darkseagreen")
        ax3.set_ylabel("T$_{b}$ in K")
        ax3.set_ylim([150,350])
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        for axis in ['bottom','left']:
            ax3.spines[axis].set_linewidth(2)
            ax3.tick_params(width=2)
        ax3.set_xticklabels("")
        ax3.legend(loc="center left",bbox_to_anchor=(1.0,0.5),fontsize=12)
        #######################################################################
        ax4= fig.add_subplot(414,sharex=ax1)
        ax4.plot(Tb_183_df["183.91"],label="(183.31 +/- 0.6) GHz",
                 color="k")
        ax4.plot(Tb_183_df["184.81"],label="(183.31 +/- 1.5) GHz",
                 color="midnightblue")
        ax4.plot(Tb_183_df["185.81"],label="(183.31 +/- 2.5) GHz",
                 color="blue")
        ax4.plot(Tb_183_df["186.81"],label="(183.31 +/- 3.5) GHz",
                 color="royalblue")
        ax4.plot(Tb_183_df["188.31"],label="(183.31 +/- 5.0) GHz",
                 color="steelblue")
        ax4.plot(Tb_183_df["190.81"],label="(183.31 +/- 7.5) GHz",
                 color="skyblue")
        try:
            ax4.plot(Tb_183_df["195.81"],label="(183.31 +/- 12.5) GHz",
                     color="grey")
        except:
            pass
        ax4.set_ylabel("T$_{b}$ in K")
        ax4.set_xlabel("Time in UTC")
        ax4.set_ylim([150,350])
        ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax4.legend(loc="center left",bbox_to_anchor=(1.0,0.5),fontsize=12)
        for axis in ['bottom','left']:
            ax4.spines[axis].set_linewidth(2)
            ax4.tick_params(width=2)
        
        sns.despine(offset=5)
        if np.isnan(hourly):
            fig_name="HAMP_Tb_"+[*self.cfg_dict["Flight_Dates_used"].keys()][0]+\
                    "_"+str(date)+".png"        
        else:
            fig_name="HAMP_Tb_"+[*self.cfg_dict["Flight_Dates_used"].keys()][0]+\
                    "_"+str(date)+"_"+str(hourly)+"00.png"        
            
        if raw_measurements:
            plt.suptitle("Raw Measurements")
            fig_name="Raw_"+fig_name
        #if np.isnan(hourly):
        #    axs[1,1].xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0]))        
        #else:
        #    axs[1,1].xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,15,30,45]))
        #axs[1,1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))        
        #axs[1,1].set_xlabel('Time (UTC)')
        #sns.despine(offset=10)
        
        fig.savefig(self.radiometer_fig_path+fig_name,
                    dpi=300,bbox_inches="tight")
        print("Figure saved as : ",self.radiometer_fig_path+fig_name)        
        return None
    
    def plot_radiometer_TB_calibration_comparison(self,hourly=np.nan):
        import Measurement_Instruments
        HALO_Devices_cls=Measurement_Instruments.HALO_Devices(self.cfg_dict)
        HAMP_cls=Measurement_Instruments.HAMP(HALO_Devices_cls)
        HAMP_cls.open_processed_hamp_data(self.cfg_dict,open_calibrated=False)
        HAMP_cls.open_processed_hamp_data(self.cfg_dict,open_calibrated=True)
        matplotlib.rcParams.update({"font.size":18})
        
        uncalib_ds=HAMP_cls.processed_hamp_ds.copy()
        calib_ds=HAMP_cls.calib_processed_hamp_ds.copy()
        if not np.isnan(hourly):
            print("Plot for Hour ",hourly)
            print("Currently this is not provided, but under construction")
        test_old=pd.Series(uncalib_ds["TB"].sel({"uniRadiometer_freq":22.24}))
        test_new=pd.Series(calib_ds["TB"].sel({"uniRadiometer_freq":22.24}))
        #######################################################################
        fig = plt.figure(figsize=(12,16))
        ax1=fig.add_subplot(411)
        ax1.plot(uncalib_ds["TB"].time[:],calib_ds["TB"].sel({"uniRadiometer_freq":22.24}),
                 color="k",ls="-",lw=2,alpha=0.4)
        ax1.plot(uncalib_ds["TB"].time[:],uncalib_ds["TB"].sel({"uniRadiometer_freq":22.24}),
                 label="22.24 GHz",color="k",ls="--",lw=1,alpha=0.4)
        
        ax1.plot(uncalib_ds["TB"].time[:],calib_ds["TB"].sel({"uniRadiometer_freq":23.04}),
                label="23.04 GHz",color="darkslateblue",ls="-",lw=2)
        ax1.plot(uncalib_ds["TB"].time[:],uncalib_ds["TB"].sel({"uniRadiometer_freq":23.04}),
                color="darkslateblue",ls="--",lw=1,alpha=0.4)
        
        ax1.plot(uncalib_ds["TB"].time[:],calib_ds["TB"].sel({"uniRadiometer_freq":23.84}),
                label="23.84 GHz",color="darkmagenta",ls="-",lw=2)
        ax1.plot(uncalib_ds["TB"].time[:],uncalib_ds["TB"].sel({"uniRadiometer_freq":23.84}),
                color="darkmagenta",ls="--",lw=1,alpha=0.4)
        
        ax1.plot(uncalib_ds["TB"].time[:],calib_ds["TB"].sel({"uniRadiometer_freq":25.44}),
                label="25.44 GHz",color="violet",ls="-",lw=2)
        ax1.plot(uncalib_ds["TB"].time[:],uncalib_ds["TB"].sel({"uniRadiometer_freq":25.44}),
                color="violet",ls="--",lw=1,alpha=0.4)
        
        ax1.plot(uncalib_ds["TB"].time[:],calib_ds["TB"].sel({"uniRadiometer_freq":26.24}),
                label="26.24 GHz",color="thistle",ls="-",lw=2)
        ax1.plot(uncalib_ds["TB"].time[:],uncalib_ds["TB"].sel({"uniRadiometer_freq":26.24}),
                color="thistle",ls="--",lw=1,alpha=0.4)
        
        ax1.plot(uncalib_ds["TB"].time[:],calib_ds["TB"].sel({"uniRadiometer_freq":27.84}),
                label="27.84 GHz",color="gray",ls="-",lw=2)
        ax1.plot(uncalib_ds["TB"].time[:],uncalib_ds["TB"].sel({"uniRadiometer_freq":27.84}),
                color="gray",ls="--",lw=1,alpha=0.4)
        
        ax1.plot(uncalib_ds["TB"].time[:],calib_ds["TB"].sel({"uniRadiometer_freq":31.4}),
                label="31.40 GHz",color="silver",ls="-",lw=2)
        ax1.plot(uncalib_ds["TB"].time[:],uncalib_ds["TB"].sel({"uniRadiometer_freq":31.4}),
                color="silver",ls="--",lw=1,alpha=0.4)
        
        ax1.set_ylabel("T$_{b}$ in K")
        ax1.set_ylim([150,210])
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        for axis in ['bottom','left']:
            ax1.spines[axis].set_linewidth(2)
            ax1.tick_params(width=2)
        ax1.legend(loc="center left",bbox_to_anchor=(1.0,0.5),fontsize=12)
        ax1.set_xticklabels("")
        
        ###########################################################################
        ax2=fig.add_subplot(412)
        ax2.plot(uncalib_ds["TB"].time[:],
                 uncalib_ds["TB"].sel({"uniRadiometer_freq":50.3}),
                 color="maroon",lw=1,ls="--",alpha=0.4)
        ax2.plot(calib_ds["TB"].time[:],
                 calib_ds["TB"].sel({"uniRadiometer_freq":50.3}),
                 label="50.3 GHz",color="maroon",lw=2,ls="-")
        
        ax2.plot(uncalib_ds["TB"].time[:],
                 uncalib_ds["TB"].sel({"uniRadiometer_freq":51.76}),
                 color="red",lw=1,ls="--",alpha=0.4)
        ax2.plot(calib_ds["TB"].time[:],
                 calib_ds["TB"].sel({"uniRadiometer_freq":51.76}),
                 label="51.76 GHz",color="red",lw=2,ls="-")
        
        ax2.plot(uncalib_ds["TB"].time[:],
                 uncalib_ds["TB"].sel({"uniRadiometer_freq":52.8}),
                 color="tomato",lw=1,ls="--",alpha=0.4)
        ax2.plot(calib_ds["TB"].time[:],
                 calib_ds["TB"].sel({"uniRadiometer_freq":52.8}),
                 label="52.8 GHz",color="tomato",lw=2,ls="-")
        
        ax2.plot(uncalib_ds["TB"].time[:],
                 uncalib_ds["TB"].sel({"uniRadiometer_freq":53.75}),
                 color="indianred",lw=2,ls="-",alpha=0.4)
        ax2.plot(calib_ds["TB"].time[:],
                 calib_ds["TB"].sel({"uniRadiometer_freq":53.75}),
                 label="53.75 GHz",color="indianred",lw=2,ls="-")
        
        ax2.plot(uncalib_ds["TB"].time[:],
                 uncalib_ds["TB"].sel({"uniRadiometer_freq":54.94}),
                 color="salmon",lw=1,ls="--",alpha=0.4)
        ax2.plot(calib_ds["TB"].time[:],
                 calib_ds["TB"].sel({"uniRadiometer_freq":54.94}),
                 label="54.94 GHz",color="salmon",lw=2,ls="-")
        
        ax2.plot(uncalib_ds["TB"].time[:],
                 uncalib_ds["TB"].sel({"uniRadiometer_freq":56.66}),
                 color="rosybrown",lw=1,ls="--",alpha=0.4)
        ax2.plot(calib_ds["TB"].time[:],
                 calib_ds["TB"].sel({"uniRadiometer_freq":56.66}),
                 label="56.66 GHz",color="rosybrown",lw=2,ls="-")
        
        ax2.plot(uncalib_ds["TB"].time[:],
                 uncalib_ds["TB"].sel({"uniRadiometer_freq":58.0}),
                 color="grey",lw=1,ls="--",alpha=0.4)
        ax2.plot(calib_ds["TB"].time[:],
                 calib_ds["TB"].sel({"uniRadiometer_freq":58.0}),
                 label="58.00 GHz",color="grey",lw=2,ls="-")
        
        ax2.set_ylabel("T$_{b}$ in K")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        for axis in ['bottom','left']:
            ax2.spines[axis].set_linewidth(2)
            ax2.tick_params(width=2)
        ax2.set_ylim([220,280])
        ax2.set_xticklabels("")
        ax2.legend(loc="center left",bbox_to_anchor=(1.0,0.5),fontsize=12)
        
        #######################################################################
        ax3= fig.add_subplot(413,sharex=ax1)
        ax3.plot(calib_ds["TB"].time[:],
                 uncalib_ds["TB"].sel({"uniRadiometer_freq":90.0}),
                 color="darkgreen",lw=1,ls="--",alpha=0.4)
        ax3.plot(calib_ds["TB"].time[:],
                 calib_ds["TB"].sel({"uniRadiometer_freq":90.0}),
                 label="90.0 GHz",color="darkgreen",
                 lw=2,ls="-")
        
        ax3.plot(calib_ds["TB"].time[:],
                 uncalib_ds["TB"].sel({"uniRadiometer_freq":120.15}),
                 color="green",lw=1,ls="--",alpha=0.4)
        ax3.plot(calib_ds["TB"].time[:],
                 calib_ds["TB"].sel({"uniRadiometer_freq":120.15}),
                 label="(118.75 +/- 1.4) GHz",
                 color="green",lw=2,ls="-")
        
        ax3.plot(calib_ds["TB"].time[:],
                 uncalib_ds["TB"].sel({"uniRadiometer_freq":121.05}),
                 color="forestgreen",lw=1,ls="--",alpha=0.4)
        ax3.plot(calib_ds["TB"].time[:],
                 calib_ds["TB"].sel({"uniRadiometer_freq":121.05}),
                 label="(118.75 +/- 2.3) GHz",
                 color="forestgreen",lw=2,ls="-")
        
        ax3.plot(calib_ds["TB"].time[:],
                 uncalib_ds["TB"].sel({"uniRadiometer_freq":122.95}),
                 color="mediumseagreen",lw=1,ls="--",alpha=0.4)
        ax3.plot(calib_ds["TB"].time[:],
                 calib_ds["TB"].sel({"uniRadiometer_freq":122.95}),
                 label="(118.75 +/- 4.2) GHz",
                 color="mediumseagreen",lw=2,ls="-")
        
        ax3.plot(calib_ds["TB"].time[:],
                 uncalib_ds["TB"].sel({"uniRadiometer_freq":127.25}),
                 color="darkseagreen",lw=1,ls="--",alpha=0.4)
        ax3.plot(calib_ds["TB"].time[:],
                 calib_ds["TB"].sel({"uniRadiometer_freq":127.25}),
                 label="(118.75 +/- 8.5) GHz",
                 color="darkseagreen",lw=2,ls="-")
        
        ax3.set_ylabel("T$_{b}$ in K")
        ax3.set_ylim([220,280])
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        for axis in ['bottom','left']:
            ax3.spines[axis].set_linewidth(2)
            ax3.tick_params(width=2)
        ax3.set_xticklabels("")
        ax3.legend(loc="center left",bbox_to_anchor=(1.0,0.5),fontsize=12)
        
        #######################################################################
        ax4= fig.add_subplot(414,sharex=ax1)
        ax4.plot(calib_ds["TB"].time[:],
                 uncalib_ds["TB"].sel({"uniRadiometer_freq":183.91}),
                 color="k",ls="--",lw=1,alpha=0.4)
        ax4.plot(calib_ds["TB"].time[:],
                 calib_ds["TB"].sel({"uniRadiometer_freq":183.91}),
                 label="(183.31 +/- 0.6) GHz",
                 color="k",lw=2,ls="-")
        
        ax4.plot(calib_ds["TB"].time[:],
                 uncalib_ds["TB"].sel({"uniRadiometer_freq":184.81}),
                 color="midnightblue",ls="--",lw=1,alpha=0.4)
        ax4.plot(calib_ds["TB"].time[:],
                 calib_ds["TB"].sel({"uniRadiometer_freq":184.81}),
                 label="(183.31 +/- 1.5) GHz",color="midnightblue",ls="-",lw=2)
        
        ax4.plot(calib_ds["TB"].time[:],
                 uncalib_ds["TB"].sel({"uniRadiometer_freq":185.81}),
                 color="blue",ls="--",lw=1,alpha=0.4)
        ax4.plot(calib_ds["TB"].time[:],
                 calib_ds["TB"].sel({"uniRadiometer_freq":185.81}),
                 label="(183.31 +/- 2.5) GHz",color="blue",ls="-",lw=2)
        
        ax4.plot(calib_ds["TB"].time[:],
                 uncalib_ds["TB"].sel({"uniRadiometer_freq":186.81}),
                 color="royalblue",ls="--",lw=1,alpha=0.4)
        ax4.plot(calib_ds["TB"].time[:],
                 calib_ds["TB"].sel({"uniRadiometer_freq":186.81}),
                 label="(183.31 +/- 3.5) GHz",color="royalblue",ls="-",lw=2)
        
        ax4.plot(calib_ds["TB"].time[:],
                 uncalib_ds["TB"].sel({"uniRadiometer_freq":188.31}),
                 color="steelblue",ls="--",lw=1,alpha=0.4)
        ax4.plot(calib_ds["TB"].time[:],
                 calib_ds["TB"].sel({"uniRadiometer_freq":188.31}),
                 label="(183.31 +/- 5.0) GHz",color="steelblue",ls="-",lw=2)
        
        ax4.plot(calib_ds["TB"].time[:],
                 calib_ds["TB"].sel({"uniRadiometer_freq":190.81}),
                 color="skyblue",ls="--",lw=1)
        ax4.plot(calib_ds["TB"].time[:],
                 calib_ds["TB"].sel({"uniRadiometer_freq":190.81}),
                 label="(183.31 +/- 7.5) GHz",color="skyblue",ls="-",lw=2)
        
        try:
            ax4.plot(calib_ds["TB"].time[:],
                     uncalib_ds["TB"].sel({"uniRadiometer_freq":195.81}),
                     color="grey",ls="--",lw=1)
            
            ax4.plot(calib_ds["TB"].time[:],
                     calib_ds["TB"].sel({"uniRadiometer_freq":195.81}),
                     label="(183.31 +/- 12.5) GHz",color="grey",ls="-",lw=2)
        
        except:
            pass
        
        ax4.set_ylabel("T$_{b}$ in K")
        ax4.set_xlabel("Time in UTC")
        ax4.set_ylim([250,300])
        ax4.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax4.legend(loc="center left",bbox_to_anchor=(1.0,0.5),fontsize=12)
        for axis in ['bottom','left']:
            ax4.spines[axis].set_linewidth(2)
            ax4.tick_params(width=2)
        
        sns.despine(offset=5)
        #if np.isnan(hourly):
        #    fig_name="HAMP_Tb_calib"+self.cfg_dict["Flight_Dates_used"].keys()[0]+\
        #        "_"+str(self.cfg_dict["Flight_Dates_used"].values[0])+".png"        
        #else:
        #    fig_name="HAMP_Tb_calib"+self.cfg_dict["Flight_Dates_used"].keys()[0]+\
        #            "_"+str(self.cfg_dict["Flight_Dates_used"].values[0])+"_"+str(hourly)+"00.png"        
        #    
        #if np.isnan(hourly):
        #    axs[1,1].xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0]))        
        #else:
        #    axs[1,1].xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,15,30,45]))
        
        #fig.savefig(self.radiometer_fig_path+fig_name,
        #            dpi=300,bbox_inches="tight")
        #print("Figure saved as : ",self.radiometer_fig_path+fig_name)        
        #return None
#%% Radar Quicklook       
class Radar_Quicklook(Quicklook_Plotter):
    def __init__(self,cfg_dict):
        super(Radar_Quicklook,self).__init__(cfg_dict)
        self.specifier()
        self.dbz_colorbar="Spectral_r"
        
    def specifier(self):    
        self.instrument="radar"
        self.specify_plot_path()
        
    def plot_raw_radar_quicklook(self,raw_radar_ds,cut_data=False):
        if cut_data:
            raw_radar_ds=raw_radar_ds
        levels=np.arange(-50,55.0,5.0)
        ldr_levels=np.arange(-80,0,1.0)#

        raw_radar_fig=plt.figure(figsize=(12,9))
        ###############################################################
        ## Reflectivity
        ax1=raw_radar_fig.add_subplot(311)
        C1=ax1.pcolormesh(pd.DatetimeIndex(np.array(raw_radar_ds.time[:])),
            np.array(raw_radar_ds.range[:])/1000,
            np.array(raw_radar_ds["dBZg"][:]),
            cmap=self.dbz_colorbar,vmin=-40,vmax=30) #
        cax=raw_radar_fig.add_axes([0.9, 0.675, 0.01, 0.2])

        cb = plt.colorbar(C1,cax=cax,
                    orientation='vertical',
                    extend="both")
        cb.set_label('Radar Reflectivity (dBZ)',fontsize=14)
        
        ax1.set_ylim([0,15])
        ax1.invert_yaxis()
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))        
        ax1.set_xticklabels('')
        ax1.set_ylabel("Range (km)")
        for axis in ['bottom','left']:
            ax1.spines[axis].set_linewidth(2)
            ax1.tick_params(width=2)
        #################################################################
        ## Linear Depolarization Ratio
        ####
        #  LDR is not given in db equivalent to Z
        #  so calc the db values of LDR
        from measurement_instruments_ql import HALO_Devices, RADAR 
        HALO_Devices_cls=HALO_Devices(self.cfg_dict)
        RADAR_cls=RADAR(HALO_Devices_cls)
        RADAR_cls.raw_radar_ds=raw_radar_ds.copy()
        del raw_radar_ds
        RADAR_cls.calc_db_from_ldr(raw_measurement=True)
        raw_radar_ds=RADAR_cls.raw_radar_ds
        ###
        ax2=raw_radar_fig.add_subplot(312)
        C2=ax2.pcolormesh(pd.DatetimeIndex(np.array(raw_radar_ds.time[:])),
                      np.array(raw_radar_ds.range[:])/1000,
                      np.array(raw_radar_ds["LDRg"][:]),
                      cmap=cmaeri.batlow,vmin=-30,vmax=5)
        cax2=raw_radar_fig.add_axes([0.9, 0.4, 0.01, 0.2])

        cb2 = plt.colorbar(C2,cax=cax2,
                    orientation='vertical',
                    extend="both")
        cb2.set_label('Linear Depolarization \n Ratio (dB)',fontsize=14)
        ax2.set_ylim([0,15])
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))        
        ax2.set_xticklabels('')
        ax2.set_ylabel("Range (km)")
        ax2.invert_yaxis()
        for axis in ['bottom','left']:
            ax2.spines[axis].set_linewidth(2)
            ax2.tick_params(width=2)
        #################################################################
        ## Doppler Velocity
        ax3=raw_radar_fig.add_subplot(313)
        C3=ax3.pcolormesh(pd.DatetimeIndex(np.array(raw_radar_ds.time[:])),
                      np.array(raw_radar_ds.range[:])/1000,
                      np.array(raw_radar_ds["VELg"][:]).T,
                      cmap=cmaeri.bam_r,vmin=-15, vmax=5)
        cax3=raw_radar_fig.add_axes([0.9, 0.125, 0.01, 0.2])

        cb3 = plt.colorbar(C3,cax=cax3,
                    orientation='vertical',
                    extend="both")
        cb3.set_label('Doppler Velocity (m/s)',fontsize=14)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))        
        ax3.set_ylim([0,15])
        ax3.invert_yaxis()
        ax3.set_ylabel("Range (km)")
        ax3.set_xlabel("Time (UTC)")
        # change spine/tick widths
        for axis in ['top','bottom','left','right']:
            ax3.spines[axis].set_linewidth(2)
            ax3.tick_params(width=2)
        sns.despine(offset=10)
        fig_name="raw_radar_quicklook_"+\
                    str([*self.cfg_dict["Flight_Dates_used"]][0])+".png"
        raw_radar_fig.savefig(self.radar_fig_path+fig_name,
                    dpi=300,bbox_inches="tight")
        print("Figure saved as:",self.radar_fig_path+fig_name)
        return None
    def plot_radar_quicklook(self,radar_ds,flight_report=True,low_level=False,
                             add_sondes=False):
        """
        This function creates a quicklook of unprocessed radar reflectivity 
        measurements onto unified grid and processed radar data.
        
        Parameters
        ----------
        ds : dict
            Dictionary containing all radar variables relevant for analysis.

        Returns
        -------
        ds : Updated input radar dataset.

        """
        radar_vars=["dBZg","LDRg"]
        #self.processed_radar=radar_ds.copy()
        # 
        # uni_height=np.array(self.processed_radar.height[:])
        # uni_time=pd.DatetimeIndex(np.array(self.processed_radar.time[:]))
        #raw_uni_radar={}
        # radar_ds=ds.copy()
        
        # Load unified but unprocessed radar dat
        #radar_files=glob.glob(self.cfg_dict["device_data_path"]+"radar_mira/*"+\
        #                      str(self.cfg_dict["Flight_Dates_used"][0])+"*")
        #if not len(radar_files)==0:
            # Take newest version
        #    version_numbers=[float(radar_file[-6:-3]) \
        #                     for radar_file in radar_files]                    
        #    newest_file=np.array(version_numbers).argmax()    
        
        #    radar_ds=xr.open_dataset(radar_files[newest_file])
        
        #radar_time      = pd.DatetimeIndex(np.array(radar_ds.time[:]))
        #radar_height    = pd.Series(np.array(radar_ds.height[:]))
        #radar_state     = pd.Series(data=np.array(radar_ds.grst[:]),
        #                            index=radar_time)
        #for var in radar_vars:
        #    raw_uni_radar[var]=pd.DataFrame()
        #    print(var)
        #    raw_radar_df=pd.DataFrame(data=np.array(radar_ds[var][:]),
        #                              columns=radar_height,
        #                              index=radar_time)

            # Discard data where radar state was not 13; 
            # i.e. local oscillator not locked and/or radiation off
        #    raw_radar_df.loc[radar_state[radar_state!=13].index]=np.nan
            
        #    raw_uni_radar[var]=raw_radar_df.iloc[:,0:radar_height.shape[0]]
        #    raw_uni_radar[var]=raw_uni_radar[var].reindex(radar_time)
        if flight_report:
            self.raw_radar=radar_ds
            
            
        #self.radar_cfad_processing_comparison()
        if not flight_report:
            self.radar_reflectivity_quicklook()
            for hour in np.unique(radar_ds.time.dt.hour.values):
                self.radar_reflectivity_quicklook(hourly=hour)
        else:
            self.plot_radar_attcorr_flight_report_quicklook(radar_vars=radar_vars,
                                                            low_level=low_level,
                                                            add_sondes=add_sondes)
    
    def processed_radar_quicklook(self,hourly=np.nan,is_calibrated=False,
                                  show_masks=False):
        # Now raw_uni_radar and ds (processed uni radar) can be compared
        # via plotting
        from matplotlib import colors
        if show_masks:
            fig,axs=plt.subplots(4,1,figsize=(12,20),sharex=True)
        else:
            fig,axs=plt.subplots(3,1,figsize=(12,20),sharex=True)
        y=np.array(self.processed_radar["height"][:])#/1000
        if not is_calibrated:
            print("Plotting HAMP Cloud Radar (processed)")
        else:
            print("Plotting HAMP Cloud Radar (processed and calibrated)")
        #######################################################################
        #######################################################################
        ### Processed radar
        # Radar reflectivity
        processed_radar=replace_fill_and_missing_values_to_nan(
                                self.processed_radar,["dBZg","LDRg","VELg",
                                                      "radar_flag"])
        
        if not np.isnan(hourly):
            processed_radar["dBZg"]=processed_radar["dBZg"][\
                                        processed_radar["dBZg"].\
                                            time.dt.hour==hourly]
            
            processed_radar["LDRg"]=processed_radar["LDRg"][\
                                        processed_radar["LDRg"].\
                                            time.dt.hour==hourly]
            processed_radar["VELg"]=processed_radar["VELg"][\
                                        processed_radar["VELg"].\
                                            time.dt.hour==hourly]
        
        time=pd.DatetimeIndex(np.array(processed_radar["dBZg"].time[:]))
        print("dBZ plotted")
        #Plotting
        C1=axs[0].pcolor(time,y,
                             np.array(processed_radar["dBZg"][:]).T,
                             vmin=-30,vmax=30,shading='nearest')
        cax1=fig.add_axes([0.9, 0.725, 0.01, 0.15])

        cb = plt.colorbar(C1,cax=cax1,
                          orientation='vertical',
                          extend="both")
        cb.set_label('Reflectivity (dBZ)')
        
        axs[0].set_title("Processed radar")
        if is_calibrated:
            axs[0].set_title("Calibrated processed radar")
        axs[0].set_xlabel('')
        axs[0].set_yticks([0,2000,4000,6000,8000,10000,12000])
        axs[0].set_ylim([0,12000])
        axs[0].set_yticklabels(["0","2","4","6","8","10","12"])
        axs[0].set_ylabel("Altitude (km)")
        
        # Radar LDR
        print("LDR plotted")
        C2=axs[1].pcolor(time,y,np.array(processed_radar["LDRg"][:].T),
                         cmap="cubehelix_r",vmin=-50, vmax=5,shading='nearest')        
        axs[1].set_yticks([0,2000,4000,6000,8000,10000,12000])
        axs[1].set_ylim([0,12000])
        axs[1].set_yticklabels(["0","2","4","6","8","10","12"])
        
        for label in axs[1].get_xticklabels():
            label.set_rotation(30)
        if not np.isnan(hourly):
            axs[1].xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0]))        
        else:
            axs[1].xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,15,30,45]))
        axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))        
        sns.despine(offset=10)
        cax2=fig.add_axes([0.9, 0.525, 0.01, 0.15])

        cb = plt.colorbar(C2,cax=cax2,
                          orientation='vertical',
                          extend="both")
        cb.set_label('LDR (dB)')
        # Radar Doppler Velocity
        print("Doppler vel")
        
        #C3=axs[2].pcolor(pd.DatetimeIndex(np.array(processed_radar.time[:])),
        #                 y,processed_radar["VELg"].T,
        #                 cmap="PuOr",vmin=-15, vmax=15)
        #cax3=fig.add_axes([0.9, 0.325, 0.01, 0.15])#

        #cb3 = plt.colorbar(C3,cax=cax3,
        #            orientation='vertical',
        #            extend="both")
        #cb3.set_label('Doppler Velocity (m/s)')
        #axs[2].set_yticks([0,2000,4000,6000,8000,10000,12000])
        #axs[2].set_ylim([0,12000])
        #axs[2].set_yticklabels(["0","2","4","6","8","10","12"])
        
        #for label in axs[2].get_xticklabels():
        #    label.set_rotation(30)
        #if np.isnan(hourly):
        #    axs[2].xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0]))        
        #else:
        #    axs[2].xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,15,30,45]))
        #axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))        
        
        #axs[2].set_xlabel('Time (UTC)')
        if show_masks:
            print("mask")
            # define colormap
            cmap_mask = colors.ListedColormap(['green','orange','black','blue',
                                           'magenta'])
            bounds=[.5,1.5,2.5,3.5,4.5]
            norm = colors.BoundaryNorm(bounds, cmap_mask.N)
        
            C4=axs[3].contourf(time,y,np.array(
                            processed_radar["radar_flag"][:].T).astype(float),
                            levels=np.arange(0.5,4.5),cmap=cmap_mask,norm=norm)
            cax4=fig.add_axes([0.9, 0.125, 0.01, 0.15])
            cb4=plt.colorbar(C4,cax=cax4,orientation="vertical")
            cb4.set_label("Radar mask")
            cb4.set_ticks([0,1,2,3,4])
            cb4.ax.set_yticklabels(["good",
                                "noise",
                                "sfc",
                                "sea",
                                "Side\nLobes"])
        
            axs[3].set_yticks([0,2000,4000,6000,8000,10000,12000])
            axs[3].set_ylim([0,12000])
            axs[3].set_yticklabels(["0","2","4","6","8","10","12"])
        
            for label in axs[3].get_xticklabels():
                label.set_rotation(30)
            if np.isnan(hourly):
                axs[3].xaxis.set_major_locator(
                    mdates.MinuteLocator(byminute=[0]))        
            else:
                axs[3].xaxis.set_major_locator(
                    mdates.MinuteLocator(byminute=[0,15,30,45]))
                axs[3].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))        
        
                axs[3].set_xlabel('Time (UTC)')
        # no mask
        else:
            for label in axs[2].get_xticklabels():
                label.set_rotation(30)
            if np.isnan(hourly):
                axs[2].xaxis.set_major_locator(
                    mdates.MinuteLocator(byminute=[0]))        
            else:
                axs[2].xaxis.set_major_locator(
                    mdates.MinuteLocator(byminute=[0,15,30,45]))
                axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))        
        
                axs[2].set_xlabel('Time (UTC)')
        
        sns.despine(offset=10)
        
        if not hasattr(self,"radar_fig_path"):
            self.specify_plot_path()
        if np.isnan(hourly):
            fig_name="processed_radar_quicklook_"+\
                    str([*self.cfg_dict["flight_date_used"]][0])+".png"
        else:
            fig_name="processed_radar_quicklook_"+\
                    str([*self.cfg_dict["flight_date_used"]][0])+"_"+\
                        str(hourly)+"00.png"
        if is_calibrated:
            fig_name="calibrated_"+fig_name
        fig.savefig(self.radar_fig_path+fig_name,bbox_inches="tight",dpi=300)
        print("Figure saved as:",self.radar_fig_path+fig_name)
    
    def unified_radar_quicklook(self,flight,plot_path, calibrated_radar=True):
        # copied from HALO-(AC)³ Budget closure
        #import matplotlib.pyplot as plt
        #import seaborn as sns
        processed_radar=self.processed_radar
        time_var=np.array(processed_radar.time[:])        
        time_extent = measurement_instruments_ql.\
                        HALO_Devices.numpydatetime64_to_datetime(time_var[-1]) -\
                    measurement_instruments_ql.\
                        HALO_Devices.numpydatetime64_to_datetime(time_var[0])

        fig,axs=plt.subplots(4,1,figsize=(14,12),
                             gridspec_kw=dict(height_ratios=(1,0.04,1,0.1),
                             hspace=0.2),sharex=True)
        y=np.array(processed_radar["height"][:])
        statement="Plotting HAMP Cloud Radar (processed"
        if not calibrated_radar: statement+=")"
        else: statement+=" and calibrated)"
        print(statement)
        matplotlib.rcParams['axes.linewidth'] = 2
        #######################################################################
        #######################################################################
        ### Processed radar
        surface_Zg=processed_radar["Zg"][:,4]
        surface_Zg=surface_Zg.where(surface_Zg!=-888.)
        
        #rain_rate=get_rain_rate(surface_Zg)
        #sys.exit()
        #processed_radar
        time=pd.DatetimeIndex(np.array(processed_radar["dBZg"].time[:]))
        #Plotting
        C1=axs[0].pcolormesh(time,y,np.array(processed_radar["dBZg"][:]).T,
                         cmap=cmaeri.roma_r,vmin=-30,vmax=30)
        print("dBZ plotted")
        
        cax1=fig.add_axes([0.95, 0.625, 0.01, 0.15])
        cb = plt.colorbar(C1,cax=cax1,orientation='vertical',extend="both")
        cb.set_label('Reflectivity (dBZ)')
        axs[0].set_xlabel('')
        axs[0].set_yticks([0,2000,4000,6000,8000,10000,12000])
        axs[0].set_ylim([0,12000])
        axs[0].set_yticklabels(["0","2","4","6","8","10","12"])
        axs[0].set_xticklabels([])
        axs[0].set_ylabel("Height (km)")
        for axis in ['bottom','left']:
            axs[0].spines[axis].set_linewidth(2)
            axs[0].tick_params(width=2,length=6)
        axs[0].spines.right.set_visible(False)
        axs[0].spines.top.set_visible(False)
        #axs[0].spines.bottom.set_bounds()
        #axs[0].spines.left.set_bounds(0,12000)
        axs[1].tick_params(left=False,bottom=False)
        axs[1].set_yticklabels([])
        axs[1].set_xticklabels([])
        axs[1].spines.right.set_visible(False)
        axs[1].spines.top.set_visible(False)
        axs[1].spines.left.set_visible(False)
        axs[1].spines.bottom.set_visible(False)
        sns.despine(offset=10,ax=axs[0])
        # Radar LDR
        C2=axs[2].pcolor(time,y,np.array(processed_radar["LDRg"][:].T),cmap=cmaeri.batlowK,vmin=-25, vmax=-10)        
        axs[2].set_yticks([0,2000,4000,6000,8000,10000,12000])
        axs[2].set_ylim([0,12000])
        axs[2].set_yticklabels(["0","2","4","6","8","10","12"])
        print("LDR plotted")
        axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))        
        axs[2].set_xticklabels([])
        axs[2].set_ylabel("Height (km)")
        for axis in ['bottom','left']:
            axs[2].spines[axis].set_linewidth(2)
            axs[2].tick_params(width=2,length=6)
        
        cax2=fig.add_axes([0.95, 0.25, 0.01, 0.15])
    
        cb = plt.colorbar(C2,cax=cax2,orientation='vertical',extend="both")
        cb.set_label('LDR (dB)')
        sns.despine(offset=10,ax=axs[2])
        
        fs = 14
        fs_small = fs - 2
        fs_dwarf = fs - 4
        marker_size = 15
            
        bah_df=pd.DataFrame()
        bah_df["sea_ice"]=pd.Series(data=np.array(processed_radar["radar_flag"][:,0]),
                            index=pd.DatetimeIndex(np.array(processed_radar.time[:])))

        blue_colorbar=cm.get_cmap('Blues_r', 22)
        blue_cb=blue_colorbar(np.linspace(0, 1, 22))
        brown_rgb = np.array(colors.hex2color(colors.cnames['brown']))
        blue_cb[:2, :] = [*brown_rgb,1]
        newcmp = ListedColormap(blue_cb)
        im = axs[3].pcolor(
            np.array([pd.DatetimeIndex(bah_df["sea_ice"].index),
                      pd.DatetimeIndex(bah_df["sea_ice"].index)]),
            np.array([0, 1]),
            np.tile(np.array(bah_df["sea_ice"].values),(2,1)),
            #np.array([bah_df["sea_ice"].values/100]),
            cmap=newcmp, vmin=-0.1, vmax=1,shading='nearest')
        axs[3].set_yticklabels([])
        for axx in axs: 
            #axx.set_visible({"top":False,"right":False})
            axx = no_pd_dt_set_xticks_and_xlabels(ax=axx,
                                time_extend=time_extent)
        #axs[3].
        cax = fig.add_axes([0.7, 0.025, 0.1, axs[3].get_position().height])
        sns.despine(ax=cax)
        C1=fig.colorbar(im, cax=cax, orientation='horizontal')
        C1.set_label(label='Sea ice [%]',fontsize=fs_small)
        C1.ax.tick_params(labelsize=fs_small,width=2)
        
        axs[3].tick_params(axis='x', labelleft=False, 
                           left=False,labelsize=fs)
        axs[3].tick_params(axis='y', labelleft=False, left=False,
                           width=2,length=6)
        axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axs[3].set_xlabel("Time (HH:MM)",fontsize=fs_small)
        #time_var=bah_df.index
        # time extent:
            #time_extent = Measurement_Instruments_QL.\
                #                        HALO_Devices.\
                    #                            numpydatetime64_to_datetime(time_var[-1]) -\
                        #                        Measurement_Instruments_QL.\
                            #                            HALO_Devices.numpydatetime64_to_datetime(time_var[0])        
                            #axs[3].set_xlabel(f"Time (HH:MM) of {str(time_var[round(len(time_var)/2)].values)[:10]}",
                            #                         fontsize=fs_small)

        # Limit axis spacing:
        plt.subplots_adjust(hspace=0.35)			# removes space between subplots
            
        #box = axs[3].get_position()        
            
        #box.y0 = box.y0 + 0.025
        #box.y1 = box.y1 + 0.025        
        #axs[3]=axs[3].set_position(box)
        radar_str="processed_radar"
        
        if calibrated_radar: radar_str="calibrated and "+radar_str
        fig_name="unified_radar_dbz_ldr_quicklook_"+\
            flight+"_"+self.cfg_dict["date"]+".png"
        print(fig_name)
        plot_path+="/unified_dataset/"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        fig.savefig(plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as:",plot_path+fig_name)
    
    def plot_only_precip_rates(self,halo_era5,halo_icon_hmp,
                               precipitation_rate,ar_of_day,
                               inflow_times,internal_times,outflow_times,
                               radar_str,reflectivity_for_snow="Ze"):
        precip_rate_fig=plt.figure(figsize=(16,9))
        ax1=precip_rate_fig.add_subplot(111)
        ax1.plot(halo_era5["Interp_Precip"],lw=2,ls="--",color="k",label="ERA5:"+\
            str(round(float(halo_era5["Interp_Precip"].mean()),2)),zorder=5)
        ax1.plot(halo_icon_hmp["Interp_Precip"],lw=3,ls="-",color="k",label="ICON-NWP:"+\
            str(round(float(halo_icon_hmp["Interp_Precip"].mean()),2)),zorder=6)
        ax1.plot(halo_icon_hmp["Interp_Precip"],lw=2,ls="--",color="w",zorder=7)
        ax1.plot(precipitation_rate["mean_rain"],lw=3,color="darkgreen",
            label="avg_r:"+str(round(float(precipitation_rate["mean_rain"].mean()),2)))
        ax1.plot(precipitation_rate["r_norris"],lw=1,color="lightgreen",
            label="nor:"+str(round(float(precipitation_rate["r_norris"].mean()),2)))
        ax1.plot(precipitation_rate["r_palmer"],lw=1,color="mediumseagreen",
            label="pal:"+str(round(float(precipitation_rate["r_palmer"].mean()),2)))
        ax1.plot(precipitation_rate["r_chandra"],lw=1,color="green",
            label="cha:"+str(round(float(precipitation_rate["r_chandra"].mean()),2)))
        # snow 
        ax1.plot(precipitation_rate["mean_snow"],lw=3,color="darkblue",
            label="avg_s:"+str(round(float(precipitation_rate["mean_snow"].mean()),2)))
        ax1.plot(precipitation_rate["s_schoger"],lw=0.5,color="lightblue",
            label="sch:"+str(round(float(precipitation_rate["s_schoger"].mean()),2)))
        ax1.plot(precipitation_rate["s_matrosov"],lw=0.5,color="blue",
            label="mat:"+str(round(float(precipitation_rate["s_matrosov"].mean()),2)))
        ax1.plot(precipitation_rate["s_heymsfield"],lw=0.5,color="cadetblue",
            label="hey:"+str(round(float(precipitation_rate["s_heymsfield"].mean()),2)))

        if inflow_times[0]<outflow_times[-1]:
            ax1.axvspan(pd.Timestamp(inflow_times[-1]),
               pd.Timestamp(internal_times[0]),
               alpha=0.5, color='grey')
            ax1.axvspan(pd.Timestamp(internal_times[-1]),
               pd.Timestamp(outflow_times[0]),
               alpha=0.5, color='grey')   
        else:
            ax1.axvspan(pd.Timestamp(outflow_times[-1]),
               pd.Timestamp(internal_times[0]),
               alpha=0.5, color='grey')
            ax1.axvspan(pd.Timestamp(internal_times[-1]),
               pd.Timestamp(inflow_times[0]),
               alpha=0.5, color='grey')   

        ax1.legend(loc="upper left",ncol=6)
        #ax1.set_xticks=axs[1].get_xticks()
        ax1.set_xticklabels([])
        ax1.set_ylabel("Precipitation\nrate ($\mathrm{mmh}^{-1}$)")
        fig_name=self.flight[0]+"_"+ar_of_day+"_Only_Rain_internal_"+\
            radar_str+"_"+reflectivity_for_snow+".png"
        precip_rate_fig.savefig(self.plot_path+fig_name,
                                dpi=300,bbox_inches="tight")
        print("Figure saved as:",self.plot_path+fig_name)
        
    def processed_radar_rain_rate_model_comparison(self,
                                  halo_era5,halo_icon_hmp,
                                  precipitation_rate,inflow_times,
                                  internal_times,outflow_times,ar_of_day,
                                  calibrated_radar=True,
                                  reflectivity_for_snow="Ze"):
        # Now raw_uni_radar and ds (processed uni radar) can be compared
        # via plotting
        font_size=12
        matplotlib.rcParams.update({"font.size":font_size})
        
        fig,axs=plt.subplots(4,1,figsize=(12,12),
                             gridspec_kw=dict(height_ratios=(1,1,1.0,0.1)),
                             sharex=True)
        processed_radar=self.processed_radar
        y=np.array(processed_radar["height"][:])
        statement="Plotting HAMP Cloud Radar (processed"
        if not calibrated_radar: statement+=")"
        else: statement+=" and calibrated)"
        print(statement)
        #######################################################################
        #######################################################################
        ### Processed radar
        print("flag nans")
        processed_radar["dBZg"]=processed_radar["dBZg"].where(
                            processed_radar["radar_flag"].isnull(), drop=True)
        processed_radar["Zg"]=processed_radar["Zg"].where(
                            processed_radar["radar_flag"].isnull(), drop=True)
        processed_radar["LDRg"]=processed_radar["LDRg"].where(
                            processed_radar["radar_flag"].isnull(), drop=True)
        print("flagging done")
        surface_Zg=processed_radar["Zg"][:,4]
        surface_Zg=surface_Zg.where(surface_Zg!=-888.)

        #processed_radar
        time=pd.DatetimeIndex(np.array(processed_radar["dBZg"].time[:]))
        #Plotting
        C1=axs[0].pcolor(time,y,np.array(processed_radar["dBZg"][:]).T,
                cmap=cmaeri.roma_r,vmin=-30,vmax=30)
        print("dBZ plotted")

        if inflow_times[0]<outflow_times[-1]:
            axs[0].axvspan(pd.Timestamp(inflow_times[-1]),
               pd.Timestamp(internal_times[0]),
               alpha=0.5, color='grey')
            axs[0].axvspan(pd.Timestamp(internal_times[-1]),
               pd.Timestamp(outflow_times[0]),
               alpha=0.5, color='grey')   
        else:
            axs[0].axvspan(pd.Timestamp(outflow_times[-1]),
               pd.Timestamp(internal_times[0]),
               alpha=0.5, color='grey')
            axs[0].axvspan(pd.Timestamp(internal_times[-1]),
               pd.Timestamp(inflow_times[0]),
               alpha=0.5, color='grey')   

        cax1=fig.add_axes([0.95, 0.725, 0.01, 0.15])
        cb = plt.colorbar(C1,cax=cax1,orientation='vertical',extend="both")
        cb.set_label('Reflectivity (dBZ)')
        title_str="Processed radar"
        if calibrated_radar: title_str+=" and calibrated"
        title_str+=" "+self.flight[0]+" "+ar_of_day

        axs[0].set_title(title_str)
        axs[0].set_xlabel('')
        axs[0].set_yticks([0,2000,4000,6000,8000,10000,12000])
        axs[0].set_ylim([0,12000])
        axs[0].set_yticklabels(["0","2","4","6","8","10","12"])
        axs[0].set_xticklabels([])
        axs[0].set_ylabel("Altitude (km)")
        axs[0].text(pd.Timestamp(inflow_times[0]),10000,"Inflow")
        axs[0].text(pd.Timestamp(internal_times[0]),10000,"Internal")
        axs[0].text(pd.Timestamp(outflow_times[0]),10000,"Outflow")
        axs[0].set_ylabel("Height (km)")

        # Radar LDR
        C2=axs[1].pcolor(time,y,np.array(processed_radar["LDRg"][:].T),
                         cmap=cmaeri.batlowK,vmin=-25, vmax=-10)        
        axs[1].set_yticks([0,2000,4000,6000,8000,10000,12000])
        axs[1].set_ylim([0,12000])
        axs[1].set_yticklabels(["0","2","4","6","8","10","12"])
        print("LDR plotted")
        if inflow_times[0]<outflow_times[-1]:
            axs[1].axvspan(pd.Timestamp(inflow_times[-1]),
                   pd.Timestamp(internal_times[0]),
                   alpha=0.5, color='grey')
            axs[1].axvspan(pd.Timestamp(internal_times[-1]),
                           pd.Timestamp(outflow_times[0]),
                           alpha=0.5, color='grey')   
        else:
            axs[1].axvspan(pd.Timestamp(outflow_times[-1]),
               pd.Timestamp(internal_times[0]),
               alpha=0.5, color='grey')
            axs[1].axvspan(pd.Timestamp(internal_times[-1]),
               pd.Timestamp(inflow_times[0]),
               alpha=0.5, color='grey')   
        axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))        
        axs[1].text(pd.Timestamp(inflow_times[0]),10000,"Inflow")
        axs[1].text(pd.Timestamp(internal_times[0]),10000,"Internal")
        axs[1].text(pd.Timestamp(outflow_times[0]),10000,"Outflow")
        axs[1].set_xticklabels([])
        axs[1].set_ylabel("Height (km)")

        cax2=fig.add_axes([0.95, 0.5, 0.01, 0.15])
        cb = plt.colorbar(C2,cax=cax2,orientation='vertical',extend="both")
        cb.set_label('LDR (dB)')
        sns.despine(offset=10)
        axs[2].plot(halo_era5["Interp_Precip"],lw=2,ls="--",
                    color="k",label="ERA5:"+\
                    str(round(float(halo_era5["Interp_Precip"].mean()),2)),
                    zorder=5)
        axs[2].plot(halo_icon_hmp["Interp_Precip"],lw=3,ls="-",
                    color="k",label="ICON-NWP:"+\
                    str(round(float(halo_icon_hmp["Interp_Precip"].mean()),2)),
                    zorder=6)
        axs[2].plot(halo_icon_hmp["Interp_Precip"],lw=2,ls="--",
                    color="w",zorder=7)
        axs[2].plot(precipitation_rate["mean_rain"],lw=3,color="darkgreen",
                    label="avg_r:"+str(round(float(\
                        precipitation_rate["mean_rain"].mean()),2)))
        axs[2].plot(precipitation_rate["r_norris"],lw=1,color="lightgreen",
                    label="nor:"+str(round(float(\
                    precipitation_rate["r_norris"].mean()),2)))
        
        axs[2].plot(precipitation_rate["r_palmer"],lw=1,color="mediumseagreen",
            label="pal:"+str(round(float(\
            precipitation_rate["r_palmer"].mean()),2)))
        axs[2].plot(precipitation_rate["r_chandra"],lw=1,color="green",
            label="cha:"+str(round(float(\
            precipitation_rate["r_chandra"].mean()),2)))
        # snow 
        axs[2].plot(precipitation_rate["mean_snow"],lw=3,color="darkblue",
                    label="avg_s:"+str(round(float(\
                    precipitation_rate["mean_snow"].mean()),2)))
        axs[2].plot(precipitation_rate["s_schoger"],lw=0.5,color="lightblue",
                    label="sch:"+str(round(float(\
                    precipitation_rate["s_schoger"].mean()),2)))
        axs[2].plot(precipitation_rate["s_matrosov"],lw=0.5,color="blue",
                    label="mat:"+str(round(float(\
                    precipitation_rate["s_matrosov"].mean()),2)))
        axs[2].plot(precipitation_rate["s_heymsfield"],lw=0.5,color="cadetblue",
            label="hey:"+str(round(float(precipitation_rate["s_heymsfield"].mean()),2)))

        if inflow_times[0]<outflow_times[-1]:
            axs[2].axvspan(pd.Timestamp(inflow_times[-1]),
               pd.Timestamp(internal_times[0]),
               alpha=0.5, color='grey')
            axs[2].axvspan(pd.Timestamp(internal_times[-1]),
                   pd.Timestamp(outflow_times[0]),
                   alpha=0.5, color='grey')   
        else:
            axs[2].axvspan(pd.Timestamp(outflow_times[-1]),
               pd.Timestamp(internal_times[0]),
               alpha=0.5, color='grey')
            axs[2].axvspan(pd.Timestamp(internal_times[-1]),
               pd.Timestamp(inflow_times[0]),
               alpha=0.5, color='grey')   

        axs[2].set_ylim([0,1.0])
        axs[2].legend(loc="upper left",ncol=6,fontsize=font_size-2)
        axs[2].set_xticks=axs[1].get_xticks()
        axs[2].set_xticklabels([])
        axs[2].set_ylabel("Precipitation\nrate ($\mathrm{mmh}^{-1}$)")
        # Add surface mask
        # plot AMSR2 sea ice concentration
        fs = 14
        fs_small = fs - 2
        fs_dwarf = fs - 4
        marker_size = 15
            
        bah_df=pd.DataFrame()
        bah_df["sea_ice"]=pd.Series(data=np.array(\
                            processed_radar["radar_flag"][:,0]),
                            index=pd.DatetimeIndex(np.array(\
                                processed_radar.time[:])))
        blue_colorbar=cm.get_cmap('Blues_r', 22)
        blue_cb=blue_colorbar(np.linspace(0, 1, 22))
        brown_rgb = np.array(colors.hex2color(colors.cnames['brown']))
        blue_cb[:2, :] = [*brown_rgb,1]
        newcmp = ListedColormap(blue_cb)
        im = axs[3].pcolormesh(np.array([\
                pd.DatetimeIndex(bah_df["sea_ice"].index),
                pd.DatetimeIndex(bah_df["sea_ice"].index)]),
                np.array([0, 1]),
                np.tile(np.array(bah_df["sea_ice"].values),(2,1)),
                #np.array([bah_df["sea_ice"].values/100,bah_df["sea_ice"].values/100]),
                cmap=newcmp, vmin=-0.1, vmax=1,
                shading='auto')

        cax = fig.add_axes([0.7, 0.06, 0.1, axs[3].get_position().height])
        C1=fig.colorbar(im, cax=cax, orientation='horizontal')
        C1.set_label(label='Sea ice [%]',fontsize=fs_small)
        C1.ax.tick_params(labelsize=fs_small)
        axs[3].tick_params(axis='x', labelleft=False, 
                           left=False,labelsize=fs_small)
        axs[3].tick_params(axis='y', labelleft=False, left=False)
        axs[3].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

        #time_var=bah_df.index
        # time extent:
        #time_extent = Measurement_Instruments_QL.\
        #                        HALO_Devices.\
        #                            numpydatetime64_to_datetime(time_var[-1]) -\
        #                        Measurement_Instruments_QL.\
        #                            HALO_Devices.numpydatetime64_to_datetime(time_var[0])        
        #axs[3].set_xlabel(f"Time (HH:MM) of {str(time_var[round(len(time_var)/2)].values)[:10]}",
        #                         fontsize=fs_small)

        # Limit axis spacing:
        plt.subplots_adjust(hspace=0.35)			# removes space between subplots
        box = axs[3].get_position()        
        box.y0 = box.y0 + 0.025
        box.y1 = box.y1 + 0.025        
        axs[3]=axs[3].set_position(box)
        radar_str="processed_radar"
        #radar_var="zg"
        if calibrated_radar: radar_str="calibrated_and_"+radar_str
        fig_name=self.flight[0]+"_"+ar_of_day+"_Rain_internal_"+\
                    radar_str+"_"+reflectivity_for_snow+".png"
        fig.savefig(self.plot_path+fig_name,dpi=300,bbox_inches="tight")
        print("Figure saved as:",self.plot_path+fig_name)
        return radar_str
        
    def plot_radar_clutter_comparison(self, clutter_removal_version="0.2"):
        # this function plot two radar attitude corrected reflectivity values.
        # The one is without clutter removal and hence version 0.1 per default.
        # The second one needs to be a higher version number. The default is
        # 0.2.
        # Initialize classes
        from measurement_instruments_ql import HALO_Devices, RADAR 
        HALO_Devices_cls=HALO_Devices(self.cfg_dict)
        RADAR_cls=RADAR(HALO_Devices_cls)
        #RADAR_cls.raw_radar_ds=raw_radar_ds.copy()
        
        clutter_radar=RADAR_cls.open_version_specific_processed_radar_data(
            version="0.1")
        clean_radar=RADAR_cls.open_version_specific_processed_radar_data(
            version="0.2")
        
        fig,axs=plt.subplots(2,1,figsize=(18,12),sharex=True)
        y=np.array(clutter_radar["height"][:])#/1000
        #######################################################################
        #######################################################################
        ### Processed radar
        # Radar reflectivity
        clutter_radar=replace_fill_and_missing_values_to_nan(
                                clutter_radar,["dBZg"])
        
        clean_radar=replace_fill_and_missing_values_to_nan(
                                clean_radar,["dBZg"])
        
        time=pd.DatetimeIndex(np.array(clutter_radar["dBZg"].time[:]))
        #Plotting
        C1=axs[0].pcolormesh(time,y,
                             np.array(clutter_radar["dBZg"][:].T),
                             vmin=-30,vmax=30,cmap=self.dbz_colorbar)
        
        cax1=fig.add_axes([0.9, 0.55, 0.01, 0.35])

        cb = plt.colorbar(C1,cax=cax1)
                          #orientation='vertical',
                          #extend="both")
        cb.set_label('Reflectivity (dBZ)')
        
        axs[0].set_xlabel('')
        axs[0].set_yticks([0,2000,4000,6000,8000,10000,12000])
        axs[0].set_ylim([0,12000])
        axs[0].set_yticklabels(["0","2","4","6","8","10","12"])
        axs[0].set_ylabel("Altitude (km)")
        
        C2=axs[1].pcolormesh(time,y,np.array(clean_radar["dBZg"][:].T),
                         cmap=self.dbz_colorbar,vmin=-30, vmax=30)        
        axs[1].set_yticks([0,2000,4000,6000,8000,10000,12000])
        axs[1].set_ylim([0,12000])
        axs[1].set_yticklabels(["0","2","4","6","8","10","12"])
        for label in axs[1].get_xticklabels():
            label.set_rotation(30)
        #if not np.isnan(hourly):
        axs[1].xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0]))        
        #else:
     #   axs[1].xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,15,30,45]))
        axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))        
        sns.despine(offset=10)
        cax2=fig.add_axes([0.9, 0.125, 0.01, 0.35])

        cb2 = plt.colorbar(C2,cax=cax2)
        cb2.set_label('Reflectivity (dBZ)')
        
        
    def plot_radar_attcorr_flight_report_quicklook(self,radar_vars=[],
                                                   hourly=np.nan,
                                                   low_level=False,add_sondes=False):
        self.processed_radar=self.raw_radar.copy()
        fig,axs=plt.subplots(2,1,figsize=(24,16),sharex=True)
            
        y=np.array(self.processed_radar.height[:])
        #if not is_calibrated:
        #    print("Plotting HAMP Cloud Radar (processed)")
        #else:
        #    print("Plotting HAMP Cloud Radar (processed and calibrated)")
        #######################################################################
        #######################################################################
        ### Processed radar
        # Radar reflectivity
        levels=np.arange(-30,30.0,1.0)
        ldr_levels=np.arange(-50,0,1.0)#
        #self.raw_radar["dBZg"]=xr.where(self.processed_radar["dBZg"]!=-888,
        #                                      self.processed_radar["dBZg"],
        #                                      np.nan)
        self.processed_radar["dBZg"]=xr.where(self.processed_radar["dBZg"]!=-999,
                                              self.processed_radar["dBZg"],
                                              np.nan)
        
        self.processed_radar["dBZg"]=xr.where(self.processed_radar["dBZg"]!=-2*888,
                                              self.processed_radar["dBZg"],
                                              np.nan)
        processed_radar={}
        processed_radar["dBZg"]=self.processed_radar["dBZg"]
        processed_radar["LDRg"]=self.processed_radar["LDRg"]
        
        if not np.isnan(hourly):
            processed_radar["dBZg"]=processed_radar["dBZg"][\
                                        processed_radar["dBZg"].\
                                            time.dt.hour==hourly]
            
            processed_radar["LDRg"]=processed_radar["LDRg"][\
                                        processed_radar["LDRg"].\
                                            time.dt.hour==hourly]
        
        time=pd.DatetimeIndex(np.array(processed_radar["dBZg"].time[:]))
        try:
            C1=axs[0].contourf(time,y,
                                 np.array(processed_radar["dBZg"][:]).T,
                               levels,extend="both")
        except:
            C1=axs[0].contourf(time,y,
                                 np.array(processed_radar["dBZg"][:]).T,
                               levels,extend="both")
                                      #                 extend="both")
        #axs[0].set_title("Processed radar")
        cax1=fig.add_axes([0.9, 0.55, 0.01, 0.325])
        cb = plt.colorbar(C1,cax=cax1,
                          orientation='vertical')
        cb.set_label("Reflectivity (dBZ)")
        if add_sondes:
            from datetime import datetime
            date=pd.DatetimeIndex(np.array(processed_radar["dBZg"].time[:])).date[0]
            sondes_time=add_sondes_in_plots(str(date).replace("-",""))
            sondes_time_hours=[int(sonde) for sonde in sondes_time]
            sondes_time_minutes=[int(sonde %1*60) for sonde in sondes_time]
            sondes_time_seconds=[int(sonde %1*60 %1*60) for sonde in sondes_time]
            sonde_time_index=pd.DatetimeIndex([datetime(2020,2,5,sondes_time_hours[i],
                                      sondes_time_minutes[i],
                                      sondes_time_seconds[i]) \
                             for i in range(len(sondes_time_hours))])
            for xc in sonde_time_index:
                axs[0].axvline(x=xc, color = 'gray', lw= '1.0',ls="--")
            
            axs[0].axvline(xc, color = 'gray', lw = '1.0',ls="--",
                           label = 'Dropsondes')    

        axs[0].set_xlabel('')
        if not low_level:
            axs[0].set_ylim([0,12000])
            axs[0].set_yticks([0,2000,4000,6000,8000,10000,12000])
            axs[0].set_yticklabels(["0","2","4","6","8","10","12"])
        
        else:
            axs[0].set_ylim([0,5000])
            axs[0].set_yticks([0,1000,2000,3000,4000,5000])
            axs[0].set_yticklabels(["0","1","2","3","4","5"])
        
        axs[0].set_ylabel("Altitude (km)")
        for axis in ['top','bottom','left','right']:
            axs[0].spines[axis].set_linewidth(2)
            axs[0].tick_params(width=2)
            
        #for label in axs[0,0].xaxis.get_ticklabels()[::8]:
        #    label.set_visible(False)
        
        
        # Radar LDR
        C2=axs[1].contourf(time,y,
                           np.array(processed_radar["LDRg"][:]).T,
                           ldr_levels,cmap=cm.get_cmap(
                                           "cubehelix_r",
                                           len(ldr_levels)-1))
        
        if not low_level:
            axs[1].set_ylim([0,12000])
            axs[1].set_yticks([0,2000,4000,6000,8000,10000,12000])
            axs[1].set_yticklabels(["0","2","4","6","8","10","12"])
    
        else:
            axs[1].set_ylim([0,5000])
            axs[1].set_yticks([0,1000,2000,3000,4000,5000])
        
            axs[1].set_yticklabels(["0","1","2","3","4","5"])
        #for label in axs[1].get_xticklabels():
        #    label.set_rotation(30)
        if not np.isnan(hourly):
            axs[1].xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0]))        
        else:
            axs[1].xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,15,30,45]))
        axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))        
        for axis in ['top','bottom','left','right']:
            axs[1].spines[axis].set_linewidth(2)
            axs[1].tick_params(width=2)
        if add_sondes:
            for xc in sonde_time_index:
                axs[1].axvline(x=xc, color = 'gray', lw = '1.0',ls="--")
                axs[1].axvline(xc, color = 'gray', lw = '1.0',ls="--",
                           label = 'Dropsondes')    
        axs[1].set_ylabel("Altitude (km)")
        axs[1].set_xlabel('Time (UTC)')
        cax2=fig.add_axes([0.9, 0.125, 0.01, 0.325])
        cb = plt.colorbar(C2,cax=cax2,
                          orientation='vertical')
        cb.set_ticks(np.linspace(-50,0,6))
        cb.set_label('LDR (dB)')
        sns.despine(offset=10)
        axs[0].legend(loc="upper right")
        fig_name="Radar_fig.png"
        if low_level:
            fig_name="low_level_"+fig_name
        plt.savefig(self.radar_fig_path+"Radar_fig.png",
                    bbox_inches="tight",
                    dpi=300)
        print("Figure saved as:",self.radar_fig_path+"Radar_fig.png")
    def radar_reflectivity_quicklook(self,hourly=np.nan,is_calibrated=False):
        
        # Now raw_uni_radar and ds (processed uni radar) can be compared
        # via plotting
        
        fig,axs=plt.subplots(2,2,figsize=(24,16),sharex=True)
            
        y=np.array(self.processed_radar["height"][:])#/1000
        if not is_calibrated:
            print("Plotting HAMP Cloud Radar (processed)")
        else:
            print("Plotting HAMP Cloud Radar (processed and calibrated)")
        #######################################################################
        #######################################################################
        ### Processed radar
        # Radar reflectivity
        levels=np.arange(-30,30.0,1.0)
        ldr_levels=np.arange(-80,0,1.0)#
        self.processed_radar["dBZg"]=xr.where(self.processed_radar["dBZg"]!=-888,
                                              self.processed_radar["dBZg"],
                                              np.nan)
        self.processed_radar["dBZg"]=xr.where(self.processed_radar["dBZg"]!=-999,
                                              self.processed_radar["dBZg"],
                                              np.nan)
        
        self.processed_radar["dBZg"]=xr.where(self.processed_radar["dBZg"]!=-2*888,
                                              self.processed_radar["dBZg"],
                                              np.nan)
        processed_radar={}
        processed_radar["dBZg"]=self.processed_radar["dBZg"]
        processed_radar["LDRg"]=self.processed_radar["LDRg"]
        
        if not np.isnan(hourly):
            processed_radar["dBZg"]=processed_radar["dBZg"][\
                                        processed_radar["dBZg"].\
                                            time.dt.hour==hourly]
            
            processed_radar["LDRg"]=processed_radar["LDRg"][\
                                        processed_radar["LDRg"].\
                                            time.dt.hour==hourly]
        
        time=pd.DatetimeIndex(np.array(processed_radar["dBZg"].time[:]))
        try:
            C1=axs[0,0].contourf(time,y,
                                 np.array(processed_radar["dBZg"][:]).T,
                               levels)
        except:
            C1=axs[0,0].contourf(time,y,
                                 np.array(processed_radar["dBZg"][:]).T,
                               levels)
                                      #                 extend="both")
        axs[0,0].set_title("Processed radar")
        axs[0,0].set_xlabel('')
        axs[0,0].set_yticks([0,2000,4000,6000,8000,10000,12000])
        axs[0,0].set_ylim([0,12000])
        axs[0,0].set_yticklabels(["0","2","4","6","8","10","12"])
        axs[0,0].set_ylabel("Altitude (km)")
            
        #for label in axs[0,0].xaxis.get_ticklabels()[::8]:
        #    label.set_visible(False)
        
        
        # Radar LDR
        C2=axs[1,0].contourf(time,y,
                           np.array(processed_radar["LDRg"][:]).T,
                           ldr_levels,cmap=cm.get_cmap(
                                           "cubehelix_r",
                                           len(ldr_levels)-1))
        
        axs[1,0].set_yticks([0,2000,4000,6000,8000,10000,12000])
        axs[1,0].set_ylim([0,12000])
        axs[1,0].set_yticklabels(["0","2","4","6","8","10","12"])
    
        for label in axs[1,0].get_xticklabels():
            label.set_rotation(30)
        if not np.isnan(hourly):
            axs[1,0].xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0]))        
        else:
            axs[1,0].xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,15,30,45]))
        axs[1,0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))        
        axs[1,0].set_xlabel('Time (UTC)')
        sns.despine(offset=10)
        #######################################################################
        #######################################################################
        # raw radar
        raw_radar={}
        raw_radar["dBZg"]=self.raw_radar["dBZg"]        
        raw_radar["LDRg"]=self.raw_radar["LDRg"]        
        if not np.isnan(hourly):
            #raw_radar=self.raw_radar[self.raw_radar.index.hour==hourly]
            raw_radar["dBZg"]=raw_radar["dBZg"][\
                                    raw_radar["dBZg"].index.hour==hourly]
            raw_radar["LDRg"]=raw_radar["LDRg"][\
                                raw_radar["LDRg"].index.hour==hourly]
        
        time_raw=raw_radar["dBZg"].index
        
        try:
            C1=axs[0,1].contourf(time_raw,y,
                                 raw_radar["dBZg"][:].T,
                               levels)
        except:
            C1=axs[0,1].contourf(time_raw,y,
                                 raw_radar["dBZg"][:].T,
                               levels)
        axs[0,1].set_title("Raw Radar")
        axs[0,1].set_xlabel('')
        axs[0,1].set_yticks([0,2000,4000,6000,8000,10000,12000])
        axs[0,1].set_ylim([0,12000])
        axs[0,1].set_yticklabels(["0","2","4","6","8","10","12"])
        axs[0,1].set_ylabel("Altitude (km)")
            
        #for label in axs[0,1].xaxis.get_ticklabels()[::8]:
        #    label.set_visible(False)
        
        cax=fig.add_axes([0.9, 0.55, 0.01, 0.3])
        
        cb = plt.colorbar(C1,cax=cax,
                          orientation='vertical',
                          extend="both")
        cb.set_label('Radar Reflectivity (dBZ)')
        
        # Radar LDR
        C2=axs[1,1].contourf(time_raw,y,
                           raw_radar["LDRg"][:].T,
                           ldr_levels,cmap=cm.get_cmap("cubehelix_r",
                                                       len(ldr_levels)-1))
        
        cax2=fig.add_axes([0.9, 0.15, 0.01, 0.3])

        cb = plt.colorbar(C2,cax=cax2,
                          orientation='vertical',
                          extend="both")
        cb.set_label('LDR (dB)')
        
        axs[1,1].set_yticks([0,2000,4000,6000,8000,10000,12000])
        axs[1,1].set_ylim([0,12000])
        axs[1,1].set_yticklabels(["0","2","4","6","8","10","12"])
    
        for label in axs[1,0].get_xticklabels():
            label.set_rotation(30)
        if np.isnan(hourly):
            axs[1,1].xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0]))        
        else:
            axs[1,1].xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,15,30,45]))
        axs[1,1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))        
        axs[1,1].set_xlabel('Time (UTC)')
        sns.despine(offset=10)
        
        if not hasattr(self,"radar_fig_path"):
            self.specify_plot_path()
        if np.isnan(hourly):
            fig_name="radar_measurements_quicklook_"+\
                    str(self.cfg_dict["Flight_Dates_used"][0])+".png"
        else:
            fig_name="radar_measurements_quicklook_"+\
                    str(self.cfg_dict["Flight_Dates_used"][0])+"_"+\
                        str(hourly)+"00.png"
        fig.savefig(self.radar_fig_path+fig_name,bbox_inches="tight",dpi=300)
        print("Figure saved as:",self.radar_fig_path+fig_name)
    
    def plot_single_radar_cfad(self,radar_df,raw_measurements=True,
                               is_calibrated=False,skip_sea_surface=True):
        if skip_sea_surface:
            radar_df=radar_df.iloc[:,5:]
        radar_cfad_dict=calc_radar_cfad(radar_df)
        radar_cfad=radar_cfad_dict["relative"]
        total_sum_dbz_values=radar_cfad_dict["absolute"].sum().sum()
        radar_cfad=radar_cfad.replace(to_replace=0, 
                                          value=np.nan)
        
        print("Plot Radar CFADs")
        cfad_fig=plt.figure(figsize=(8,9))
        matplotlib.rcParams.update({"font.size":26})
        
        # Processed Radar CFAD
        ax1=cfad_fig.add_subplot(111)
        ax1.set_xlabel("Reflectivity (dBZ)")
        if raw_measurements:
            ax1.set_ylabel("Range (km)")
        else:
            ax1.set_ylabel("Height (km)")
        x_data=np.array(radar_cfad.columns)[5:]
        y_data=np.array(radar_cfad.index).astype("float")
        yy,xx=np.meshgrid(y_data,x_data)
        z=radar_cfad.iloc[:,5:]
        
        C1=ax1.pcolormesh(xx,yy,z.T,cmap=cmaeri.batlow)
        ax1.set_yticks(np.linspace(0,12000,7))
        ax1.set_yticklabels(np.linspace(0,12,7).astype(int).astype(str))
        fig_title="Raw Reflectivities"
        #if not raw_measurements:
        #    fig_title="Processed Reflectivities"
        #    if is_calibrated:
        #        fig_title="Calibrated "+fig_title
        #ax1.set_title(fig_title)
        
        ax1.set_xlim([-40,40])
        ax1.set_ylim([0,10000])
        
        if raw_measurements:
            ax1.text(10,1000,"Total value number: \n"+\
                 str(int(total_sum_dbz_values)),fontsize=20)
        else:
            ax1.text(10,9000,"Total value number: \n"+\
                 str(int(total_sum_dbz_values)),fontsize=20)
        
        cax = cfad_fig.add_axes([0.125, -0.05, 0.75, 0.03])

        cb = plt.colorbar(C1,cax=cax,orientation='horizontal',extend="max")
        cb.set_label("Relative Counts")
        for axis in ['top','bottom','left','right']:
            ax1.spines[axis].set_linewidth(2)
            ax1.tick_params(width=2)
        
        if raw_measurements:
            ax1.invert_yaxis()
        sns.despine(offset=10)
        fig_name="raw_radar_cfad_"+\
                    str([*self.cfg_dict["Flight_Dates_used"]][0])+".png"
        #cfad_fig.savefig(self.radar_fig_path+fig_name,dpi=300,
        #                 bbox_inches="tight")
        plot_path=os.getcwd()+"/"
        cfad_fig.savefig(plot_path+fig_name,dpi=300,
                         bbox_inches="tight")
        print("Figure saved as:", plot_path+fig_name)
        
    def radar_cfad_processing_comparison(self,roll_threshold=10):
        """
        

        Parameters
        ----------
        roll_threshold : TYPE, optional
            DESCRIPTION. The default is 10.

        Returns
        -------
        None.

        """
        
        if not hasattr(self,"raw_cfad_df"):
            raw_radar_df=self.raw_radar["dBZg"]
            self.raw_cfad=calc_radar_cfad(raw_radar_df)
        else:
            pass
        if not hasattr(self,"processed_cfad"):
           processed_radar_df=pd.DataFrame(
               data=np.array(self.processed_radar["dBZg"][:]),
               columns=np.array(self.processed_radar["height"][:]),
               index=pd.DatetimeIndex(np.array(self.processed_radar.time[:])))
           
           self.processed_cfad=calc_radar_cfad(processed_radar_df)
           
        cfad_df_raw=self.raw_cfad.replace(to_replace=0, 
                                          value=np.nan)
        cfad_df_processed=self.processed_cfad.replace(to_replace=0, 
                                                      value=np.nan)
        
        print("Plot Radar CFADs")
        cfad_fig=plt.figure(figsize=(16,12))
        
        # Processed Radar CFAD
        ax1=cfad_fig.add_subplot(121)
        ax1.set_xlabel("Reflectivity (dBZ)")
        ax1.set_ylabel("Height (km)")
        x_data=np.array(cfad_df_processed.columns)[5:]
        y_data=np.array(cfad_df_processed.index).astype("float")
        yy,xx=np.meshgrid(y_data,x_data)
        z=cfad_df_processed.iloc[:,5:]
        
        ax1.pcolormesh(xx,yy,z.T,cmap="cividis")
        ax1.set_yticks(np.linspace(0,13000,14))
        ax1.set_yticklabels(np.linspace(0,13,14).astype(int).astype(str))
        ax1.set_title("Processed Radar")
        ax1.set_xlim([-60,40])
        ax1.set_ylim([0,12000])
        
        # Raw Radar CFAD
        ax2=cfad_fig.add_subplot(122,sharey=ax1)
        ax2.set_xlabel("Reflectivity (dBZ)")
        ax2.set_ylabel("Height (km)")
        x_data=np.array(cfad_df_raw.columns)[5:]
        y_data=np.array(cfad_df_raw.index).astype("float")
        yy,xx=np.meshgrid(y_data,x_data)
        z_raw=cfad_df_raw.iloc[:,5:]
        C2=ax2.pcolormesh(xx,yy,z_raw.T,cmap="cividis")
        
        cax = cfad_fig.add_axes([0.15, -0.05, 0.7, 0.03])

        cb = plt.colorbar(C2,cax=cax,orientation='horizontal',extend="neither")
        cb.set_label('Relative Counts')
        ax2.set_title(" Raw Radar")
        ax2.set_xlim([-60,40])
        
        # Remove Spines of Plots
        sns.despine(offset=10)
        plt.suptitle(str(self.cfg_dict["Flight_Dates_used"].keys()[0])+":"+\
                     str(self.cfg_dict["Flight_Dates_used"].values[0])+\
                          ", roll angles $<$ "+str(roll_threshold)+"deg")
        
        fig_name=str(self.cfg_dict["Flight_Dates_used"].keys()[0])+\
                    "_Radar_CFAD_processed_raw_"+\
                        str(self.cfg_dict["Flight_Dates_used"].values[0])+".png"
        
        cfad_fig.savefig(self.radar_fig_path+fig_name,
                          dpi=300,bbox_inches="tight")
        print("Figure saved as: ",self.radar_fig_path+fig_name)
        