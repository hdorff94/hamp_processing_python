# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 08:46:26 2021

@author: u300737
"""

import sys
import os
import glob

import numpy as np
import numpy.matlib

import Performance
import pandas as pd
import scipy as sc
import scipy.interpolate as scint

import xarray as xr

import campaign_netcdf
from  campaign_time import sdn_2_dt

#%%
def radar_mask_table():
    #radar mask for questionable data

    # How to get the data:
    #  - plot radar data using lookForNoiseManually.m
    #  - use zoom tool to zoom in to the desired interval
    #  - set xlim
    #  - copy values into variable (see 'mask' below)

    # mask = {date, [sdn_start  sdn_end],   [height_start height_end]
    radar_mask_dict = {}
    
    #To be added in radar mask dict
         #--- NAWDEX ----
         # '20160917',    [736590.309164937          736590.311465207], 'noise';   % Noise
         # '20160917',    [736590.580453776          736590.580486586], 'noise';   % Noise
         # '20160921',    [736594.586277239          736594.588610378], 'noise';   % Noise
         # '20160921',    [736594.797549189          736594.797579726], 'noise';   % Noise
         # '20160923',    [736596.323084651           736596.32575763], 'noise';   % Noise
         # '20160926',    [736599.414619348          736599.414636999], 'noise';   % Noise
         # '20160926',    [736599.78057761           736599.780615758], 'noise';   % Noise
         # '20161001',    [736604.357215172          736604.359773586], 'noise';   % Noise 
         # '20161001',    [736604.44349709           736604.443531104], 'noise';   % Noise 
         # '20161006',    [736609.299127322            736609.3017089], 'noise';   % Noise 
         # '20161006',    [736609.665314155          736609.665350141], 'noise';   % Noise 
         # '20161009',    [736612.439801159          736612.443065847], 'noise';   % Noise 
         # '20161009',    [736612.786632154           736612.78667683], 'noise';   % Noise 
         # '20161010',    [736613.499059426          736613.508987323], 'noise';   % Noise 
         # '20161010',    [736613.805200446          736613.805290252], 'noise';   % Noise 
         # '20161013',    [736616.332106095          736616.332137241], 'noise';   % no continous measurement
         # '20161013',    [736616.58257123           736616.585582662], 'noise';   % something funny/turn?
         # '20161014',    [736617.357897676          736617.358584828], 'noise';   % Noise/Radar not operating
         # '20161014',    [736617.51891423           736617.527277008], 'noise';   % Noise/Radar not operating
         # '20161015',    [736618.389554936           736618.40149238], 'noise';   % Noise/Radar not operating
         # '20161015',    [736618.682270508          736618.682576946], 'noise';   % Noise/Radar not operating
         # '20161018',    [736621.454697596           736621.45479961], 'noise';   % Noise
         # '20161018',    [736621.386424758           736621.38923241], 'noise';   % Noise
         # '20161018',    [736621.595232179          736621.595491731], 'noise';   % Noise
         # ---- NARVAL-South ----
         # '20131210',    [735578.436482661           735578.44566576], 'noise';
         # '20131210',    [735578.853722838          735578.853890321], 'noise';
         # '20131211',    [735579.612637506          735579.623594667], 'noise';
         # '20131211',    [735579.890672201          735579.892608193], 'noise';  % high cloud looks funny
         # '20131211',    [735579.905338614          735579.905722114], 'noise';
         # '20131212',    [735580.583061802          735580.588218512], 'noise';
         # '20131212',    [735580.840207375          735580.840716856], 'noise';
         # '20131214',    [735582.839628856          735582.840419308], 'noise';
         # '20131215',    [735583.641230726          735583.645612998], 'noise';
         # '20131215',    [735583.899147737          735583.899453902], 'noise';
         # '20131216',    [735584.558088194          735584.562709897], 'noise';
         # '20131216',    [735584.945719085          735584.946770554], 'noise';
         # '20131219',    [735587.821024722          735587.821454733], 'noise';
         # '20131220',    [735589.083858778           735589.08390801], 'noise';
         # ---- NARVAL-North ----
         # '20140107',    [735606.509291557          735606.511137876], 'noise';
         # '20140107',    [735606.729681438          735606.729814865], 'noise';
         # '20140109',    [735608.481534017          735608.524246839], 'noise';
         # '20140109',    [735608.709953373           735608.71003703], 'noise';
         # '20140112',    [735611.35667459           735611.385315245], 'noise';
         # '20140112',    [735611.622818541          735611.623289132], 'noise';
         # '20140120',    [735619.771254704          735619.771327525], 'noise';
         # '20140121',    [735620.696101403          735620.696120677], 'noise';
         # '20140122',    [735621.417973978          735621.427157399], 'noise';
         # '20140122',    [735621.584894167          735621.584988785], 'noise';

    #NARVAL-II
    radar_mask_dict['20160810']=[736552.505398464,736552.506460273,"noise"]
    radar_mask_dict['20160812']=[[736554.527963097,736554.54358662, 'calibration'],
                                [736554.495527672,736554.49703469,"noise"]]
    radar_mask_dict['20160815']=[[736557.498984144,736557.528981682, 'noise'],      #  Noise/Radar not operating?
                                 [736557.535592669,736557.535671235, 'noise'],
                                 [736557.52903268,736557.534427799, 'calibration'], # Radar calibration
                                 [736557.806844602,736557.809051076, 'calibration']]# Radar calibration
    radar_mask_dict["20160817"]=[[736559.621772716,736559.623691942, 'noise']]      # Noise
    radar_mask_dict["20160819"]=[[736561.528088995,736561.530241181, 'noise'],      # Noise
                                 [736561.56674357,736561.571673848, 'calibration']] # Radar calibration
    radar_mask_dict["20160822"]=[[736564.559180551,736564.561112127, 'noise'],      # Noise
                                 [736564.82173855,736564.826712768, 'calibration']] # Radar calibration
    #EUREC4A
    radar_mask_dict['20200119']=[[737809.672516653,737809.675925044, 'calibration']]# calibration
    radar_mask_dict['20200126']=[[737816.715371586,737816.720483249, 'calibration']]# calibration
    radar_mask_dict['20200128']=[[737818.792001425,737818.810110551, 'calibration']]# calibration
    radar_mask_dict['20200131']=[[737821.795429663,737821.820376296, 'calibration']]# calibration
    radar_mask_dict['20200202']=[[737823.822031771,737823.827400287, 'calibration']]# calibration
    radar_mask_dict['20200207']=[[737828.676978126,737828.688448642, 'calibration'],# calibration
                                 [737828.708437327,737828.711482264, 'calibration']]# calibration
    radar_mask_dict['20200209']=[[737830.552105169,737830.567275506, 'calibration']]# calibration
    radar_mask_dict['20200211']=[[737832.822930835,737832.836675366, 'calibration']]# calibration
    radar_mask_dict['20200213']=[[737834.507867989,737834.510002615, 'calibration']]# calibration

    return radar_mask_dict     

#%%
def load_existing_mask(flight_date,cfg_dict,mask_type="land"):
    """
    

    Parameters
    ----------
    flight_date : str
        date of flight.
    cfg_dict : TYPE
        DESCRIPTION.
    mask_type : str, optional
        this is the mask type to land. The default is "land". 
        Other possibilities are noise, calibration, surface and sea_surface
        

    Returns
    -------
    radar_mask : pd.DataFrame or pd.Series
        the specific radar mask that was desired.

    """
    mask_path=cfg_dict["radar_mask_path"]
    if mask_type=="land":
        mask_name="Land_Mask_"
        if cfg_dict["campaign"]=="HALO_AC3":
            mask_name="Land_Sea_Ice_Mask_"
    elif mask_type=="noise":
        mask_name="Noise_Mask_"
    elif mask_type=="calibration":
        mask_name="Calibration_Mask_"
    elif mask_type=="surface":
        mask_name="Surface_Mask_"
    elif mask_type=="sea_surface":
        mask_name="Sea_Surface_Mask_"
    else:
        raise Exception("the given mask_type ", mask_type," is not defined!")
    try:
        # load landmask for given day
        radar_mask=pd.read_csv(mask_path+mask_name+str(flight_date)+".csv")
        radar_mask.index=pd.DatetimeIndex(radar_mask["Unnamed: 0"])
        del radar_mask["Unnamed: 0"]
    
    except:
        FileNotFoundError("The mask ",mask_type,
                          " is generally defined but not yet calculated")
        
    return radar_mask

def make_haloLandMask(flight_dates,outfile,cfg_dict,add_sea_ice_mask=False):
    """
    

    Parameters
    ----------
    flight_dates : TYPE
        DESCRIPTION.
    outfile : TYPE
        DESCRIPTION.
    cfg_dict : dict
        Dictionary containing the configuration arguments
    Returns
    -------
    None.

    """
    
    # Set file paths
    outpath = cfg_dict["radar_mask_path"]#cfg_dict["data_path"]+'Auxiliary/Masks/'
    ls_path = cfg_dict["data_path"]+'Auxiliary/lsmask-world8-var.dist5.5.nc'

    if not os.path.exists(ls_path):
        raise FileNotFoundError('Land mask file not found. ',
                    'Please download the file',
                    'lsmask-world8-var.dist5.5.nc',
                    'from https://www.ghrsst.org/ghrsst-data-services/tools/')
    else:
        pass

    if add_sea_ice_mask:
        #% Loop flights
        for flight in flight_dates:
            print("Land Sea ice mask for ",flight)
            
            #  Get file names
            f="*"+str(flight)+"*_*"+cfg_dict["version"]+"."+cfg_dict["subversion"]+"*.nc"
            radar_fpath = glob.glob(cfg_dict["device_data_path"]+"radar_mira/"+f)
            radar_ds = xr.open_dataset(radar_fpath[0])
            # Read position data and add it to a dataframe that includes the 
            # land mask to built
            #radar_pos= pd.DataFrame(data=np.nan,
            #            columns=["lat","lon","landmask"],
            #            index=pd.DatetimeIndex(np.array(radar_ds["time"])))
        
            #radar_pos["lat"]      = np.array(radar_ds["IRS_LAT"][:])
            #radar_pos["lon"]      = np.array(radar_ds["IRS_LON"][:])
            #radar_pos["landmask"] = np.nan
            
            import measurement_instruments_ql
            # this creates a new surface mask that includes the sea ice cover
            radar_pos=measurement_instruments_ql.BAHAMAS.add_surface_mask_to_data(
                radar_ds,cfg_dict,resolution="120s")
            
    else:    
        # Define dates to use
        # flightdates_mask = get_campaignDates('2016');
        performance=Performance.performance()
        # Read data
        print("Read landmask")
        land_mask_ds=xr.open_dataset(ls_path)
        
        land_mask_array=np.array(land_mask_ds['dst'][:])
        #  original: 0 indicates land; numbers >0 show distance from coast in
        #  pixels (??); fill values <0 indicate sea
        #  1 indicates land; set all sea values to 0
    
        land_mask_array[land_mask_array>1]  = 0
        land_mask_array[land_mask_array==0] = -1 ##-> added to -1 for sea ice mask
        land_mask_array[land_mask_array<0]  = 0
    
        lats_ls_mask= np.array(land_mask_ds["lat"][:])
        lons_ls_mask= np.array(land_mask_ds["lon"][:])                        
        dst = pd.DataFrame(data=land_mask_array,
                       index=lats_ls_mask,
                       columns=lons_ls_mask)
    
        del land_mask_array
        #% Preallocate
        radar_landMask = {}
    
        #% Loop flights
        for flight in flight_dates:
            print("Land mask for ",flight)
            
        #  Get file names
            f="*"+str(flight)+"*_*"+cfg_dict["version"]+"."+cfg_dict["subversion"]+"*.nc"
            radar_fpath = glob.glob(cfg_dict["device_data_path"]+"radar_mira/"+f)
            radar_ds = xr.open_dataset(radar_fpath[0])
            # Read position data and add it to a dataframe that includes the 
            # land mask to built
            radar_pos= pd.DataFrame(data=np.nan,
                                  columns=["lat","lon","landmask"],
                                  index=pd.DatetimeIndex(np.array(radar_ds["time"])))
        
            radar_pos["lat"]      = np.array(radar_ds["IRS_LAT"][:])
            radar_pos["lon"]      = np.array(radar_ds["IRS_LON"][:])
            radar_pos["landmask"] = np.nan
            
            # Loop time
            for t in range(radar_pos.shape[0]):
            
                if not (np.isnan(radar_pos["lat"].iloc[t])) and \
                    not (np.isnan(radar_pos["lon"].iloc[t])):
                        #  Calculate differences of aircraft position to land sea mask grid
                    lat_diff = np.array(abs(radar_pos["lat"].iloc[t]-lats_ls_mask))
                    lon_diff = np.array(abs(radar_pos["lon"].iloc[t]-lons_ls_mask))
                
                    #  Get indices of closest latitude/longitude grid
                    lon_ind = np.argmin(lon_diff)
                    lat_ind = np.argmin(lat_diff)
            
                    # Copy value into radar_pos dataframe
                    radar_pos["landmask"].iloc[t]=dst.iloc[lat_ind,lon_ind]
                else:
                    radar_pos["landmask"].iloc[t]=radar_pos["landmask"].iloc[t-1]
                performance.updt(radar_pos.shape[0],t)                                                             
    # Save land mask of flight as csv
    # If file already exists, overwrite it
    # THIS WAS in append mode BEFORE!
    radar_pos=radar_pos.resample("1s").mean()
    radar_pos=radar_pos.bfill()
    if not add_sea_ice_mask:
        landmask_file="Land_Mask_"+str(flight)+".csv"
    else:
        landmask_file="Land_Sea_Ice_Mask_"+str(flight)+".csv"
        
    radar_pos.to_csv(path_or_buf=outpath+landmask_file,index=True)
    #sys.exit()
    print("Land mask saved as:", outpath+landmask_file)    


#%%        
def make_radar_noise_calibration_mask(flightdates,outfile,
                                      mask_type,
                                      cfg_dict):
    
    # Input 'noise_or_calibration' can either be 'noise' or 'calibration'

    # Set file paths
    outpath = cfg_dict["radar_mask_path"]#cfg_dict["data_path"]+'Auxiliary/Masks/'
    

    # Load noise info data (generated by hand, using lookForRadarCalManually)
    mask = radar_mask_table();

    for f in flightdates.index:
        flight=flightdates.loc[f]
        
        # Find entries that match flight date and contain the chosen mask type
        try:
            ind_use = mask[str(flight)]
        except:
            ind_use=[]
        #  Get file names
        file="*"+str(flight)+"*_*"+cfg_dict["version"]+"."+cfg_dict["subversion"]+"*.nc"
        radar_fpath = glob.glob(cfg_dict["device_data_path"]+"radar_mira/"+file)
        radar_ds = xr.open_dataset(radar_fpath[0])
        
        # Read time and preallocate noise/calibration mask
        mask_series=pd.Series(data=np.zeros(radar_ds["time"].shape[0]),
                              index=pd.DatetimeIndex(np.array(radar_ds["time"])))
        
        # If an entry for this date exists
        if ind_use:        
            #% Loop all entries for this date
            for j in range(len(ind_use)):
                if ind_use[j][2]==mask_type:
                    dt_times=sdn_2_dt(ind_use[j][0:2])
                    mask_period_idx=pd.DatetimeIndex(dt_times)
                    # Set value in noise mask to true
                    mask_series.loc[mask_period_idx[0]:mask_period_idx[1]]=1
                else:
                    pass
        # Set output variable name depending if this is done for noise or
        # calibration
        if mask_type=="calibration":
            outvar = 'Calibration_Mask_'
        else:
            outvar = 'Noise_Mask_'

        mask_file=outvar+str(flight)+".csv"
                
        # Save mask series
        mask_series.to_csv(outpath+mask_file,
                           index=True)
        print(outvar,"saved as:", outpath+mask_file)    

#%%        
def make_radar_surface_mask(flightdates,outfile,cfg_dict,show_quicklook=False):
    """
    

    Parameters
    ----------
    flight_dates : TYPE
        DESCRIPTION.
    outfile : TYPE
        DESCRIPTION.
    cfg_dict : dict
        Dictionary containing the configuration arguments
    Returns
    -------
    None.

    """
    
    # Set file paths
    outpath = cfg_dict["radar_mask_path"]#cfg_dict["data_path"]+'Auxiliary/Masks/'    
    # Define dates to use
    performance=Performance.performance()
    # Read data
    
    # Preallocate
    radar_surfaceMask = {}
    
    
    # Loop all dates from file
    for f in flightdates.index:
        flight=flightdates.loc[f]
        # Find entries that match flight date and contain the chosen mask type
        
        # Find radar files from day. 
        # ! Obs: use version 2.3 for this analysis since side lobes during
        # turns have not been removed in this data set
        # if str2double(flightdates_mask)<20160000
        # radarfiles = listFiles([getPathPrefix 'NARVAL-I_campaignData/all_nc/*' flightdates_mask{f} 
        # '*v2.3*.nc'],'fullpath');
        # else
        
        file="*"+str(flight)+"*_*"+cfg_dict["version"]+"."+cfg_dict["subversion"]+"*.nc"
        radar_fpath = glob.glob(cfg_dict["device_data_path"]+"radar_mira/"+file)
        # Look for version numbers below v1.0 to ensure that side lobes haven't
        # been removed from the data yet
        versionNum = cfg_dict["version"]+"."+cfg_dict["subversion"]
        # Find index of current date in land mask
        #     ind = strcmp(flightdates_mask_input{i}, flightdates_mask);
        if not float(versionNum)<1.0:
            raise Exception("You assigned the wrong version number for the ",
                            "surface mask.")
        
        # load landmask for given day
        landmask_filename=outpath+"Land_Mask_"+str(flight)+".csv"
        if cfg_dict["campaign"]=="HALO_AC3":
            landmask_filename=outpath+"Land_Sea_Ice_Mask_"+str(flight)+".csv"
        landmask_df=pd.read_csv(landmask_filename)
        landmask_df.index=pd.DatetimeIndex(landmask_df["Unnamed: 0"])
        del landmask_df["Unnamed: 0"]
        
        # load noisemask for given day
        #noisemask_df=pd.read_csv(outpath+"Noise_Mask_"+str(flight)+".csv")
        # landmask_nan(landmask_nan==0) = nan;
        # landmask_flight = landMask{ind};
        
        # % Read radar data
        radar_ds = xr.open_dataset(radar_fpath[0])
         # Check if radar was working
        if not "dBZg" in radar_ds.keys():
            raise Exception("no dBZ is calculated ")
        try:
            z_df=pd.DataFrame(data=np.array(radar_ds["dBZg"][:]),
                          index=pd.DatetimeIndex(np.array(radar_ds.time[:])))
        except:
            z_df=pd.DataFrame(data=np.array(radar_ds["dBZg"][:].T),
                          index=pd.DatetimeIndex(np.array(radar_ds.time[:])))
        
        #         % Remove -inf values
        z_df[z_df==-np.inf] = np.nan
        
        # Omit the first and last two minutes of each flight in land mask, since
        # the radar has not been operating during these times...
        landmask_df.iloc[0:120]   = np.nan
        landmask_df.iloc[-120:-1] = np.nan
        landmask_df=landmask_df.reindex(z_df.index)
        # Remove noise and set to nan
        noise_file=outpath+"Noise_Mask_"+str(flight)+".csv"
        if os.path.exists(noise_file):
            noise_mask=pd.read_csv(noise_file)
            noise_mask.index=pd.DatetimeIndex(noise_mask["Unnamed: 0"])
            noise_mask=pd.Series(noise_mask.iloc[:,1])
        else:
            raise Exception("no dBZ is calculated")
        
        mask_arg="landmask"
        if cfg_dict["campaign"]=="HALO_AC3":
            mask_arg="sea_ice"
            mask_value=-0.1
        z_df.loc[noise_mask.loc[noise_mask==1].index]=np.nan        
        # Calculate maximum reflectivity for each profile
        zMax = z_df.max(axis=1)
        #zMax.loc[landmask_df[mask_arg].loc[landmask_df[mask_arg]==-0.1].index]\
        #        = np.nan ---> has this to be commented out?
        av_zMax = zMax.mean()
        std_zMax = zMax.std()
        # Preallocate
        indZMax = pd.Series(data=np.nan,index=landmask_df.index);
        hSurf = pd.Series(data=np.nan,index=z_df.index)     
        
        # Get indices of profiles without reflectivity measured
        height=np.array(radar_ds["height"])
        
        ind_no_dbz_profile = z_df.isnull().sum(axis=1)==len(height)
        
        # Loop time
        print("Check for surface values")
        for j in range(hSurf.shape[0]):
            # Check if landmask is -1 
            # and at least one measurement in radar profile
            # and profile's reflectivity maximum is larger 30 dBZ 
            # %%%%%than average zMaximum - 1 standard deviation
            # !change this value after radar data has been recalculated!
            if landmask_df[mask_arg].iloc[j]==mask_value and not \
                ind_no_dbz_profile.iloc[j] and \
                    zMax.iloc[j]>=30: #av_zMax-std_zMax
                indZMax.iloc[j] = z_df.iloc[j,:].idxmax()
                hSurf.iloc[j] = height[int(indZMax.iloc[j])]
            performance.updt(hSurf.shape[0],j)
        #  Fill gaps in surface height
            
        # Find first and last time step over land
        # (in doing this, gaps in the beginning and the end of the flight
        # without supporting surface height data are not filled)
        ind_first = hSurf[hSurf.isnull()].index[0]
        ind_last = hSurf[hSurf.isnull()].index[-1]
            
        # Preallocate
        hSurf_filled_nan = pd.Series(data=np.nan,index=z_df.index);
        # Fill gaps of surface height and write into time vector
        # accordingly
        hSurf_filled_nan.loc[ind_first:ind_last] = \
            hSurf.loc[ind_first:ind_last].interpolate(
                method="barycentric",limit=20)
    
        # Threshold of 30 dBZ worked for NAWDEX but not for NARVAL-II, to
        # be sure, just discard the lowest three to four range gates
        #z_low_level=z_df.iloc[1:4,:]
        
        
        # Remove time steps over ocean and with empty profiles
        hSurf_filled_nan.loc[landmask_df[mask_arg]==np.nan] = np.nan
        hSurf_filled_nan[ind_no_dbz_profile] = np.nan
            
        # Generate empty array for surface mask
        surface_mask = pd.DataFrame(data=np.nan,
                                   columns=radar_ds["height"],
                                   index=pd.DatetimeIndex(np.array(radar_ds["time"][:])))
            
        # Preallocate
        ind_hSurf = pd.Series(data=np.nan,
                              index=pd.DatetimeIndex(np.array(radar_ds["time"][:])))
            
        # Loop time
        print("Fill surface mask from zero to height of surface +2 extra levels")
        for j in range(ind_hSurf.shape[0]):
            # If time step is over land and surface height is not nan
            if (landmask_df[mask_arg].iloc[j]==mask_value) \
                and not (np.isnan(hSurf_filled_nan[j])):
                # Find range gate in which surface height falls in
                diff_hSurf = abs(hSurf_filled_nan[j]-np.array(radar_ds["height"]))
                ind_hSurf[j] = np.argmin(diff_hSurf)
                # Write one to surface mask from bottom to surface height
                # plus two range gates (just to be sure)
                surface_mask.iloc[j,0:int(ind_hSurf[j])+2] = mask_value
            performance.updt(ind_hSurf.shape[0],j)        
        
        if show_quicklook:
            # Plot resulting figure
            # fh = figure;
            # imagesc(t,h,z)
            # addWhiteToColormap
            # set(gca,'YDir','normal')
            # datetick('x','HH:MM')
            # title(flightdates_mask{i})
            # hold on
            # plot(t,landmask_nan-100,'rx')
            # ylim([-1000 15000])
            # figure(fh)
            # hold on
            # plot(t,hSurf,'xk')
            # plot(t,hSurf_filledNan,'co')
            # plot(t,ind_SeaSurf-50,'+g') 
            print("Surface Mask Quick Look not yet implemented")
            pass
        # Save surface mask dataframe
        mask_file="Surface_Mask_"+str(flight)+".csv"
        surface_mask.to_csv(outpath+mask_file,
                           index=True)
        print("Surface Mask saved as:", outpath+mask_file)    
    return None

#%%
def make_radar_sea_surface_mask(flightdates, outfile, cfg_dict):
    
    ####
    # Set file paths
    outpath = cfg_dict["radar_mask_path"]#+'Auxiliary/Masks/'    
    # Define dates to use
    performance=Performance.performance()
    
    # Preallocate
    radar_sea_surfaceMask = {}
    
    #ind_sea_surf = z_low_level[z_low_level>=30].sum(axis=1)
        #ind_sea_surf = ind_sea_surf.replace(to_replace=0,value=np.nan)
    
    # % Loop all dates from file
    for f in flightdates.index:
        flight=flightdates.loc[f]
        # Find entries that match flight date and contain the chosen mask type
        
        # Find radar files from day. 
        # ! Obs: use version 2.3 for this analysis since side lobes during
        # turns have not been removed in this data set
        
        file="*"+str(flight)+"*_*"+cfg_dict["version"]+"."+cfg_dict["subversion"]+"*.nc"
        radar_fpath = glob.glob(cfg_dict["device_data_path"]+"radar_mira/"+file)
        
        # Look for version numbers below v1.0 to ensure that side lobes haven't
        # been removed from the data yet
        versionNum = cfg_dict["version"]+"."+cfg_dict["subversion"]
        # Find index of current date in land mask
        if not float(versionNum)<1.0:
            raise Exception("You assigned the wrong version number for the ",
                            "surface mask.")
        
        # Read radar data
        radar_ds = xr.open_dataset(radar_fpath[0])
        
        # Check if radar was working
        if not "dBZg" in radar_ds.keys():
            raise Exception("no dBZ is calculated ")
        try:
            z_df=pd.DataFrame(data=np.array(radar_ds["dBZg"][:].T),
                          index=pd.DatetimeIndex(np.array(radar_ds.time[:])))
        except:
            z_df=pd.DataFrame(data=np.array(radar_ds["dBZg"][:]),
                          index=pd.DatetimeIndex(np.array(radar_ds.time[:])))
            
        # Remove -inf values
        z_df[z_df==-np.inf] = np.nan
        
        # Load mask data
        # load landmask for given day
        if not cfg_dict["campaign"]=="HALO_AC3":
            landmask_df=pd.read_csv(outpath+"Land_Mask_"+str(flight)+".csv")
            mask_arg="landmask"
            mask_value=1
        else:
            landmask_df=pd.read_csv(outpath+"Land_Sea_Ice_Mask_"+str(flight)+".csv")
            mask_arg="sea_ice"
            mask_value=-0.1
        landmask_df.index=pd.DatetimeIndex(landmask_df["Unnamed: 0"])
        del landmask_df["Unnamed: 0"]
        
        #load noise mask for given day
        #noise_file=outpath+"Noise_Mask_"+str(flight)+".csv"
        #if os.path.exists(noise_file):
        #    noise_mask=pd.read_csv(noise_file)
        #    noise_mask.index=pd.DatetimeIndex(noise_mask["Unnamed: 0"])
        #    noise_mask=pd.Series(noise_mask.iloc[:,1])
        #else:
        #    raise Exception("no dBZ is calculated")
            
        # Find instances where there is radar signal in any of the lowest
        # range gates defined
        range_gates=int(cfg_dict["num_rangegates_for_sfc"])
        ind_no_SeaSurf = z_df.iloc[:,0:range_gates].isnull().all(1)                    
        # Generate empty dataframe for surface mask
        sea_surface_mask = pd.DataFrame(data=np.nan,
                                columns=radar_ds["height"],
                                index=pd.DatetimeIndex(np.array(radar_ds["time"][:])))        
        # Set lowest four range gates to sea surface if there was a radar
        # signal in any of them and HALO was not over land
        sea_surf_index=ind_no_SeaSurf[ind_no_SeaSurf==False].index
        no_land_index=landmask_df[landmask_df[mask_arg]!=mask_value].index
        
        intersection_index=no_land_index.intersection(sea_surf_index)
        sea_surface_mask.loc[intersection_index,0:30*(range_gates-1)] = 1        
        # Save surface mask dataframe
        mask_file="Sea_Surface_Mask_"+str(flight)+".csv"
        sea_surface_mask.to_csv(outpath+mask_file,
                           index=True)
        print("Sea Surface Mask saved as:", outpath+mask_file)
    
    return None
    
#%%
def make_radar_info_mask(flightdates,outfile,cfg_dict):
    
    mask_path =  cfg_dict["radar_mask_path"]#+'Auxiliary/Masks/'    
    # Load data
    key = {'0':'good',
           '0-1':'sea_ice_cover',
           '.1':'surface',
           '3':'noise',
           '4':'radar calibration'}
    
    #radarInfoMask = 
    
    for f in flightdates.index:
        
        #Preallocate all masks
        noise_mask_df       = None
        calibration_mask_df = None
        surface_mask_df     = None
        sea_mask_df         = None
        flight=flightdates.loc[f]
        
        file="*"+str(flight)+"*_*"+cfg_dict["version"]+"."+cfg_dict["subversion"]+"*.nc"
        radar_fpath = glob.glob(cfg_dict["device_data_path"]+"radar_mira/"+file)
        
        # Read radar data
        radar_ds = xr.open_dataset(radar_fpath[0])
        
        mask_found=False
        try:
            noise_mask_df=load_existing_mask(flight,cfg_dict,mask_type="noise")
            radarInfoMask = np.zeros([noise_mask_df.shape[0],
                                      radar_ds["height"].shape[0]])
            noise_mask_df=pd.Series(noise_mask_df.iloc[:,0])
            mask_found=True
        except:
            pass    
        try:
            surface_mask_df=load_existing_mask(flight,cfg_dict,mask_type="surface")
            radarInfoMask = np.zeros([surface_mask_df.shape[0],
                                      radar_ds["height"].shape[0]])
            
            mask_found=True
        except:
            pass
        
        try:
            sea_mask_df=load_existing_mask(flight,cfg_dict,
                                           mask_type="sea_surface")
            # this mask is just binary representing sea surface in the lowest 
            # range gates defined by num_of_range_gates in cfg_dict
            
            if cfg_dict["campaign"]=="HALO_AC3":
                performance_cls=Performance.performance()
                sea_ice_mask=load_existing_mask(flight,cfg_dict,
                                                mask_type="land")
                sea_ice_mask=sea_ice_mask.reindex(sea_mask_df.index)    
                for t in range(sea_ice_mask.shape[0]):
                    sea_mask_df.iloc[t,:]=sea_mask_df.iloc[t,:]*\
                        sea_ice_mask["sea_ice"].iloc[t]
                    performance_cls.updt(sea_ice_mask.shape[0],t)
            radarInfoMask = np.zeros([sea_mask_df.shape[0],
                                      radar_ds["height"].shape[0]])
            
            mask_found=True
        
        except:
            pass
        
        try:
            calibration_mask_df=load_existing_mask(flight, cfg_dict,
                                                   mask_type="calibration")
            radarInfoMask = np.zeros([sea_mask_df.shape[0],
                                      radar_ds["height"].shape[0]])
            calibration_mask_df=pd.Series(calibration_mask_df.iloc[:,0])
            
            mask_found=True
        except:
            pass
        
        # Handle the maps
        if not mask_found:
            raise Exception("No mask file found")
        else:
            if surface_mask_df is not None:
                radarInfoMask[np.array(~surface_mask_df.isnull())] = -0.1
            if sea_mask_df is not None:
                radarInfoMask[:,0:int(cfg_dict["num_rangegates_for_sfc"])-1] =\
                    sea_mask_df.iloc[:,0:int(cfg_dict["num_rangegates_for_sfc"])-1]
            if calibration_mask_df is not None:
                radarInfoMask = pd.DataFrame(data=radarInfoMask,
                                             columns=radar_ds["height"],
                                             index=calibration_mask_df.index)
            
                radarInfoMask.loc[calibration_mask_df[calibration_mask_df==1].index] = 4
            
            # make noise at last to overwrite all other masks
            if noise_mask_df is not None:
                radarInfoMask.loc[noise_mask_df[noise_mask_df==1].index,:] = 3
        radarInfoMask["key"]=np.nan
        radarInfoMask["key"].iloc[0:5]=[*key.keys()]
        radarInfoMask["mask_value"]=np.nan
        radarInfoMask["mask_value"].iloc[0:5]=[*key.values()]
        # % Plot and save figure with radar mask if specified in varargin
        # if nargin>2 && strcmp(varargin{1},'figures')
            
        #     % Load Bahamas data
        #     bahamasfile = listFiles([getPathPrefix getCampaignFolder(flightdates_mask{i})  'all_nc/*bahamas*' flightdates_mask{i} '*.nc'],'fullpath');
        #     t = ncread(bahamasfile{end},'time');
        #     h = ncread(bahamasfile{end},'height');
        #     if ~issdn(t(1))
        #         t = unixtime2sdn(t);
        #     end
            
        #     % Plot
        #     fh = figure;
        #     cm = brewermap(5,'Set1');
        #     cm(1,:) = [];
        #     set(gcf, 'color','white');
        #     imagesc(t,h,radarInfoMask{i})
        #     colormap(cm)
        #     addWhiteToColormap
        #     set(gca,'CLim',[0 max(cell2mat(key(:,1)))])
        #     ch = colorbar;
        #     ch.Ticks = ch.Limits(1)+ch.Limits(2)/size(colormap,1)/2 : ch.Limits(2)/size(colormap,1) : ch.Limits(2);
        #     ch.TickLabels = key(:,2);
        #     set(gca,'YDir','normal')
        #     datetick('x','HH:MM','Keeplimits')
        #     title(flightdates_mask{i})
        #     xlabel('Time (UTC)')
        #     ylabel('Height (m)')
        #     setFontSize(gca,12)
        
        # Saving data
        fname="Radar_Info_Mask_"+str(flight)+".csv"
        radarInfoMask.to_csv(mask_path+fname,
                           index=True)
        print("Radar Info Mask saved as: ",mask_path+fname)
        
#%%
def run_make_masks(flightdates, cfg_dict):
    """
    Run script to call functions for radar data masks


    Parameters
    ----------
    flightdates : pd.Series
        Series containing the flights to mask the radar data from.
    cfg_dict : dict
        Dictionary including the configurations for the overarching script.

    Returns
    -------
    None.

    """
    


    #campaign = getCampaignName(flightdates(1));

    # Define output file
    outfile_path=cfg_dict["device_data_path"]+"auxiliary/"
    if not os.path.exists(outfile_path):
        os.mkdir(outfile_path)
    cfg_dict["radar_mask_path"]=outfile_path
    outfile = outfile_path+"radar_mask_"+cfg_dict["campaign"]+".npy"
    
    
    # Define dates to use
    # flightdates = get_campaignDates(campaign);
    
    # Land Mask
    add_sea_ice_mask=False
    if cfg_dict["campaign"]=="HALO_AC3":
        add_sea_ice_mask=True
    if Performance.str2bool(cfg_dict["land_mask"]):
        print('Generating land sea mask for the given flights:',flightdates)
        make_haloLandMask(flightdates,outfile,cfg_dict,
                          add_sea_ice_mask=add_sea_ice_mask)
    else:
        print('Skipping land sea mask...')
    
    # Noise Mask
    if Performance.str2bool(cfg_dict["noise_mask"]):
        print('Generating radar noise mask for the given flights:',flightdates)
        make_radar_noise_calibration_mask(flightdates,outfile,
                                          'noise',cfg_dict)
    else:
        print('Skipping radar noise mask...')
    
    
    # Calibration Mask
    if Performance.str2bool(cfg_dict["calibration_mask"]):
        print('Generating calibration mask for the given flights:',flightdates)
        make_radar_noise_calibration_mask(flightdates,outfile,
                                          'calibration',cfg_dict)
    else:
        print('Skipping radar calibration mask...')
    
    # Surface Mask    
    if Performance.str2bool(cfg_dict["surface_mask"]):
        print('Generating surface mask for the given flights: ',flightdates)
        make_radar_surface_mask(flightdates,outfile,cfg_dict)
    else:
        print('Skipping surface mask...')

    
    # Sea Surface Mask
    if Performance.str2bool(cfg_dict["seasurface_mask"]):
        print('Generating sea surface mask for all flights: ',flightdates)
        make_radar_sea_surface_mask(flightdates, outfile, cfg_dict)
    else:
        print('Skipping sea surface mask...')
    
    
    print('Combining all masks into one')
    make_radar_info_mask(flightdates,outfile,cfg_dict)