# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 20:18:23 2021

@author: u300737
"""
import sys
import os
import glob

import numpy as np
import numpy.matlib

import Performance
import pandas as pd
import scipy.interpolate as scint

import xarray as xr

import campaign_netcdf
#from Campaign_netCDF import CPGN_netCDF
from measurement_instruments_ql import HALO_Devices, BAHAMAS, SMART

def regrid_flight_angles(radar_range,rll_angle,ptch_angle,hGPS,Z_grid,
                         var_df,time_max):
    """
    This function regrid the radar ranges to the unified 30-m vertical axis of 
    the unified grid. This requires the flight attitude data, i.e. roll and pitch

    Parameters
    ----------
    radar_range : xr.DataArray
        values of ranges given in meters. This is around 30 m.
    rll_angle : pd.Series
        Series of the roll angles given in degrees.
    ptch_angle : pd.Series
        Series of the pitch angles given in degrees.
    hGPS : Series
        Series of the Aircraft Height given in meters.
    Z_grid : np.array
        30-m res vertical axis of the unified grid until maximum cruising level.
    var_df : pd.DataFrame
        radar variable to be regridded in this function.

    Returns
    -------
    var_intp : pd.DataFrame
        radar variable regridded onto the unified grid by nearest-neighbour method.

    """
    
    perform=Performance.performance()
    var_df=pd.DataFrame(data=np.array(var_df[:]),index=hGPS.index)
    # Calculate height on axis perpendicular to earth
    # h = range * cos(rollAngle) * cos (pitchAngle)
    rng             = np.matlib.repmat(radar_range[:],rll_angle.shape[0],1)
    cos_rllangl     = np.matlib.repmat(np.cos(np.deg2rad(rll_angle)),
                                       radar_range.shape[0],1).T
    cos_ptchangl    = np.matlib.repmat(np.cos(np.deg2rad(ptch_angle)),
                                       radar_range.shape[0],1).T
    h_array         = rng*cos_rllangl*cos_ptchangl
    h=pd.DataFrame(data=h_array,columns=np.arange(0,radar_range.shape[0]),
                   index=rll_angle.index)
    
    # Convert distance from aircraft to height by substracting it from flight
    # altitude
    h2=np.matlib.repmat(hGPS,radar_range.shape[0],1).T-h
    
    # loop over all timesteps
    considered_time=time_max
    #print("Regrid radar variable:",var_df)
    interp_values=np.empty((considered_time,len(Z_grid)))
    for i in range(considered_time):#range(var_df.shape[0]):
        # loop all columns
        interp_func=scint.interp1d(h2.iloc[i,:],var_df.iloc[i,:],
                                   kind="nearest",bounds_error=False,
                                   fill_value=np.nan)
        interp_values[i,:]=interp_func(Z_grid)
        #if i==0:
        #interp_values=np.expand_dims(interp_func(Z_grid),axis=1).T
        #else:
        #    new_array=np.expand_dims(interp_func(Z_grid),axis=1).T
        #    interp_values=np.vstack((interp_values,new_array))
        perform.updt(considered_time,i)
    var_intp=pd.DataFrame(data=interp_values,
                          columns=Z_grid,
                          index=var_df.index[0:considered_time])
    return var_intp

def correct_att_smart(radar_fname,version_number,radarOut_dir,
                      cfg_dict,remove_side_lobes=False,used_device="SMART"):
    """
    Comments to be filled in

    Parameters
    ----------
    radar_fname : TYPE
        DESCRIPTION.
    version_number : TYPE
        DESCRIPTION.
    radarOut_dir : TYPE
        DESCRIPTION.
    cfg_dict : TYPE
        DESCRIPTION.
    remove_side_lobes : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    # Load device classes
    # Bahamas class
    HALO_Devices_cls=HALO_Devices(cfg_dict)
    SMART_cls=SMART(HALO_Devices_cls)
    
    
    campaign_path   = cfg_dict["device_data_path"]
    SmartPath       = SMART_cls.device_data_path
    
    #   read radar file and 
    #   extract time information
    #   as ds["time"]['units'] is not correctly defined in the attributes 
    #   a local copy is necessary (temporary_ds)
    radar_ds = xr.open_dataset(radar_fname)
    attrs= {'units': str(radar_ds.time.long_name)}
    temporary_ds=xr.Dataset({'time':('time',np.array(radar_ds.time[:]),attrs)})
    temporary_ds=xr.decode_cf(temporary_ds)
    radar_ds["time"]=temporary_ds["time"]
    del temporary_ds
    
    radar_time = pd.DatetimeIndex(np.array(radar_ds["time"][:]))
    radar_date = radar_time.strftime("%Y%m%d")
    #check if more than one date is read, if so stop executing
    if len(radar_date.unique())>1:
        print("Under the given arguments more than one day is included.",
              "This causes errors.")
        raise(Exception("Script is not yet capable to read more than",
                        " one day at the same time."))
        sys.exit()
    else:
        #% Date as string
        date=radar_date.unique()[0]    
    
    #   Outfile Definition
    outfile_path            = cfg_dict["device_data_path"]+'radar_mira/' 
    
    #   Adjust radar time to file name
    attcorr_radar_fname     = "mira_"+str(radar_date[0])+'_'+\
                                radar_time.time[0].strftime('%H%M')    
    #   Adjust version specifications to file name
    attcorr_radar_fname     = attcorr_radar_fname+'_g40_v'+\
                                str(cfg_dict["version"])+'.'+\
                                    str(cfg_dict["subversion"])+".nc"
    
    outfile                 = outfile_path+attcorr_radar_fname
    cfg_dict["radar_outf"]  = outfile
    #delete file if outfile already exists
    #if os.path.isfile(outfile):
    #   del outfile

    #%% SMART Variable Selection
    # one dimensional data to copy
    
    
    ###### ----> has to be changed from radar to smart
    """
    
    # OLD NAMES---> To be ignored 
    # % varBahamas = {'P','RH','abshum','mixratio','speed_air','T','Td','theta',...
    # %               'theta_v','Tv','U','V','W','lat','lon','pitch','heading',...
    # %               'roll','Ts','alpha','beta','h','palt','mc','qc','wa','ws',...
    # %               't_sys','galt','nsv','ewv','vv','p','q',...
    # %               'r','axb','ayb','azb','azg','ata','speed_gnd'};
    var_bahamas =  ['PS','RELHUM','ABSHUM','MIXRATIO','TAS','TAT','TD','THETA',
                   'THETA_V','TV','U','V','W','IRS_LAT','IRS_LON','IRS_THE',
                   'IRS_HDG','IRS_PHI','TS','ALPHA','BETA','H','HP','MC','QC',
                   'WA','WS','IRS_ALT']
       
    """
    #%% SMART essentials 
    #smart dataset
    SMART_cls.open_data()
    smart_ds = SMART_cls.ds

    var_copy = ['nfft','prf','NyquistVelocity','nave','zrg','rg0','drg','lambda',
                'tpow','npw1','npw2','cpw1','cpw2','grst']

    # Check Konow var copy list
    var_edit = ['SNRg','VELg','RMSg','LDRg','HSDco','HSDcx','Zg']

    #   List Bahamas variables in file
    var_smart =  ['alt','roll','pitch','lat','lon','yaw']
    
    #   time
    #timename_use = Bahamas_cls.replace_bahamas_varname('TIME',vars_in_bahamas)
    smart_time = pd.DatetimeIndex(np.array(smart_ds["time"]))
    
    #   location
    h_GPS = pd.Series(smart_ds["alt"],
                      index=smart_time[:])
    
    #   flight data
    roll_angle   = pd.Series(smart_ds["roll"],
                             index=smart_time[:])
    #varname_use = SMART_cls.replace_bahamas_varname('IRS_THE',vars_in_bahamas)
    pitch_angle = pd.Series(smart_ds["pitch"],
                            index=smart_time[:])

    #   Replace variable names
    #for var in var_bahamas:
    #    varname_use_dict[var] = Bahamas_cls.replace_bahamas_varname(
    #        var,vars_in_bahamas)
    # #    Get maximum altitude 
    alt_max = h_GPS.max()
    # %% Adjust time series
    print('Adjust time series')
    # round times to avoid numerical deviations
    t_both              = pd.DatetimeIndex.intersection(smart_time,
                                                        radar_time)
    smart_time_series = pd.Series(data=range(len(smart_time)),
                                    index=smart_time)
    radar_time_series   = pd.Series(data=range(len(radar_time)),
                                    index=radar_time)
    smart_int_idx     = smart_time_series.loc[\
                                smart_time_series.index].values
    if t_both.shape[0]==0:
        raise AssertionError('SMART and mira do not match.',
                             ' Did you select the correct files?')
    time_max=t_both.shape[0] #10000# for faster tests
    
    #%% Edit Data
    print('Edit Data')
    # Read essentials
    radar_range = radar_ds['range']
    ####
    ##### -----> here def create_radar_att_corr_could start
    
    # Read data of variables to be edited
    radar_data = {}
    for var in var_edit:
        #print("Include Radar ",var)
        radar_data[var] = radar_ds[var]
        radar_data[var] = radar_data[var].fillna(float("-Inf"))
        radar_data[var]["time"]=radar_time
    
    
    #% Adjust to common time steps
    h_GPS        = h_GPS.loc[t_both]
    roll_angle   = roll_angle.loc[t_both]
    pitch_angle  = pitch_angle[t_both]

    #%% Modify Bahamas Data
    #% Read data from NetCDF
    attitude_data = {}
    for var in var_smart:
        attitude_data[var] = smart_ds[var][smart_int_idx]
    print("smart data allocated and cutted")
    
    #%% Correct Flight Attitude
    print('Correct Attitude')
    # Look for roll angles larger than specified deg, i.e. turning
    # Create flag value for turning
    print("Roll threshold: ",cfg_dict["roll_threshold"])
    turning_tracks  = roll_angle.loc[abs(roll_angle)>\
                                     float(cfg_dict["roll_threshold"])]
    turning_int_idx = smart_time_series.loc[turning_tracks.index].values
    turning_flag    = (abs(roll_angle)>\
                       float(cfg_dict["roll_threshold"])).astype(int)
    # % Round up to next 100 m
    alt_max = int(alt_max/100)*100+100
    
    # define vertical grid for variables
    if alt_max<=14000:  # keep this for compatibility between flights
        z_Grid = np.arange(0,14000,30)
    else:               # if aircraft ceiling was higher, extend vertical coord.
        z_Grid = np.arange(0,alt_max+30,30)
    
    radar_data_corr={}
    
    #%%
    ## Perform actual attitude correction
    if not remove_side_lobes:
        # Remove data from side lobes during turns
        for var in var_edit:
            #radar_no_side_lobes = radar_data[var].loc[turning_int_idx]
            #radar_no_side_lobes = data_radar_no_side_lobes[var].iloc[:,0:len(Z_grid)]
            
            # % Regrid radar data to acount for flight attitudes
            #print("Regrid radar variable:",var)
            radar_data_corr[var]=regrid_flight_angles(radar_range,
                                                 roll_angle,
                                                 pitch_angle,
                                                 h_GPS,z_Grid,
                                                 radar_data[var],
                                                 time_max)
    else:   
            ## !!!!!! This will follow laterly
            def correct_side_lobes():
                pass
            #correct_side_lobes()
    
    #%% Initialize netCDF file
    import netCDF4 as nc4
    nc_attributes=nc4.Dataset(SMART_cls.file).__dict__
    #Set format to bahamas_file format, but "NETCDF4_CLASSIC" would be better
    nc_attributes["Format"]=nc4.Dataset(SMART_cls.file).data_model
    #
    outfile_ds=xr.Dataset()
    outfile_time=xr.DataArray(t_both[0:time_max])#.astype(np.int64)
    
    vertical_coordinate= z_Grid
    outfile_height = xr.DataArray(vertical_coordinate)
    # Adjust name
    outfile_ds=outfile_ds.expand_dims({"time":outfile_time,"height":outfile_height})
    outfile_ds.time.attrs["long_name"]     = "Seconds since 01.01.1970 00:00 UTC"
    #outfile_ds.time.attrs["units"]         = "Seconds"
    outfile_ds.time.attrs["axis"]          = "T"
    outfile_ds.height.attrs["long_name"]   = "Height above MSL"
    outfile_ds.height.attrs["name"]        = "height"
    #outfile_ds.height.attrs["units"]       = "m"
    outfile_ds.height.attrs["axis"]        = "Y" 
    
    #print("Check reference dataset. To be deleted")
    #reference_ds=xr.open_dataset(cfg_dict["radarOut_dir"]+"mira_20200131_1515_g40_v0.2.nc")
    #produced_ds=xr.open_dataset(cfg_dict["radar_outf"])
    
    #%%Copy Unmodified Variables
    print('Copy Unmodified Variables')
    # Copy one-dimensional radar variables
    
    for var in var_copy:
        if var in radar_ds.keys():
            #print(var, "is in radar_ds")
            # keep in mind different types of variables (single ints or time_series)
            if radar_ds[var].size==1: #single int
                outfile_ds[var]=radar_ds[var]
            elif radar_ds[var].coords.dims[0]=="time":# only time dimension
                new_data_array=radar_ds[var][0:time_max]
                new_data_array=new_data_array.assign_coords({
                                            "time":outfile_ds.coords["time"]})
                outfile_ds[var]=new_data_array  
                outfile_ds[var].attrs=radar_ds[var].attrs               
            else:
                raise AssertionError('Unexpected dimensions.',
                                     ' Please consider listing the variabel ',
                                     var,' under ''var_edit''.')
                
    # Copy and slightly modify global attributes from radar standard file 
    CPGN_netCDF=Campaign_netCDF.CPGN_netCDF()
    outfile_ds = CPGN_netCDF.copy_modify_netCDF_global_att(radar_fname,
                                                           outfile_ds,
                                                           cfg_dict)
    
    
    
    #%% Write Bahamas data to NetCDF
    print('Write SMART data to NetCDF')
    #nc_attributes=nc4.Dataset(bahamas_file).__dict__  
    smart_copy_ds=smart_ds.copy()
    # Rename time dimension
    if "tid" in [*smart_copy_ds.dims.keys()]:
        smart_copy_ds=smart_copy_ds.rename_dims({"tid":"time"})
    elif "utc_time" in [*smart_copy_ds.dims.keys()]:
        smart_copy_ds=smart_copy_ds.rename_dims({"tid":"time"})
    elif "time" in [*smart_copy_ds.dims.keys()]:
        print("Time variable name already as desired")
    else:
        raise ValueError('No time dimension named tid or',
                         'utc_time is in the dataset',
                         "Recheck the dimensions",smart_copy_ds.dims)
    for var in var_smart:
        #print("SMART var:",var)
        # Remove global fill value
        time_adj_var_ds=smart_copy_ds[var][smart_int_idx][0:time_max]
        time_adj_var_ds.encoding['_FillValue']=None
        # Add Atribute 'yrange'
        time_adj_var_ds.attrs['yrange']=[np.min(np.array(time_adj_var_ds)),
                                np.max(np.array(time_adj_var_ds))]
        
        #Set format to SMART_file format, but "NETCDF4_CLASSIC" would be better
        time_adj_var_ds=time_adj_var_ds.assign_coords({"time":outfile_ds.time})
        #% Write variable to new file
        outfile_ds[var]=time_adj_var_ds
        
    
    #%% Write Radar data to NetCDF
    print('Write Radar data to NetCDF')
    radar_corr_copy_ds=radar_data_corr.copy()
    for var in var_edit:
        #print("Radar var:",var)
        radar_corr_array=np.array(radar_corr_copy_ds[var])
        radar_corr_array=np.where(radar_corr_array==-np.inf,
                             cfg_dict["missing_value"],
                             radar_corr_array)
        temporary_dataarray=xr.DataArray(radar_corr_array.T,
                                         coords=outfile_ds.coords,
                                         dims=outfile_ds.dims,
                                         attrs=radar_ds[var].attrs).T
        #% Add fill value information, need to be transposed
        temporary_dataarray.attrs["fill_value"]="NaN"
        
        outfile_ds[var]=temporary_dataarray
    
    #%% Calculate dBZ
    print('Calculate dBZ')
    if 'Zg' in var_edit:
        # Zg(Zg==missingvalue) = -Inf;
        # Calculate dBZ
        # using the emath extension to get complexe values and -inf instead of nan 
        dBZg = np.array(10*np.emath.log10(radar_data_corr['Zg'][:]))
        dBZg = dBZg.astype(np.float32)
        #% Only keep real part of array (imaginary numbers were created when
        #% taking log10 of -Inf: log10(-Inf) = Inf +      1.36437635384184i)
        dBZg = np.real(dBZg)
        #% And convert positive infinity back to negative infinity
        dBZg = np.where(dBZg==np.inf,-np.inf,dBZg)
        
    else:
        print('No Reflectivity Z found. Skipping dBZ calculation...')

    #%% Write dBZ to new radar file
    #% Change name
    dBZg_ds=xr.DataArray(dBZg.astype(np.float32).T,
                         coords=outfile_ds.coords,
                         dims=outfile_ds.dims).T
    dBZg_ds.attrs={"long_name"  :'Reflectivity dBzg',
                   "units"      :' ',
                   "yrange"     : [np.nanmin(np.nanmin(dBZg)),
                                   np.nanmax(np.nanmax(dBZg))]}
    
    #% Apply missing value to dBZ variable
    #dBZg(isinf(dBZg)) = missingvalue;
        
    #% Write data to outfile
    outfile_ds["dBZg"]= dBZg_ds
    var_edit.append("dBZg")
    
    #%% Flags
    print('Create flags')
    
    turning_flag_da=xr.DataArray(turning_flag[0:time_max],
                                 dims={"time":outfile_ds.dims["time"]})
    turning_flag_da.attrs={'units':' ',
        'long_name':'flag for roll angle > {threshold} deg, regarded as turn'\
            .format(threshold=cfg_dict["roll_threshold"]),
                'yrange':[0,1]}
    turning_flag_da=turning_flag_da.assign_coords({"time":outfile_ds.time})
        
    outfile_ds['curveFlag']=turning_flag_da
    #Add to global attributes
    if not remove_side_lobes:
        version_explanation=": data flipped, time offset corrected,"+\
                            " flight attitude corrected ("+used_device+\
                                "), sidelobes not removed"
    else:
        version_explanation=": data flipped, time offset corrected,"+\
                            " flight attitude corrected ("+used_device+")"
   
    outfile_ds.attrs["version_history"]='v'+str(cfg_dict["version"])+'.'+\
                                                str(cfg_dict["subversion"])+\
                                                    version_explanation
    convert_att="converted by Henning Dorff on "
    actual_time_att=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    f_used       =", using: '"+__name__+".py"
    outfile_ds.attrs["conversion_information"]=convert_att+actual_time_att+\
                                                    f_used
    
    #%% Save the file
    print("Save file, this takes a while")
    
    if os.path.exists(cfg_dict["radar_outf"]):
        os.remove(cfg_dict["radar_outf"])
    
    # set compression of radar variables
    nc_compression=dict(zlib=True,complevel=1,dtype=np.float64)
    nc_encoding= {radar_var:nc_compression for radar_var in var_edit}
    outfile_ds["time"]=outfile_ds["time"].astype(np.int64)
    outfile_ds.to_netcdf(path=cfg_dict["radar_outf"],mode="w",format="NETCDF4",
                         engine="netcdf4",
                         encoding=nc_encoding)
    
    outfile_ds.close()
    return None
    
    
def correct_att_bahamas(radar_fname,version_number,radarOut_dir, 
                        cfg_dict,remove_side_lobes=False,
                        used_device="BAHAMAS"):
    """
     Combines data from Mira radar and Bahamas into one file and 
     corrects for time offset of radar and flight attitude:
          - Loads Bahamas and Mira data files
          - Loads time offset data from lookup table and corrects for
           those
          - Variables that should converted are defined in the Variable
            Selection Section below
          - Common time steps from both systems are selected and only
            data at those times is considered further
          - Data measured during turns of the aircraft (roll angle > 3
            deg) is flagged
          - Radar data that is in a range bin farther from the aircraft
            than the ground is omitted
          - Radar data matrix is flipped to reflect the downward-looking
            of the instrument
          - Radar data is regridded and corrected for flight attitude
          - All data is written to NetCDF
          - dBZ are calculated

     Syntax:  HaloRadarBahamasCombCorrectTime(RadarFile,versionNumber,netCDFPath)
      
  
     Example: 
          see script: run_att_correction
          See also: perform_att_comb,
             HaloRadarBahamasCombCorrectTime, runRadarBahamasComb

     20160615: added additionaly input argument: varargin, set to be 'nolobes'
                   to prevent removal to side lobes (use this to see if
                   derivation of surface mask is easier now)
     20210221: rewritten to python by Henning Dorff

    Author: Dr. Heike Konow
    Meteorological Institute, Hamburg University
    email address: heike.konow@uni-hamburg.de
    Website: http://www.mi.uni-hamburg.de/
    MatLab-Version: January 2014; Last revision: June 2015
    
    Modificator: Henning Dorff
    Meteorological Institute, Hamburg University
    email address: henning.dorff@uni-hamburg.de
    Website: http://www.mi.uni-hamburg.de
    February 2021: last revision: February 2021
    ---------------------------------------------------------------------------
    Parameters
    ---------------------------------------------------------------------------
    radar_fname : TYPE
        DESCRIPTION.
    version_number : TYPE
        DESCRIPTION.
    radarOut_dir : TYPE
        DESCRIPTION.
    cfg_dict : TYPE
        DESCRIPTION.
    remove_side_loobs : bool
        boolean if side lopes should be removed/corrected or not (just flagged)

    Returns
    -------
    None.

    """
    
    #%%------------- BEGIN CODE --------------

    
    #ind_folder_string = regexp(RadarFile,'/radar/');

    # Version Number is assigned during function call, subversion number is
    # defined as '2' for this type of operation 
    # (exception: if side lobes should not be removed, subversion is 3)
    #-------------------------------------------------------------------------#
    # Old stuff will be added in another frame
    #if remove_side_lobes:
    #    subversionNumber = '2';
    #    print("Radar side lobes will be ignored from dataset. No correction")
    #else:
    #    subversionNumber = '3'
    #    print("Radar side lobes correctly. This is new and will be")
    #-------------------- current status, below is not yet rewritten
    # Check if radar data file is assigned during function call, otherwise 
    # predefined files
    #if ~exist('RadarFile','var')
    #RadarFile = ...
    #'/data/share/u231/u231107/HAMP/NARVAL-North/flight09_20140107_EDMO-BIKF/mira36/20140107_all_g40.mmclx';
    #netCDFPath = '/data/share/u231/u231107/HAMP/NARVAL-North/mira-allNetCDF/';
    #end

    #   read radar file and 
    #   extract time information
    #   as ds["time"]['units'] is not correctly defined in the attributes 
    #   a local copy is necessary (temporary_ds)
    radar_ds = xr.open_dataset(radar_fname)
    attrs= {'units': str(radar_ds.time.long_name)}
    if not np.issubdtype(radar_ds.time.values.dtype,np.datetime64):
        temporary_ds=xr.Dataset({'time':('time',np.array(radar_ds.time[:]),attrs)})
        temporary_ds=xr.decode_cf(temporary_ds)
        radar_ds["time"]=temporary_ds["time"]
        del temporary_ds
    
    radar_time = pd.DatetimeIndex(np.array(radar_ds["time"][:]))
    radar_date = radar_time.strftime("%Y%m%d")
    #check if more than one date is read, if so stop executing
    if len(radar_date.unique())>1:
        print("Under the given arguments more than one day is included.",
              "This causes errors.")
        raise(Exception("Script is not yet capable to read more than",
                        " one day at the same time."))
        sys.exit()
    else:
        #% Date as string
        date=radar_date.unique()[0]
    
    #   Search for corresponding bahamas data file
    #   Look for files from flight
    
    fname      = "*"+str(date)+"*.nc"
    bahamas_file=glob.glob(cfg_dict["bahamas_dir"]+fname,recursive=True)[0]
    
    #   Between the bahamas data and observations time offsets can frequently appear
    #   throughout the campaigns.
    #   The resulting time offsets are listed in a look up table.
    
    #   Correct time offset and get offset (in seconds) for specific day 
    import campaign_time as Campaign_Time
    Time_cpgn=Campaign_Time.Campaign_Time("EUREC4A",date)
    time_offset = Time_cpgn.look_up_bahamas_time_offs()

    #   Adjust time by adding offset
    radar_time = radar_time + pd.Timedelta(value=time_offset,unit="seconds")
    
    #   Outfile Definition
    outfile_path            = cfg_dict["device_data_path"]+'radar_mira/' 
    
    #   Adjust radar time to file name
    attcorr_radar_fname     = "mira_"+radar_date[0]+'_'+\
                                radar_time.time[0].strftime('%H%M')
    
    #   Adjust version specifications to file name
    attcorr_radar_fname     = attcorr_radar_fname+'_g40_v'+\
                                str(cfg_dict["version"])+'.'+\
                                    str(cfg_dict["subversion"])+".nc"
    
    outfile                 = outfile_path+attcorr_radar_fname
    cfg_dict["radar_outf"]  = outfile
    #delete file if outfile already exists
    #if os.path.isfile(outfile):
    #   del outfile

    #%% BAHAMAS Variable Selection
    # one dimensional data to copy
    var_copy = ['nfft','prf','NyquistVelocity','nave','zrg','rg0','drg','lambda',
                'tpow','npw1','npw2','cpw1','cpw2','grst']

    var_edit = ['SNRg','VELg','RMSg','LDRg','HSDco','HSDcx','Zg','Ze']

    # OLD NAMES---> To be ignored 
    # % varBahamas = {'P','RH','abshum','mixratio','speed_air','T','Td','theta',...
    # %               'theta_v','Tv','U','V','W','lat','lon','pitch','heading',...
    # %               'roll','Ts','alpha','beta','h','palt','mc','qc','wa','ws',...
    # %               't_sys','galt','nsv','ewv','vv','p','q',...
    # %               'r','axb','ayb','azb','azg','ata','speed_gnd'};
    var_bahamas =  ['PS','RELHUM','ABSHUM','MIXRATIO','TAS','TAT','TD','THETA',
                   'THETA_V','TV','U','V','W','IRS_LAT','IRS_LON','IRS_THE',
                   'IRS_HDG','IRS_PHI','TS','ALPHA','BETA','H','HP','MC','QC',
                   'WA','WS','IRS_ALT']
       
    
    #%% BAHAMAS essentials 
    # Bahamas class
    HALO_Devices_cls=HALO_Devices(cfg_dict)
    Bahamas_cls=BAHAMAS(HALO_Devices_cls)
    #bahamas dataset
    bahamas_ds =   xr.open_dataset(bahamas_file)
    
    #   List Bahamas variables in file
    vars_in_bahamas = bahamas_ds.variables
    
    #   time
    timename_use = Bahamas_cls.replace_bahamas_varname('TIME',
                                                       vars_in_bahamas)
    bahamas_time = pd.DatetimeIndex(np.array(bahamas_ds[timename_use]))
    
    #   location
    heightname_use = Bahamas_cls.replace_bahamas_varname('IRS_ALT',
                                                         vars_in_bahamas)
    h_GPS = pd.Series(bahamas_ds[heightname_use],index=bahamas_time[:])
    
    #   flight data
    varname_use = Bahamas_cls.replace_bahamas_varname('IRS_PHI',
                                                      vars_in_bahamas)
    roll_angle   = pd.Series(bahamas_ds[varname_use],index=bahamas_time[:])
    varname_use = Bahamas_cls.replace_bahamas_varname('IRS_THE',
                                                      vars_in_bahamas)
    pitch_angle = pd.Series(bahamas_ds[varname_use],index=bahamas_time[:])

    #   Replace variable names
    varname_use_dict={}
    for var in var_bahamas:
        varname_use_dict[var] = Bahamas_cls.replace_bahamas_varname(var,
                                                            vars_in_bahamas)
    
    #    Get maximum altitude 
    alt_max = h_GPS.max()

    # %% Adjust time series
    print('Adjust time series')
    # round times to avoid numerical deviations
    t_both              = pd.DatetimeIndex.intersection(bahamas_time,
                                                        radar_time)
    bahamas_time_series = pd.Series(data=range(len(bahamas_time)),
                                    index=bahamas_time)
    radar_time_series   = pd.Series(data=range(len(radar_time)),
                                    index=radar_time)
    bahamas_int_idx     = bahamas_time_series.reindex(
                                radar_time_series.index.values)
    if t_both.shape[0]==0:
        raise AssertionError('Bahamas and mira do not match.',
                             ' Did you select the correct files?')
    time_max=t_both.shape[0] #10000# for faster tests
    
    #%% Edit Data
    print('Edit Data')
    # Read essentials
    radar_range = radar_ds['range']

    # Read data of variables to be edited
    radar_data = {}
    for var in var_edit:
        #print("Include Radar ",var)
        radar_data[var] = radar_ds[var].loc[t_both]
        radar_data[var] = radar_data[var].fillna(float("-Inf"))
        radar_data[var]["time"]=t_both
    
    
    #% Adjust to common time steps
    h_GPS        = h_GPS.loc[t_both]
    roll_angle   = roll_angle.loc[t_both]
    pitch_angle  = pitch_angle[t_both]

    #%% Modify Bahamas Data
    #% Read data from NetCDF
    bahamas_data = {}
    bahamas_int_idx=bahamas_int_idx.dropna()
    for var in var_bahamas:
        bahamas_data[var] = bahamas_ds[var][bahamas_int_idx.values.astype(int)]
    print("bahamas_data allocated and cutted")
    
    #%% Correct Flight Attitude
    print('Correct Attitude')
    # Look for roll angles larger than specified deg, i.e. turning
    # Create flag value for turning
    print("Roll threshold: ",cfg_dict["roll_threshold"])
    turning_tracks  = roll_angle.loc[abs(roll_angle)>\
                                     float(cfg_dict["roll_threshold"])]
    turning_int_idx = bahamas_time_series.loc[turning_tracks.index].values
    turning_flag    = (abs(roll_angle)>\
                       float(cfg_dict["roll_threshold"])).astype(int)
    # % Round up to next 100 m
    alt_max = int(alt_max/100)*100+100
    
    # define vertical grid for variables
    if alt_max<=14000:  # keep this for compatibility between flights
        z_Grid = np.arange(0,14000,30)
    else:               # if aircraft ceiling was higher, extend vertical coord.
        z_Grid = np.arange(0,alt_max+30,30)
    
    radar_data_corr={}
    #print("Regrid radar")
    if not remove_side_lobes:
        # Remove data from side lobes during turns
        for var in var_edit:
            #radar_no_side_lobes = radar_data[var].loc[turning_int_idx]
            #radar_no_side_lobes = data_radar_no_side_lobes[var].iloc[:,0:len(Z_grid)]
            
            # % Regrid radar data to acount for flight attitudes
            print("Regrid radar variable:",var)
            radar_data_corr[var]=regrid_flight_angles(radar_range,
                                                 roll_angle,
                                                 pitch_angle,
                                                 h_GPS,z_Grid,
                                                 radar_data[var],
                                                 time_max)
    else:   
            ## !!!!!! This will follow laterly
            def correct_side_lobes():
                pass
            #correct_side_lobes()
   
    
    #%% Initialize netCDF file
    import netCDF4 as nc4
    nc_attributes=nc4.Dataset(bahamas_file).__dict__
    #Set format to bahamas_file format, but "NETCDF4_CLASSIC" would be better
    nc_attributes["Format"]=nc4.Dataset(bahamas_file).data_model
    #
    outfile_ds=xr.Dataset()
    outfile_time=xr.DataArray(t_both[0:time_max])#.astype(np.int64)
    
    vertical_coordinate= z_Grid
    outfile_height = xr.DataArray(vertical_coordinate)
    # Adjust name
    outfile_ds=outfile_ds.expand_dims({"time":outfile_time,"height":outfile_height})
    outfile_ds.time.attrs["long_name"]     = "Seconds since 01.01.1970 00:00 UTC"
    #outfile_ds.time.attrs["units"]         = "Seconds"
    outfile_ds.time.attrs["axis"]          = "T"
    outfile_ds.height.attrs["long_name"]   = "Height above MSL"
    outfile_ds.height.attrs["name"]        = "height"
    #outfile_ds.height.attrs["units"]       = "m"
    outfile_ds.height.attrs["axis"]        = "Y" 
    
    #print("Check reference dataset. To be deleted")
    #reference_ds=xr.open_dataset(cfg_dict["radarOut_dir"]+"mira_20200131_1515_g40_v0.2.nc")
    #produced_ds=xr.open_dataset(cfg_dict["radar_outf"])
    
    #%%Copy Unmodified Variables
    print('Copy Unmodified Variables')
    # Copy one-dimensional radar variables
    for var in var_copy:
        if var in radar_ds.keys():
            #print(var, "is in radar_ds")
            # keep in mind different types of variables (single ints or time_series)
            if radar_ds[var].size==1: #single int
                outfile_ds[var]=radar_ds[var]
            elif radar_ds[var].coords.dims[0]=="time":# only time dimension
                new_data_array=radar_ds[var][0:time_max]
                new_data_array=new_data_array.assign_coords({
                                            "time":outfile_ds.coords["time"]})
                outfile_ds[var]=new_data_array  
                outfile_ds[var].attrs=radar_ds[var].attrs               
            else:
                raise AssertionError('Unexpected dimensions.',
                                     ' Please consider listing the variabel ',
                                     var,' under ''var_edit''.')
                
    # Copy and slightly modify global attributes from radar standard file 
    import campaign_netcdf as Campaign_netCDF
    
    CPGN_netCDF=Campaign_netCDF.CPGN_netCDF()
    outfile_ds = CPGN_netCDF.copy_modify_netCDF_global_att(radar_fname,
                                                           outfile_ds,
                                                           cfg_dict)
    
    
    
    #%% Write Bahamas data to NetCDF
    print('Write Bahamas data to NetCDF')
    #nc_attributes=nc4.Dataset(bahamas_file).__dict__  
    bahamas_copy_ds=bahamas_ds.copy()
    # Rename time dimension
    if "tid" in [*bahamas_copy_ds.dims.keys()]:
        bahamas_copy_ds=bahamas_copy_ds.rename_dims({"tid":"time"})
    elif "utc_time" in [*bahamas_copy_ds.dims.keys()]:
        bahamas_copy_ds=bahamas_copy_ds.rename_dims({"tid":"time"})
    else:
        raise ValueError('No time dimension named tid or',
                         'utc_time is in the dataset',
                         "Recheck the dimensions",bahamas_copy_ds.dims)
    for var in var_bahamas:
        #print("BAHAMAS var:",var)
        # Remove global fill value
        time_adj_var_ds=bahamas_copy_ds[var][\
                                    bahamas_int_idx.values.astype(int)]\
                                        [0:time_max]
        time_adj_var_ds.encoding['_FillValue']=None
        # Add Atribute 'yrange'
        time_adj_var_ds.attrs['yrange']=[np.min(np.array(time_adj_var_ds)),
                                np.max(np.array(time_adj_var_ds))]
        
        #Set format to bahamas_file format, but "NETCDF4_CLASSIC" would be better
        #nc_attributes["Format"]=nc4.Dataset(bahamas_file).data_model
        time_adj_var_ds=time_adj_var_ds.assign_coords({"time":outfile_ds.time})
        #% Write variable to new file
        outfile_ds[var]=time_adj_var_ds
        #xr.DataArray(np.array(time_adj_var_ds[:]),
        #             coords=,dims=outfile_ds.dims,
        #attrs=time_adj_var_ds.attrs)
        #ncwriteschema(outfile,schemaCopy);
        
    
    #%% Write Radar data to NetCDF
    print('Write Radar data to NetCDF')
    radar_corr_copy_ds=radar_data_corr.copy()
    for var in var_edit:
        #print("Radar var:",var)
        #height_adj_var_ds=radar_copy_ds[var][:,0:len(vertical_coordinate)]
        radar_corr_array=np.array(radar_corr_copy_ds[var])
        radar_corr_array=np.where(radar_corr_array==-np.inf,
                             cfg_dict["missing_value"],
                             radar_corr_array)
        try:
            temporary_dataarray=xr.DataArray(radar_corr_array.T,
                                         coords=outfile_ds.coords,
                                         dims=outfile_ds.dims,
                                         attrs=radar_ds[var].attrs).T
        except:
            temporary_dataarray=xr.DataArray(radar_corr_array,
                                         coords=outfile_ds.coords,
                                         dims=outfile_ds.dims,
                                         attrs=radar_ds[var].attrs).T

        #% Add fill value information, need to be transposed
        temporary_dataarray.attrs["fill_value"]="NaN"
        
        outfile_ds[var]=temporary_dataarray
    
    #%% Calculate dBZ
    print('Calculate dBZ')
    if 'Zg' in var_edit:
        # Zg(Zg==missingvalue) = -Inf;
        # Calculate dBZ
        # using the emath extension to get complexe values and -inf instead of nan 
        dBZg = np.array(10*np.emath.log10(radar_data_corr['Zg'][:]))
        dBZg = dBZg.astype(np.float32)
        #% Only keep real part of array (imaginary numbers were created when
        #% taking log10 of -Inf: log10(-Inf) = Inf +      1.36437635384184i)
        dBZg = np.real(dBZg)
        #% And convert positive infinity back to negative infinity
        dBZg = np.where(dBZg==np.inf,-np.inf,dBZg)
        
    else:
        print('No Reflectivity Z found. Skipping dBZ calculation...')
        
    #%% Write dBZ to new radar file
    #schemaCopy = ncinfo(outfile,'Zg');
    #% Change name
    try:
        dBZg_ds=xr.DataArray(dBZg.astype(np.float32).T,
                         coords=outfile_ds.coords,
                         dims=outfile_ds.dims).T
    except:
        dBZg_ds=xr.DataArray(dBZg.astype(np.float32),
                         coords=outfile_ds.coords,
                         dims=outfile_ds.dims).T
    dBZg_ds.attrs={"long_name"  :'Reflectivity dBZg',
                   "units"      :' ',
                   "yrange"     : [np.nanmin(np.nanmin(dBZg)),
                                   np.nanmax(np.nanmax(dBZg))]}
    
    #% Write data to outfile
    outfile_ds["dBZg"]= dBZg_ds
    var_edit.append("dBZg")
    
    ###########################################################################
    ## Temporary Ze
    if 'Ze' in var_edit:
        # Zg(Zg==missingvalue) = -Inf;
        # Calculate dBZ
        # using the emath extension to get complexe values and -inf instead of nan 
        dBZe = np.array(10*np.emath.log10(radar_data_corr['Ze'][:]))
        dBZe = dBZe.astype(np.float32)
        #% Only keep real part of array (imaginary numbers were created when
        #% taking log10 of -Inf: log10(-Inf) = Inf +      1.36437635384184i)
        dBZe = np.real(dBZe)
        #% And convert positive infinity back to negative infinity
        dBZe = np.where(dBZe==np.inf,-np.inf,dBZe)
        #%% Write dBZ to new radar file
        #schemaCopy = ncinfo(outfile,'Zg');
        #% Change name
        try:
            dBZe_ds=xr.DataArray(dBZe.astype(np.float32).T,
                         coords=outfile_ds.coords,
                         dims=outfile_ds.dims).T
        except:
            dBZe_ds=xr.DataArray(dBZe.astype(np.float32),
                         coords=outfile_ds.coords,
                         dims=outfile_ds.dims).T
        dBZe_ds.attrs={"long_name"  :'Reflectivity dBZe (Hydrometeors)',
                   "units"      :' ',
                   "yrange"     : [np.nanmin(np.nanmin(dBZe)),
                                   np.nanmax(np.nanmax(dBZe))]}
    
        #% Write data to outfile
        outfile_ds["dBZe"]= dBZe_ds
        var_edit.append("dBZe")
        print("Ze and dBZe added to the files.")
        
    else:
        print('No Reflectivity Z found. Skipping dBZ calculation...')
        
    ###########################################################################
    LDRg = np.array(10*np.emath.log10(radar_data_corr['LDRg'][:]))
    LDRg = np.real(LDRg)
    LDRg=np.where(LDRg==np.inf,-np.inf,LDRg)
    LDRg_attrs={"long_name"  : "Linear Depolarization Ratio (LDR)",
                "units"      : "dB",
                "yrange"     : [np.nanmin(np.nanmin(LDRg)),
                                np.max(np.nanmax(LDRg))]}
    try:
        outfile_ds["LDRg"]=xr.DataArray(LDRg.astype(np.float32).T,
                                    coords=outfile_ds.coords,
                                    dims=outfile_ds.dims).T
    except:
        outfile_ds["LDRg"]=xr.DataArray(LDRg.astype(np.float32),
                                    coords=outfile_ds.coords,
                                    dims=outfile_ds.dims).T
    outfile_ds["LDRg"].attrs=LDRg_attrs
    #% Apply missing value to dBZ variable
    #dBZg(isinf(dBZg)) = missingvalue;
        
    
    #%% Flags
    print('Create flags')
    
    turning_flag_da=xr.DataArray(turning_flag[0:time_max],
                                 dims={"time":outfile_ds.dims["time"]})
    turning_flag_da.attrs={'units':' ',
        'long_name':'flag for roll angle > {threshold} deg, regarded as turn'\
            .format(threshold=cfg_dict["roll_threshold"]),
                'yrange':[0,1]}
    turning_flag_da=turning_flag_da.assign_coords({"time":outfile_ds.time})
        
    outfile_ds['curveFlag']=turning_flag_da
    #Add to global attributes
    if not remove_side_lobes:
        version_explanation=": data flipped, time offset corrected,"+\
                            " flight attitude corrected ("+used_device+\
                                "), sidelobes not removed"
    else:
        version_explanation=": data flipped, time offset corrected,"+\
                            " flight attitude corrected ("+used_device+")"
   
    outfile_ds.attrs["version_history"]='v'+str(cfg_dict["version"])+'.'+\
                                    str(cfg_dict["subversion"])+\
                                        version_explanation
    convert_att="converted by Henning Dorff on "
    actual_time_att=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    f_used       =", using: '"+__name__+".py"
    outfile_ds.attrs["conversion_information"]=convert_att+actual_time_att+\
                                                    f_used
    
    #%% Save the file
    print("Save file, this takes a while")
    
    if os.path.exists(cfg_dict["radar_outf"]):
        os.remove(cfg_dict["radar_outf"])
    
    # set compression of radar variables
    nc_compression=dict(zlib=True,complevel=1,dtype=np.float64)
    nc_encoding= {radar_var:nc_compression for radar_var in var_edit}
    outfile_ds["time"]=outfile_ds["time"].astype(np.int64)
    outfile_ds.to_netcdf(path=cfg_dict["radar_outf"],mode="w",format="NETCDF4",
                         engine="netcdf4",
                         encoding=nc_encoding)
    
    outfile_ds.close()
    return None
    #%------------- END OF CODE --------------
    
def perform_att_comb(convertmarker,flight,cfg_dict):
    
    """
    ------ Information
    Radar Attitude Combination
    Radar Bahamas Combination - Combines HAMP radar data with Bahamas data
    %   Syntax:  runRadarBahamasComb
    %               Adjust file location and desired conversion in header!!
    %
    %   Other m-files required: listFiles.m, HaloRadarBahamasComb.m,
    %                           HaloRadarBahamasCombCorrectTime.m,
    %                           HaloRadarBahamasCombCorrectTimeAngles.m
    %   See also: 
    %
    %   Authors: Dr. Heike Konow (1),Henning Dorff (1)
    %   (1) Meteorological Institute, Hamburg University
    %   email address: henning.dorff@uni-hamburg.de
    %   Website: http://www.mi.uni-hamburg.de/
    %   
        MatLab-Version (Heike Konow)
        --------------------------------------
        January 2014; Last revision: June 2015
    	March 2017: added processing for cases with multiple radar files
    				during one flight (only subversion 2)
        August 2019: restructured file in preparation for EUREC4A: use 
                    SMART attitude data for first conversion
        --------------------------------------
        Python-Version (Henning Dorff)
        February 2021: Rewritten to Python for later easier implementation
    
        
    -----------
    Parameters
    ----------
    convertmarker : TYPE
        DESCRIPTION.
    flight : TYPE
        DESCRIPTION.
    cfg_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """ 
    
    #% Set version number
    version_number = '0';
    #------------- BEGIN CODE --------------
    #%%
    #% Get flight dates to use in this program
    
    radar_dir    = cfg_dict["device_data_path"]+'radar/'
    bahamas_dir  = cfg_dict['device_data_path']+'bahamas/'
    radarOut_dir = cfg_dict['device_data_path']+'radar_mira/'
    if not os.path.exists(radarOut_dir):
        os.makedirs(radarOut_dir)
    cfg_dict["radar_dir"]=radar_dir
    cfg_dict["bahamas_dir"]=bahamas_dir
    cfg_dict["radarOut_dir"]=radarOut_dir
    ### According to convertmarker rotate radar coordinate frame with 
    ###(1) bahamas dataset angles or (2) smart
    
    #%% Subversion 1: flight angles corrected with bahamas
    if convertmarker==1:
        convert_device="BAHAMAS"
    else:
        convert_device="SMART"
    print('=================================')
    print('Correct flight attitude without side lobes removal')
    print('using ',convert_device,' attitude data')
    print('=================================')

        #   List relevant files
    radar_fnames = glob.glob(radar_dir+"*"+str(flight)+'*.nc')
    print("RADAR files",radar_fnames)
    #% If no files were found, try mmclx
    if not radar_fnames:
        try:
            radar_fnames = glob.glob(radar_dir+'*.mmclx')
            print("Found following mira files:",radar_fnames)
            
        except:
            raise FileNotFoundError(
                                "No radar data is found for flight ",flight) 
    
    else:
        #   Start processing
        print('Start Attitude Coordinate Frame Rotation')
    
        for radar_fname in radar_fnames:
            print(radar_fname)
            if convertmarker==1:
                correct_att_bahamas(radar_fname,version_number,
                                radarOut_dir, cfg_dict)
            else:
                correct_att_smart(radar_fname,version_number,
                                  radarOut_dir,cfg_dict)
        print('Finished Coordinate Frame Rotation: Attitude Correction')
    
    #Check: Look for files
    #fileNames = glob.glob(radarOut_dir+version_number+'.2.nc');
    #print('Found the following .2 version files:')
    #print(fileNames)
            #% Loop files if multiple files from one flight exist
            #for j=1:length(fileNameUse)
                #% Concatenate path and file name
                #RadarFile = fileNameUse{j};

                #disp(['  file: ' fileNameUse{j}])
                #fprintf('%s\n','')

                #% Combine radar data with Smart data
                #radarCorrectAtt_smart(RadarFile,versionNumber,
                                        #radarOutDir, missingvalule,'nolobes')
    #fprintf('%s\n','')
        #disp('Finished processing')

        #% Look for files
        #fileNames = listFiles([radarOutDir '*' versionNumber '.1.nc']);
        #fprintf('%s\n','')
    
        #% Display
        #%     disp('Found the following .2 version files:')
        #%     fprintf('\t%s\n',fileNames{:})
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def run_att_correction(flight_dates,cfg_dict,use_smart=False):
    """
    

    Parameters
    ----------
    flight_dates : pd.Series
        Series with index of flight number and date as value.
    cfg_dict : dict
        .
    use_smart : boolean
        D
    Returns
    -------
    None.

    """
    if isinstance(flight_dates,pd.Series):
        flight_dates=flight_dates.to_dict()
    for flight in flight_dates.values():
        # check if bahamas exists
        
        # Set path to bahamas file
        
        bahamasDir = cfg_dict["device_data_path"]+"bahamas/"
        smartDir   = cfg_dict["device_data_path"]+"smart/"
        fname      = "*"+str(flight)+"*"
        print(bahamasDir+fname)
        if not use_smart:
            # Look for files from flight if BAHAMAS is not available, 
            # SMART will be used
            if (len(glob.glob(bahamasDir+fname,recursive=True)) > 0):
                convertmarker=1
                # temporary
                print("Use bahamas dataset for attitude.")
            elif not (len(glob.glob(bahamasDir+fname,recursive=True)) == 0)\
                    and (len(glob.glob(smartDir+fname),recursive=True) > 0):
                print("Bahamas is not available.",
                      "Use smart data for attitude correction.")      
                convertmarker=2
            else:
                raise FileNotFoundError(
                        "No aircraft attitude data. Can't convert radar data")
        else:
            print("You set to use SMART data to transform the radar attitude.")
            convertmarker=2
        perform_att_comb(convertmarker,flight,cfg_dict)
