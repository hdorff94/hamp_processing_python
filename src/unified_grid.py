# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 12:07:07 2021

@author: u300737
"""

import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd
import xarray as xr

import campaign_netcdf as cpgn_nc
import Performance
performance=Performance.performance()

###############################################################################
#%% Standard functions
#def get_timeidx():
#    pass
#def get_heightidx():
#    pass

def transfer_data(uni_data,instr_var_data,height):
    """
    Transfer the data and insert it onto the unified grid level

    Parameters
    ----------
    uni_data : TYPE
        the empty array to be filled.
    instr_var_data : pd.DataFrame/Series
        pandas df/series of variable to be inserted on unified grid.
    height : pd.Series
        pandas Series containing the height levels of given data.
    ind_time : pd.DatetimeIndex
        pandas DatetimeIndex of unified 1Hz grid.

    Returns
    -------
    uni_data : TYPE
        DESCRIPTION.

    """
    uni_data_index  = uni_data.index
    uni_data_columns= uni_data.columns
    
    uni_data=np.array(uni_data)
    
    # Only take values were height is given
    height.index=range(height.shape[0])
    height=height.dropna()
    instr_var_data=np.array(instr_var_data.iloc[height.index])
    
    uni_data[np.array(height.index),np.array(height)]=instr_var_data
    
    uni_data=pd.DataFrame(data=uni_data,
                          index=uni_data_index,
                          columns=uni_data_columns)
    return uni_data

def sonde_remove_height_increase(height_series):
    """
    If sonde height does not decrease continually, delete values in beginning
    of profile (close to aircraft). Height allocation is then not reliable.

    Parameters
    ----------
    height_series : pd.Series
        Initial series of heights from dropsondes.

    Returns
    -------
    height_series_purged : pd.Series
        Filtered series of heights only containing decreasing values.
    

    """
    
    nfirst=20
    
    # Copy variable
    height_series_purged    = height_series.copy()
    height_series_purged    = height_series_purged.dropna()
    #ind_non_nan_heights=~height_series_purged.isnull()
    
    # Calculate differences between neighbouring values
    # and identify increasing values
    height_diffs            = height_series_purged.diff()
    idx_height_increase     = height_series_purged.loc[height_diffs>0].index
    
    ind_height_increase=pd.Series()
    
    if not idx_height_increase.shape[0]==0:
        for idx in idx_height_increase:
            ind_height_increase=ind_height_increase.append(pd.Series(
                        height_diffs.index.get_loc(idx))) 
    
    # If all height increases are below first "nfirst" values --> problem,
    # it is dealed with that later
    if (ind_height_increase.shape[0]>0) and \
        (sum(ind_height_increase>nfirst)==ind_height_increase.shape[0]):        
        height_series_purged.iloc[ind_height_increase+1]=np.nan#
    elif ind_height_increase.shape[0]>0:
        # Delete values from start to last height increase above first
        height_series_purged.iloc[0:max(\
                    ind_height_increase[ind_height_increase<=nfirst])+1]=np.nan
    return height_series_purged

def filter_spikes(data,spike_threshold=None):
    # if no threshold is specified in input
    if spike_threshold==None:
        # set allowed difference to half of overall data range in profile
        threshold=0.5*abs(data.max()-data.min())
    else:
        threshold=spike_threshold
    differences=abs(data.diff())
    data[differences>threshold]=np.nan
    return data

###############################################################################
#%% Major functions in order to unify grid
def unifyGrid_bahamas(flight,cfg_dict,bahamas_vars_use):
    import measurement_instruments_ql as Measurement_Instruments_QL
    import campaign_time as Campaign_Time
    
    #%% Configurate work around
    date=str(flight)    
    
    interpolate_data=True
    allowed_gap_length=3000
    major_path=cfg_dict["campaign_path"]
    outpath = cfg_dict["device_data_path"]+"all_pickle/"
    fname_pickle   = "uniData_bahamas_"+date+".pkl"
    outfile = outpath+fname_pickle
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    else:
        pass
    if os.path.exists(outfile):
        os.remove(outfile)
    #%% Load bahamas data
    cfg_dict["bahamas_dir"]=major_path+"Flight_Data/"+\
                                cfg_dict["campaign"]+"/Bahamas/"
    
    fname      = "*"+str(date)+"*.nc"
    bahamas_file=glob.glob(cfg_dict["bahamas_dir"]+fname,recursive=True)[0]
    
    #   Between the bahamas data and observations time offsets can frequently appear
    #   throughout the campaigns.
    #   The resulting time offsets are listed in a look up table.
    
    #   Correct time offset and get offset (in seconds) for specific day 
    Time_cpgn=Campaign_Time.Campaign_Time(cfg_dict["campaign"],
                                          date)
    HALO_Devices_cls=Measurement_Instruments_QL.HALO_Devices(cfg_dict)
    
    Bahamas_cls=Measurement_Instruments_QL.BAHAMAS(HALO_Devices_cls)
    
    bahamas_ds=xr.open_dataset(bahamas_file)
    
    # List Bahamas variables in file
    vars_in_bahamas = bahamas_ds.variables
    
    # time
    timename_use   = Bahamas_cls.replace_bahamas_varname('TIME',
                                                         vars_in_bahamas)
    heightname_use = Bahamas_cls.replace_bahamas_varname('IRS_ALT',
                                                         vars_in_bahamas)
    bahamas_time=bahamas_ds["TIME"]
    bahamas_alt=pd.Series(data=bahamas_ds[heightname_use],
                          index=pd.DatetimeIndex(np.array(bahamas_time[:])))
    
    bahamas_1hz=pd.date_range(start=bahamas_ds["TIME"][0].values,
                              end=bahamas_ds["TIME"][-1].values,
                              freq="1s")
    uni_height=pd.Series(np.arange(0,bahamas_alt.max()+30,30))
    uni_time=bahamas_1hz
    
    extra_info=pd.DataFrame(data=np.nan,
                            columns=["varname","units",
                                     "variable","unify_varname"],
                            index=np.arange(0,2,1))
    
    extra_info.iloc[0,:]=["time","seconds since 1970-01-01 00:00:00 UTC",
                          "time","uni_time"]
    extra_info.iloc[1,:]=["height","m","height","uni_height"]
    
    #Display information of current gridding period
    print("Flight Time from ",str(uni_time.time[0])," to ",
          str(uni_time.time[-1]))
    
    #Check if bahamas Data is 10 Hz:
    if len(bahamas_time)>=10*len(uni_time):
        bahamas_alt=bahamas_alt.resample("s").mean()        
        
    var_ind= [in_var in vars_in_bahamas for in_var in bahamas_vars_use]
    bahamas_vars_use_tmp=bahamas_vars_use.copy()
    bahamas_vars_use=np.array(bahamas_vars_use_tmp)[np.array(var_ind)].tolist()
    vars_in_bahamas_list=[*vars_in_bahamas]
    
    if bahamas_vars_use==[]:
        #Check if variables exist with different names
        bahamas_vars_use=bahamas_vars_use_tmp
        i=0
        for var in bahamas_vars_use:
            varNameUse=Bahamas_cls.replace_bahamas_varname(var,vars_in_bahamas)
            if varNameUse:
                vars_in_bahamas_list[i]=varNameUse
            i+=1
        del bahamas_vars_use_tmp
    else:
        # only read if there are variables
        true_names=Bahamas_cls.lookup_varnames(bahamas_vars_use)
    
    uni_time_index=pd.DatetimeIndex(uni_time[:])
    uni_time_index=uni_time_index.intersection(
                                    pd.DatetimeIndex(np.array(bahamas_time[:])))
    
    #Get closest height value index
    bahamas_alt=pd.Series(bahamas_alt,
                          index=uni_time_index)
    height_index=pd.Series(data=np.nan,index=bahamas_alt.index)
    
    print("find nearest height level")
    i=0
    
    tmp_bahamas_alt=pd.DataFrame(np.tile(bahamas_alt,(uni_height.shape[0],1)).T)
    uni_height_2d=pd.DataFrame(np.tile(uni_height,(bahamas_alt.shape[0],1)))
    difference=abs(tmp_bahamas_alt-uni_height_2d)
    closest_height=pd.Series(data=difference.idxmin(
                                    axis=1,skipna=True).values,
                             index=uni_time_index)
    
    del tmp_bahamas_alt, uni_height_2d
    print("Loop over variables and interpolate")
    globals()["uni_time"]=uni_time_index
    globals()["uni_height"]=uni_height
    for var in bahamas_vars_use:
        uni_df=pd.DataFrame(data=np.nan,index=uni_time,columns=uni_height)
    
        print("Variable: ",var)
        bahamas_series=pd.Series(data=np.array(bahamas_ds[var][:]),
                        index=pd.DatetimeIndex(np.array(bahamas_ds["TIME"][:])))
        
        # Replace missing values with
        bahamas_series=bahamas_series.replace(
            to_replace=float(cfg_dict["missing_value"]),
            value=np.nan)
        
        interpolate_flag = None
        interp_series    = None
        # If interpolate flag is set and there are gaps in data 
        # Preallocate a interpolation flag afterwards
        interpolate_flag=pd.Series(data=0,
                                   index=bahamas_series.index)
                
        if (interpolate_data) and (bahamas_series.isna().sum()>0):
            # if not all values are nans
            if bahamas_series.dropna().shape[0]>0:
                # Interpolate for Gap Filling but consider maximum gap length
                interp_series=bahamas_series.interpolate(method="time",
                                                         limit=allowed_gap_length)
                # Flag all values that are not anymore nans as interpolated
                was_flagged=interp_series.loc[bahamas_series.isna()].isna()==False
                interpolate_flag.loc[was_flagged.index]=1
        else:
            interp_series=bahamas_series
        
        # Handle attributes and variable names
        units_att        = bahamas_ds[var].attrs["units"]
        long_name_att    = bahamas_ds[var].attrs["long_name"]
        long_name_1d_att = long_name_att+"; 1d data"
        long_name_2d_att = long_name_att+"; 2d data"
        
        extra_info=extra_info.append({"varname":true_names[i],
                           "units":units_att,
                           "variable":long_name_2d_att,
                           "unify_varname":"uniBahamas_"+str(true_names[i])},
                          ignore_index=True)
        # Resample 10Hz data to 1Hz
        # currently picking the values and not regridding
        if interp_series is not None:
            interp_series    = interp_series.loc[uni_time_index]
            interpolate_flag = interpolate_flag.loc[uni_time_index]
            print("Transfer data onto unified grid")
            uni_df=transfer_data(uni_df, interp_series,
                                   closest_height)
        
        #Test if only one value per timestep is in variable
        one_value_test=uni_df.shape[1]-uni_df.isnull().sum(axis=1)
        if abs(1-one_value_test.mean(axis=0))<0.25:
            uni_df_1d = uni_df.sum(axis=1,skipna=True)
            extra_info=extra_info.append(
                        {"varname":str(true_names[i]),
                         "units":units_att,
                         "variable":long_name_1d_att,
                         "unify_varname":"uniBahamas_"+str(true_names[i])+"_1d"},
                        ignore_index=True)
            extra_info=extra_info.append(
                        {"varname":str(true_names[i])+"_int_flag",
                         "units":"",
                         "variable":long_name_att+" interpolation_flag",
                         "unify_varname":"uniBahamas_"+str(true_names[i])+"_interpolate_flag"},
                        ignore_index=True)
            globals()["uniBahamas_"+str(true_names[i])]=uni_df
            globals()["uniBahamas_"+str(true_names[i])+"_1d"]= uni_df_1d
            globals()["uniBahamas_"+str(true_names[i])+"_interpolate_flag"]= interpolate_flag
        else:
            raise Exception("Something went wrong with the bahamas files")
        del uni_df, uni_df_1d, interpolate_flag
        i+=1
    #del uni_time, uni_height
    print("Data Transfer done")
    
    #Save variables to pickle
    print("Save as pickle")
    pickle_variables={}
    pickle_path=cfg_dict["device_data_path"]+"all_pkl/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        
    for pkl_var in extra_info["unify_varname"]:
        pickle_variables[pkl_var]=globals()[pkl_var]
    pickle_variables["bahamas_extra_info"]=globals()["extra_info"]=extra_info
    with open(outpath+fname_pickle,'wb') as file:
        pickle.dump(pickle_variables,file,protocol=-1)
        
    with open(outpath+fname_pickle,'rb') as file:
        bahamas_var_dict=pickle.load(file)
    del bahamas_var_dict
    return uni_time,uni_height

def get_height_time_index_sonde(uni_time,uni_height,
                                sonde_time,sonde_height):
    
    a=uni_time.get_loc(sonde_time[0],method="ffill")
    b=uni_time.get_loc(sonde_time[-1],method="bfill")
    
    #length of relevant time interval
    t=b-a
    ind_time=uni_time[a:b]
    ind_height=pd.DataFrame(data=np.nan,
                            index=ind_time,
                            columns=["ind_sonde","ind_height"])
    sonde_time=sonde_height.index
    sonde_time=sonde_time.drop_duplicates()
    j=0
    
    for i in np.arange(a+j,b):
        
        sonde_idx=sonde_time.get_loc(uni_time[i],"nearest")
        ind_height_min=abs(sonde_height.iloc[sonde_idx]-uni_height).idxmin()
        
        if not np.isnan(ind_height_min):
            ind_height["ind_sonde"].iloc[j]=sonde_idx
            ind_height["ind_height"].iloc[j]=ind_height_min
        performance.updt(len(np.arange(a+1,b)),j)
        j+=1
            
    return ind_height

def unifyGrid_dropsondes(flight,uni_df,
                         uni_time,uni_height,
                         cfg_dict,sonde_vars):
    """
    unifyGrid_dropsondes - Transfer dropsonde data to uniform grid
    Read the data from original data files and do some quality checks
    (remove increasing height, remove spikes), interpolate data gaps. The
    checked data is then transfered to the uniform grid and some extra
    information is saved to be used later for netCDF file generation.

    In general, three types of data are created (with rh as example):
       - uniSonde_rh:       a height time matrix with measurements filled at the
                           exact time/height point as they occured
       - uniSonde_rh_inst:  a height time matrix with measurements filled
                           the height they occured but with an assumed
                           instantaneous drop, i.e. entire profile with
                           only one time stamp)
       - uniSonde_rh_sondes:  a matrix with height/sonde_number dimensions;
                           all sondes on the uniform height grid but
                           directly in succession

    In addition, an interpolation flag is added for each variable. This has
    the suffix _intFlag.

    Original MatLab-Version    
    Author: Dr. Heike Konow
    Meteorological Institute, Hamburg University
    email address: heike.konow@uni-hamburg.de
    Website: http://www.mi.uni-hamburg.de/
    June 2017; Last revision: April 2020
    
    Python-Version
    Author: Henning Dorff
    Meteorological Institute, Hamburg University
    email address: henning.dorff@uni-hamburg.de
    Website: http://www.mi.uni-hamburg.de/
    April 2021
    
    Parameters
    ----------
    
    (Old)
    Inputs:
    pathtofolder -  Path to base data folder
    flightdate -    string yyyymmdd for data to be converted
    uniHeigh -      array for uniform height grid
    uniTime -       array for uniform time grid
    uniData -       matrix with uniform time/height grid
    sondeVars -     list of dropsonde variable names to convert

    (Old end)
    flight   : str
        flighdate as string in format YYMMDD
    uni_time : TYPE
        DESCRIPTION.
    uni_height : TYPE
        DESCRIPTION.
    cfg_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    None. 
    #data is saved in [pathtofolder 'all_pickle/uniData_dropsondes'flightdate'.pkl']
    """
    #------------- BEGIN CODE --------------
    test_plots=False
    #Sonde number to test
    no=2
    
    #No need to change this unless problems with interpolation occur
    interpolate=True
    outpath = cfg_dict["device_data_path"]+"all_pickle/"
    fname_pickle   = "uniData_dropsondes_"+flight+".pkl"
    outfile = outpath+fname_pickle
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    else:
        pass
    if os.path.exists(outfile):
        os.remove(outfile)
    
    extra_info=pd.DataFrame(data=np.nan,
                            columns=["varname","units",
                                     "variable","unify_varname"],
                            index=np.arange(0,1,1))
    
    extra_info.iloc[0,:]=["flightdate","","Date of flight","flightdate"]

    extra_info=extra_info.append({"varname":"time",
                                  "units":"seconds since 1970-01-01 00:00:00 UTC",
                                  "variable":"time",
                                  "unify_varname":"uni_time"},
                                 ignore_index=True)
    
    extra_info=extra_info.append({"varname":"height",
                                  "units":"m",
                                  "variable":"height",
                                  "unify_varname":"uni_height"},
                                 ignore_index=True)
    
    #%% Dropsonde data
    sonde_files=glob.glob(cfg_dict["device_data_path"]+"dropsonde/*"+flight+"*")
   
    # Preallocate 
    sonde_height_for_interp={}
    snd=1
    ind_height   = {}
    nan_idx_tmp  = {}
    data         = {}
    uni_df_sonde = {}
    uni_sonde_launchtime=pd.Series(index=["Sonde_"+"{:02d}".format(snd+1) for snd in range(len(sonde_files))])
    uni_df_sonde_instantan={}
    interpolate_df = {}
    for var in sonde_vars[0:2]:
        uni_df_sonde_instantan[var]=uni_df.copy()
        uni_df_sonde[var]=uni_df.copy()
        interpolate_df[var]=0*uni_df
            
    if len(sonde_files)==0:
        raise Exception("No dropsonde files found for ",flight)
        
    nan_sonde_number={}
                    
    for var in sonde_vars[0:2]:
        print("Variable: ",var)
        for sonde_file in sonde_files:
            if snd>len(sonde_files):
                snd=1
            sonde_no="Sonde_"+"{:02d}".format(snd)
            print("Sonde No.: ",sonde_no)
            sonde_ds=xr.open_dataset(sonde_file)
        
            # Variables are sorted other way round; from ground to cruising level
            # Read all variables inversely so that time is increasing
            sonde_time      = pd.DatetimeIndex(np.array(sonde_ds.time[::-1]))
            sonde_height    = pd.Series(np.array(sonde_ds.gpsalt[::-1]),
                                    index=sonde_time)
            #if snd==8:
            #    print("critical")
            # Find indices of nan entries
            nan_idx_tmp[sonde_no] = sonde_height.isnull()
            sonde_height_copy     = sonde_height.copy()
            sonde_height          = sonde_height.dropna()
        
            # Remove instances with height increase during drop
            sonde_height            = sonde_remove_height_increase(sonde_height)
        
            sonde_height            = sonde_height.dropna()
            sonde_height            = sonde_height[sonde_height>0]
            if interpolate:
                sonde_height_for_interp[sonde_no]=sonde_height
        
            # Write launch time to variable
            uni_sonde_launchtime[sonde_no]=pd.to_datetime(
                                            np.array(sonde_ds["launch_time"]))
        
            print("Get indices for dropsonde and unified time/height")
            ind_height[sonde_no]=get_height_time_index_sonde(uni_time,uni_height,
                                               sonde_time,sonde_height)
            ## If dropsondes were released
            data[sonde_no]={}
            if int(sonde_no[-2:])==9:
                print("debugging")
            tmp_data=pd.Series(
                    data=sonde_ds[var][::-1],
                    index=pd.DatetimeIndex(np.array(sonde_ds["time"][::-1])))
            if sonde_height.min()>5000:
                snd+=1
                continue
            #print("Variable: ",var)
            tmp_data=tmp_data.loc[nan_idx_tmp[sonde_no]==False]
            tmp_data=filter_spikes(tmp_data)
            # Replace first in value in profile in any case 
            # (is either nan or unplausible)
            tmp_data.iloc[0]=np.nan
            # Interpolate data if desired
            if interpolate:
                allowed_gap_length=10
                # Make sure that at least two not-nan valuves are in profile
                tmp_data=pd.DataFrame(tmp_data,columns=["data"])
                tmp_data["height"]=sonde_height_for_interp[sonde_no]
                tmp_data=tmp_data.dropna(subset=["height"])
                    
                if sum(tmp_data["data"].isnull())<=tmp_data.shape[0]-2:
                    #reassign tmp_data from series to dataframe with 2nd column
                    interpolate_flag = None
                    interp_series    = None
                    # If interpolate flag is set and there are gaps in data 
                    # Preallocate a interpolation flag afterwards
                    interpolate_flag=pd.Series(data=0,
                                   index=tmp_data["height"])
                    interp_tmp_data=tmp_data["data"].copy()
                    interp_tmp_data.index=tmp_data["height"]
                    if interp_tmp_data.isna().sum()>0:
                        # if not all values are nans
                        if interp_tmp_data.dropna().shape[0]>0:
                            # Interpolate for Gap Filling but 
                            # consider maximum gap length
                            interp_series=interp_tmp_data.interpolate(
                                                    method="index",
                                                    limit=allowed_gap_length)
                            # Flag all values that are not 
                            # anymore nans as interpolated
                            was_flagged=interp_series.loc[
                                interp_tmp_data.isna()].isna()==False
                            interpolate_flag.loc[was_flagged[was_flagged==1].index]=1
                            #Reindex back to time
                            interpolate_flag.index=tmp_data.index
                        else:
                            interp_series=interp_tmp_data
                        tmp_data["data"]=interp_series.values
                else:
                    tmp_data["data"]=np.nan
                    print("Measurement values of ",var, "not correct or existent.")
            # Transfer data onto unified grid at time according to times of
            # individual data points (i.e. not the same time for all
            # measurements of one dropsonde)
            launch_idx=uni_sonde_launchtime[sonde_no]
            
            print("Launch time:",launch_idx)
            
            data[sonde_no][var]=tmp_data["data"]
            
            print("Data onto unified grid at time according to individual",
                  "data points.")
            
            t_idx=ind_height[sonde_no].index
            h_idx=uni_height.iloc[ind_height[sonde_no]["ind_height"].astype(int)]
            uni_df_sonde[var].loc[t_idx,h_idx] = data[sonde_no][var].iloc[ind_height[sonde_no]["ind_sonde"].astype(int)].values
           
            # Transfer data onto unified grid at time according to  start
            # time of dropsonde (i.e. the same time for all measurements 
            # of one dropsonde)
            print("Data onto unified grid at time according to start time of",
                  "sondes (same time for all measurements of one sonde).")
            data_height_index=tmp_data["height"].values
            tmp_data_2=pd.Series(data=tmp_data["data"].values,index=data_height_index)
            ind_height_2=pd.Series()
            for h in range(uni_height.shape[0]):
                ind_height_2=ind_height_2.append(pd.Series(tmp_data_2.index.get_loc(uni_height.iloc[h],
                                                method="nearest")))
        
            uni_df_sonde_instantan[var].loc[launch_idx,:]=tmp_data_2.iloc[ind_height_2.values].values
            interpolate_df[var].loc[launch_idx,:]=interpolate_flag.iloc[ind_height_2.values].values
            
            # Check if entire sonde profile is filled with nans
            if uni_df_sonde_instantan[var].loc[launch_idx,:].dropna().shape[0]==0:
                # Add current sonde number to nan sonde index
                nan_sonde_number[sonde_no] = sonde_file
            
            if snd==len(sonde_files[:-1]):
                # Read units and long name
                units_temp = sonde_ds[var].attrs["units"]
                long_name_temp = sonde_ds[var].attrs["long_name"]
                
                # Add long name information
                long_name_temp_inst = long_name_temp+', instantaneous drop'
                long_name_temp_sondes = long_name_temp+', single sondes'
                
                # Write variable information
                            
                extra_info=extra_info.append({"varname":var,
                                              "units":units_temp,
                                              "variable":long_name_temp,
                                              "unify_varname":'uniSondes_'+var},
                                             ignore_index=True)
                
                extra_info=extra_info.append({"varname":var+'_inst',
                                              "units":units_temp,
                                              "variable":long_name_temp_inst,
                                              "unify_varname":'uniSondes_'+var+'_sondes'},
                                             ignore_index=True)
                
                extra_info=extra_info.append({"varname":var+'_sondes',
                                              "units":units_temp,
                                              "variable":long_name_temp_sondes,
                                              "unify_varname":'uniSondes_'+var+'_sondes'},
                                             ignore_index=True)
                
                extra_info=extra_info.append({"varname":var+'_intFlag',
                                              "units":'',
                                              "variable":long_name_temp+', interpolation flag',
                                              "unify_varname":'uniSondes_'+var+'_interpolate_flag'},
                                             ignore_index=True)
                
            snd+=1
    
        # Reduce variable data to only sondes
        uniData_sondes = uni_df_sonde_instantan[var].loc[uni_sonde_launchtime.values,:]
        uniData_sondes_flag = interpolate_df[var].loc[uni_sonde_launchtime.values,:]
            
        # If sonde file with only nans exist, reproduce this
        if uniData_sondes.dropna(axis=1).shape[0]==0:
            nanSondeidx = nan_sonde_number.keys()
            nan_sonde_launch_time=uni_sonde_launchtime[nanSondeidx]
            for nan_launch in nan_sonde_launch_time:
                uniData_sondes.loc[nan_launch,:]=int(cfg_dict["missing_value"])
                uniData_sondes_flag.loc[nan_launch,:]=int(cfg_dict["missing_value"])
        # Rename variables
        globals()["uniSondes_"+var]                      = uni_df_sonde[var]
        globals()["uniSondes_"+var+"_inst"]              = uni_df_sonde_instantan[var]
        globals()["uniSondes_"+var+"_sondes"]            = uniData_sondes
        globals()["uniSondes_"+var+"_interpolate_flag"]  = interpolate_flag
            
    #Generate sonde number array
    uni_sonde_no = uniData_sondes.shape[1]
    
    # Write variable information
    extra_info=extra_info.append({"varname":"uni_sonde_no",
                                 "units":'',
                                 "variable":'Dropsonde number',
                                 "unify_varname":'uniSondes_number'},
                                 ignore_index=True)
    
    extra_info=extra_info.append({"varname":'uni_sonde_launch_time',
                                 "units":'seconds since 1970-01-01 00:00:00 UTC',
                                 "variable":'Dropsonde launch time',
                                 "unify_varname":'uniSondes_launchtime'},
                                 ignore_index=True)
    
    # If test figures should be plotted
    if test_plots:
        print("function plotFigure not yet defined.")
        
        # Call plotting function --> not yet defined
        #plotFigure(pathtofolder, f)    
    
    #Save variables to pickle
    print("Save as pickle")
    globals()["uniSondes_number"]=uni_sonde_no
    globals()["uniSondes_launchtime"]=uni_sonde_launchtime
    globals()["flightdate"]=flight
    globals()["uni_time"]=uni_time
    globals()["uni_height"]=uni_height        
    pickle_variables={}
    pickle_path=cfg_dict["device_data_path"]+"all_pkl/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        
    for pkl_var in extra_info["unify_varname"]:
        pickle_variables[pkl_var]=globals()[pkl_var]
    pickle_variables["sonde_extra_info"]=extra_info
    with open(outpath+fname_pickle,'wb') as file:
        pickle.dump(pickle_variables,file,protocol=-1)
    print('Dropsonde unifygrid pickle file saved as \n:',outpath+fname_pickle)
    
    return None        
                    
def unifyGrid_radar(flight,uni_df,
                    uni_time,uni_height,
                    cfg_dict,radar_vars):
    """
    unifyGrid_radar - Transfer radar data to uniform grid

    Original MatLab-Version    
    Author: Dr. Heike Konow
    Meteorological Institute, Hamburg University
    email address: heike.konow@uni-hamburg.de
    Website: http://www.mi.uni-hamburg.de/
    June 2017; Last revision: April 2020
    
    Python-Version
    Author: Henning Dorff
    Meteorological Institute, Hamburg University
    email address: henning.dorff@uni-hamburg.de
    Website: http://www.mi.uni-hamburg.de/
    April 2021
    
    Old Parameters
    ----------
    
    (Old)
    Inputs:
    flightdate -    string yyyymmdd for data to be converted
    uniHeigh -      array for uniform height grid
    uniTime -       array for uniform time grid
    uniData -       matrix with uniform time/height grid
    sondeVars -     list of dropsonde variable names to convert

    (Old end)
    
    Parameters
    ----------
    flight   : str
        flighdate as string in format YYMMDD
    uni_time : TYPE
        DESCRIPTION.
    uni_height : TYPE
        DESCRIPTION.
    cfg_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    None. 
    #data is saved in 'all_pickle/uniData_radar+flight+.pkl'
    """
    #------------- BEGIN CODE --------------
    test_plots=False
    
    #No need to change this unless problems with interpolation occur
    interpolate=True
    outpath = cfg_dict["device_data_path"]+"all_pickle/"
    fname_pickle   = "uniData_radar_"+flight+".pkl"
    outfile = outpath+fname_pickle
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    else:
        pass
    if os.path.exists(outfile):
        os.remove(outfile)
    
    extra_info=pd.DataFrame(data=np.nan,
                            columns=["varname","units",
                                     "variable","unify_varname"],
                            index=np.arange(0,1,1))
    
    extra_info.iloc[0,:]=["flightdate","","Date of flight","flightdate"]

    extra_info=extra_info.append({"varname":"time",
                                  "units":"seconds since 1970-01-01 00:00:00 UTC",
                                  "variable":"time",
                                  "unify_varname":"uni_time"},
                                 ignore_index=True)
    extra_info=extra_info.append({"varname":"height",
                                  "units":"m",
                                  "variable":"height",
                                  "unify_varname":"uni_height"},
                                 ignore_index=True)
    
    #%% Dropsonde data
    
    # Preallocate 
    ind_height   = {}
    nan_idx_tmp  = {}
    data         = {}
    uni_df_radar = {}
    interpolate_df = {}
    for var in radar_vars:
        uni_df_radar[var]=uni_df.copy()
        interpolate_df[var]=0*uni_df
            
    radar_files=glob.glob(cfg_dict["device_data_path"]+"radar_mira/*"+flight+"*")
    if not len(radar_files)==0:
        # Take newest version
        version_numbers=[float(radar_file[-6:-3]) for radar_file in radar_files]                    
        newest_file=np.array(version_numbers).argmax()    
        
        radar_ds=xr.open_dataset(radar_files[newest_file])
        
        radar_time      = pd.DatetimeIndex(np.array(radar_ds.time[:]))
        radar_height    = pd.Series(np.array(radar_ds.height[:]))
        radar_state     = pd.Series(data=np.array(radar_ds.grst[:]),
                                    index=radar_time)
        for var in radar_vars:

            print(var)
            try:
                radar_df=pd.DataFrame(data=np.array(radar_ds[var][:]),
                                      columns=radar_height,
                                      index=radar_time)
            except:
                radar_df=pd.DataFrame(data=np.array(radar_ds[var][:].T),
                                      columns=radar_height,
                                      index=radar_time)
                radar_df=radar_df.interpolate(method="time",limit=1)
                
            # Discard data where radar state was not 13; i.e. local oscillator
            # not locked and/or radiation off
            radar_df.loc[radar_state[radar_state!=13].index]=np.nan
            uni_df_radar[var].loc[radar_time,:]=radar_df.iloc[:,0:uni_height.shape[0]]
            
            units_temp      = radar_ds[var].attrs["units"]
            long_name_temp  = radar_ds[var].attrs["long_name"]
            
            globals()["uniRadar_"+var]= uni_df_radar[var]
                
            extra_info=extra_info.append({"varname":var,
                                  "units":units_temp,
                                  "variable":long_name_temp,
                                  "unify_varname":"uniRadar_"+var},
                                 ignore_index=True)
        
    globals()["flightdate"]=flight
    globals()["uni_time"]=uni_time
    globals()["uni_height"]=uni_height        
    
    #Save variables to pickle
    print("Save Radar as pickle")
    pickle_variables={}
    pickle_path=cfg_dict["device_data_path"]+"all_pkl/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        
    for pkl_var in extra_info["unify_varname"]:
        pickle_variables[pkl_var]=globals()[pkl_var]
    pickle_variables["radar_extra_info"]=extra_info
    with open(outpath+fname_pickle,'wb') as file:
        pickle.dump(pickle_variables,file,protocol=-1)
    print('Radar unifygrid pickle file saved as \n:',outpath+fname_pickle)
    return None

#%% Radiometer
def unifyGrid_radiometer(flight,uni_df,
                         uni_time,uni_height,
                         cfg_dict,radiometer_vars,correct_radiometer_time=True):
    
    #------------- BEGIN CODE --------------
    #test_plots=False
    
    interpolate=True
    allowed_gap_length=30           # Allowed gap length for interpolation: 30s
    
    outpath = cfg_dict["device_data_path"]+"all_pickle/"
    fname_pickle   = "uniData_radiometer_"+flight+".pkl"
    outfile = outpath+fname_pickle
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    else:
        pass
    if os.path.exists(outfile):
        os.remove(outfile)
    
    extra_info=pd.DataFrame(data=np.nan,
                            columns=["varname","units",
                                     "variable","unify_varname"],
                            index=np.arange(0,1,1))
    
    extra_info.iloc[0,:]=["flightdate","","Date of flight","flightdate"]

    extra_info=extra_info.append({"varname":"time",
                                  "units":"seconds since 1970-01-01 00:00:00 UTC",
                                  "variable":"time",
                                  "unify_varname":"uni_time"},
                                 ignore_index=True)
    # Preallocate 
    tb_data_dict = {}
    tb_data_dict["performed_processing"]="No further processing done."
    uni_df_radiometer = {}
    interpolate_flag = {"183":None,
                      "11990":None,
                      "KV":None}
    # Define shift_comment_str for later nc-Infos
    shift_comment_series=pd.Series(data=[[],[],[]],index=interpolate_flag.keys())
        
    for var in radiometer_vars:
        
        # List files from specified date
        radiometer_files=glob.glob(cfg_dict["device_data_path"]+\
                                   "radiometer/"+var+"/*"+flight+"*")
        # Display var name
        print("Unify Radiometer Channel: ",var)
        
        # If no files were found, it's probably because the date is written in 
        # yymmdd format in file names instead of yyyymmdd
        if len(radiometer_files)==0:
            flight_backup=flight    
            flight=flight[2::]
            radiometer_files=glob.glob(cfg_dict["device_data_path"]+\
                                   "radiometer/"+var+"/*"+flight+"*")
            
                # Check if flight ended after 00z
            uni_time_str=uni_time.date[-1].strftime('%Y%m%d')[2::]
            if not uni_time_str==flight:
                file_name_2=glob.glob(cfg_dict["device_data_path"]+\
                                   "radiometer/"+var+"/*"+uni_time_str+"*")
                radiometer_files.append(file_name_2)
            flight=flight_backup
        
        # If radiometer files for day exists    
        if not len(radiometer_files)==0:
            # Loop all files from day which should work either way: 
            # if it is only one file and if it contains several files
            i=0
            for file in radiometer_files:
                radiometer_ds=xr.open_dataset(file)
                if "frequencies" in radiometer_ds.keys():
                    freq_var_name="frequencies"
                else:
                    freq_var_name="Freq"
            
                tb_data_dict[file]=pd.DataFrame(
                    data=np.array(radiometer_ds["TBs"][:].astype(float)),
                    index=pd.DatetimeIndex(np.array(radiometer_ds.time[:])),
                    columns=np.array(radiometer_ds[freq_var_name][:]).\
                        astype(float).round(2))
                #Concatenate the files
                if i==0:
                    tb_data=tb_data_dict[file].copy()
                else:
                    tb_data_fnames=[tb_data,tb_data_dict[file]]
                    tb_data=pd.concat(tb_data_fnames)
                i+=1
            #i
            if tb_data.shape[1]!=radiometer_ds[freq_var_name].shape[0]:
                tb_data=tb_data.loc[:,0:radiometer_ds[freq_var_name].shape[0]]
            
            ##### Time Correction Radiometer
            # the Radiometer data is provided in 4 Hz but time only specified
            # in seconds. Accordingly, data has duplicated indexes
            # secondly mean is applied
            tb_data=tb_data.resample("1s",convention="start").mean()
            # drop nan values 
            tb_data=tb_data.dropna(axis=0,how="all")
            # offset correction
            if correct_radiometer_time:
                time_shift_command=""
                #shift_comment_series.loc[var]=[]
                import processing_unified_grid as process_grid
                Radiometer_processing=process_grid.Radiometer_Processing(cfg_dict)
                Radiometer_processing.lookup_radiometer_timeoffsets()
                tb_data,shift_comment_series[var]=Radiometer_processing.\
                    shift_time_offsets(
                            tb_data,np.array(radiometer_ds[freq_var_name][:]),
                            shift_comment_series[var])
                for shift_comment in shift_comment_series.values:
                    if not len(shift_comment)==0:
                        time_shift_command="Radiometer time shifts adjusted."
                if not time_shift_command=="":    
                    if tb_data_dict["performed_processing"].startswith("No further"):
                        tb_data_dict["performed_processing"]=time_shift_command
                    else:
                        tb_data_dict["performed_processing"]=\
                            tb_data_dict["performed_processing"]+\
                                time_shift_command
            
            # remove times in the future and past
            tb_data=tb_data.loc[uni_df.index[0]:uni_df.index[-1]]
            
            ind_time_jumps=tb_data.index.to_series().diff()
            ind_time_jumps=ind_time_jumps.loc[ind_time_jumps>"1s"]
            tb_data.loc[ind_time_jumps.index,:]=np.nan
            #units_temp      = radiometer_ds["TBs"].attrs["units"]
            #long_name_temp  = "Brightness Temperature"
            
            #Preallocate unified radiometer channel as pd.DataFrame()
            if cfg_dict["fill_value"]=="nan":
                fill_value=np.nan
            
            uni_df_radiometer[var]=pd.DataFrame(data=fill_value,
                                           index=uni_time,
                                           columns=tb_data.columns)
            
            uni_df_radiometer[var].loc[tb_data.index]=tb_data.values
            
            if uni_df_radiometer[var].isna().all(axis=1).sum()>0:
                interpolate_flag[var]=pd.DataFrame(data=0,
                            columns=np.array(radiometer_ds[freq_var_name][:]),
                            index=uni_df_radiometer[var].index)
            
                to_interp_tmp_data=uni_df_radiometer[var].copy()
                if to_interp_tmp_data.dropna().shape[0]>0:
                    # Interpolate for Gap Filling but 
                    # consider maximum gap length
                    interp_tmp_data=to_interp_tmp_data.interpolate(
                                                    method="index",
                                                    limit=allowed_gap_length)
                    # Flag all values that are not 
                    # anymore nans as interpolated
                    was_flagged=interp_tmp_data.loc[\
                        to_interp_tmp_data.isna().all(axis=1)]\
                        .isna().all(axis=1)==False
                    
                    interpolate_flag[var].loc[was_flagged[\
                                                was_flagged==True].index]=1
                    #Reindex back to time
                    interpolate_flag[var].index=to_interp_tmp_data.index
                    uni_df_radiometer[var]=interp_tmp_data
                    gap_filling_command="Gaps filled."
        # if no data is available
        else:
            """
                ----> handles the case when still no data is found,
                to be added later on.
                
            """
            pass
    if "gap_filling_command" in locals():
        if tb_data_dict["performed_processing"].startswith("No further"):
            tb_data_dict["performed_processing"]=gap_filling_command
        else:
            tb_data_dict["performed_processing"]=\
                tb_data_dict["performed_processing"]+\
                    gap_filling_command

    # Remove data from flight maneouvers (Turn, Ascent, Descent)    
    uni_df_radiometer,tb_data_dict=Radiometer_processing.\
                        remove_turn_ascent_descent_data(uni_df_radiometer,
                                                        tb_data_dict)
    uni_df_radiometer_freq=[]
    for freq in radiometer_vars:
        uni_df_radiometer_freq.append(\
                    uni_df_radiometer[freq].columns.astype(float))
    uni_df_radiometer_freq=pd.Series(np.hstack([np.array(freq).round(2) \
                                      for freq in uni_df_radiometer_freq]))
        
    globals()["uniRadiometer"]=uni_df_radiometer
    globals()["uniRadiometer_freq"]=uni_df_radiometer_freq
    globals()["uniRadiometer_interp_flag"]=interpolate_flag
    
    extra_info=extra_info.append({"varname":"TB",
                                  "units":"K",
                                  "variable":"Brightness Temperature",
                                  "unify_varname":"uniRadiometer"},
                                 ignore_index=True)
    
    extra_info=extra_info.append({"varname":"freq",
                                  "units":"GHz",
                                  "variable":"channel center frequency",
                                  "unify_varname":"uniRadiometer_freq"},
                                 ignore_index=True)
    
    extra_info=extra_info.append({"varname":"interpolate_flag",
                                  "units":"",
                                  "variable":"Flag for interpolation",
                                  "unify_varname":"uniRadiometer_interp_flag"},
                                 ignore_index=True)
        
    globals()["flightdate"]=flight
    globals()["uni_time"]=uni_time
    globals()["shift_comment_series"]=shift_comment_series
    #Save variables to pickle
    print("Save radiometer as pickle")
    pickle_variables={}
    pickle_path=cfg_dict["device_data_path"]+"all_pkl/"
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        
    for pkl_var in extra_info["unify_varname"]:
        pickle_variables[pkl_var]=globals()[pkl_var]
    pickle_variables["radiometer_extra_info"]=extra_info
    pickle_variables["performed_processing"]=tb_data_dict["performed_processing"]
    with open(outpath+fname_pickle,'wb') as file:
        pickle.dump(pickle_variables,file,protocol=-1)
    print('Radiometer unifygrid pickle file saved as \n:',outpath+fname_pickle)
    #"""
    return None
    
    
    
    
def run_unify_grid(flightdates_use,cfg_dict):
    """
    

    Parameters
    ----------
    flightdates : TYPE
        DESCRIPTION.
    cfg_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    print("Function under progress")
     
    #%% Switches 
    # usually all set to True, but can be useful for debugging
    
    # Unify data onto common grid
    unify = True
    # Save data to netcdf
    savedata = True
    #redoBahamas = False

    # Load information on flight dates and campaigns
    all_flight_dates = cfg_dict["Flight_Dates"];

    t1 = flightdates_use[0];

    # Set path to base folder
    pathtofolder = cfg_dict["campaign_path"]#getPathPrefix getCampaignFolder(t1)];

    #%% Specify variables to consider

    # Bahamas
    bahamas_vars = ['MIXRATIO','PS','RELHUM','THETA','U','V','W','IRS_ALT',
                   'IRS_HDG','IRS_THE','IRS_PHI','IRS_LAT','IRS_LON','TS',
                   'IRS_GS','IRS_VV']

    # Radiometer
    radiometer_vars = ['183','11990','KV']

    # Radar
    radar_vars = ['dBZg','Zg','Ze','dBZe','LDRg','RMSg','VELg','SNRg']

    # Dropsondes
    sonde_vars = ['pres','tdry','dp','rh','u_wind','v_wind','wspd','wdir',
                  'dz','mr','vt','theta','theta_e','theta_v','lat','lon']

    #%% Data processing
    instruments_to_unify=eval(cfg_dict["instruments_to_unify"])
    #% Loop all dates
    if unify:
        for flight in flightdates_use:
            #% Return date
            print(flight)

            # Unify data on one common grid    
            # Bahamas
            # Redo unified bahamas data, otherwise only load

            if "bahamas" in instruments_to_unify:
                [uni_time,uni_height] = unifyGrid_bahamas(flight,cfg_dict,
                                                          bahamas_vars)
            else:
                filepath  = cfg_dict["device_data_path"]+"all_pickle/"
                pkl_fname = "uniData_bahamas_"+str(flight)+".pkl"
                try:
                    with open(filepath+pkl_fname,"rb") as pkl_file:
                        bahamas_dict=pickle.load(pkl_file)
                        uni_time=bahamas_dict["uni_time"]
                        uni_height=bahamas_dict["uni_height"]
                        #del bahamas_dict
                        print("Unified Bahamas already accessible and loaded.")
                
                except:
                    print("Bahamas pickle does not exist and need to be rebuilt")
                    [uni_time,uni_height] = unifyGrid_bahamas(flight,cfg_dict,
                                                              bahamas_vars)
                
            # Create empty variable according to unified grid
            uni_df = pd.DataFrame(data=float(cfg_dict["missing_value"]),
                                   index=uni_time,
                                   columns=uni_height)

            filepath  = cfg_dict["device_data_path"]+"all_pickle/"
            
            # Dropsondes
            if "dropsondes" in instruments_to_unify:
                pkl_fname = "uniData_dropsondes_"+str(flight)+".pkl"
                if not os.path.exists(filepath+pkl_fname):
                    # Dropsondes
                    print("Unify Dropsondes")
                    unifyGrid_dropsondes(str(flight),uni_df,
                                 uni_time,uni_height,
                                 cfg_dict,sonde_vars)
                else:
                    print("Unified Dropsondes are already accessible.")
            
            
            # Radar
            if "radar" in instruments_to_unify:
                pkl_fname = "uniData_radar_"+str(flight)+".pkl"
                if not os.path.exists(filepath+pkl_fname):
                    # Radar
                    print("Unify radar")
                    unifyGrid_radar(str(flight),uni_df,
                            uni_time,uni_height,
                            cfg_dict,radar_vars)
                else:
                    print("Unified Radar is already accessible.")
            
            # Radiometer
            if "radiometer" in instruments_to_unify:
                pkl_fname= "uniData_radiometer_"+str(flight)+".pkl"
                if not os.path.exists(filepath+pkl_fname):
                    unifyGrid_radiometer(str(flight),uni_df,
                            uni_time,uni_height,
                            cfg_dict,radiometer_vars)
                else:
                    print("Unified Radiometer is already accessible.")
            #%% Start Exporting to netCDF
            file_format=".nc"
            if savedata:
                campaign_netcdf=cpgn_nc.CPGN_netCDF()
                # perform additional processing
                print("Processing flight on:",flight)
                for instr in instruments_to_unify:
                    print("Processing data from:",instr)
                    out_path=cfg_dict["device_data_path"]+"all_nc/"
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)
                    in_path=cfg_dict["device_data_path"]+"all_pickle/"
                    temporary_version=cpgn_nc.CPGN_netCDF.\
                        check_outfile_name_version_for_calibration(
                                                            cfg_dict,instr)
                    fname=instr+"_"+str(flight)+"_v"+temporary_version+\
                        "."+cfg_dict["subversion"]+file_format
                    outfile=out_path+fname
                    print(instr," ds will be stored as:",outfile)
                    infile=in_path+"uniData_"+instr+"_"+str(flight)+".pkl"
                    if os.path.exists(infile):
                        campaign_netcdf.prepare_pkl_for_xr_dataset(
                            infile,outfile,cfg_dict,instr)

    """
    #######
    ### to be continued!!!!!!!
    """
    return None