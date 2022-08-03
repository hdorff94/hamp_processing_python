# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 12:57:22 2021

@author: u300737
"""
import glob
import numpy as np
import os
import pandas as pd

import Performance
import sys


import xarray as xr

    
class CPGN_netCDF():
    def __init__(self):
        pass
    
    def copy_modify_netCDF_global_att(self,infile,outfile_ds,cfg_dict):
        """
        

        Parameters
        ----------
        infile : str 
            name of netcdf file.
        outfile_ds : xr.Dataset
            xarray Dataset of newly processed nc-file.
        flight_dates : dict
            dict with flight dates
        Returns
        -------
        outfile_ds.

        """

        #% Function for copying a netcdf global attributes to a new file
        
        #if not os.path.exists(cfg_dict["radar_outf"]):
        #    raise FileNotFoundError('File ',cfg_dict["radar_outf"],
        #                            ' does not exist. Please create first.')

        #% Read flight date and numbers
        cpgn_name=cfg_dict["campaign"]
        
        # Read nc file information
        infile_ds           = xr.open_dataset(infile)
        infile_attrs        = infile_ds.attrs
        # Read date
        infile_t           = pd.to_datetime(np.array(\
                                            infile_ds["time"][:]),unit='s')
        infile_date        = infile_t.date[0]
        
        flight_mission     = cpgn_name
        
       
        ##### List global attributes
        # Replace information
        infile_attrs["location"]        = 'HALO aircraft, D-ADLR'
        infile_attrs["institute"]       = 'University of Hamburg'
        
        infile_attrs['authors']         = 'Lutz Hirsch, Heike Konow, Henning Dorff'
        infile_attrs['contact']         = 'lutz.hirsch@mpimet.mpg.de, \
                        heike.konow@uni-hamburg.de, henning.dorff@uni-hamburg.de'
        infile_attrs['mission']         = flight_mission;
        infile_attrs['flight number']   = [*cfg_dict["Flight_Dates_used"].keys()] 
        
        #% Write global attributes
        #cellfun(@(x,y) ncwriteatt(outfile,'/',x,y),glAttNames,glAttValues);
        outfile_ds.attrs=infile_attrs
        return outfile_ds
    
    def add_fill_missing(self,ds,cfg_dict,do_filling=False):
        """
        Add fill value/missing value information
        
        Parameters
        ----------
        ds : xr.Dataset
            dataset for which information need to be added.
        cfg_dict : dict
            config dictionary containing the relevant info for missing values.
        do_filling : Boolean
            whether missing values will be filled with filling value
        Returns
        -------
        ds.

        """
        for var in ds.keys():
            if do_filling:
                ds[var].attrs["missing"]=cfg_dict["missing_value"]
            
                import Processing_Unified_Grid as prcsuni
                ds=prcsuni.fill_missing_values(ds,cfg_dict)
        return ds
    
    def prepare_pkl_for_xr_dataset(self,pkl_file,outfile,cfg_dict,
                                    instrument="bahamas"):
        """
        Prepare the pickle data to xr Dataset that can easily be saved as nc
        
        Parameters
        ----------
        pkl_file : str 
            name of pkl file to load and to convert data in xr.Dataset.
        
        Returns
        -------
        None.

        """
        dimensions={"uni_time":False,
                    "uni_height":False,
                    "uni_sonde_no":False,
                    "uniRadiometer_freq":False}
        
        import pickle
        
        
        with open(pkl_file,"rb") as pkl:
            device_dict=pickle.load(pkl)
        
        # Write var names to variable
        var_names=device_dict.keys()
        dim_sizes={}
        if not instrument=="dropsondes":
            extra_info_arg=instrument+"_extra_info"
        else:
            extra_info_arg="sonde"+"_extra_info"
        extra_info_tmp=device_dict[extra_info_arg]
        extra_info_tmp.index=extra_info_tmp["unify_varname"]
        del device_dict[extra_info_arg]
        # Loop over all dimensions
        coords_dict={}
        for dim in dimensions.keys():
            if dim in var_names:
                #Get size of current dimension
                dim_sizes[dim]=device_dict[dim].shape
                dimensions[dim]=True
                if dim.startswith("uni_"):
                    dim_name=dim[4::]
                else:
                    dim_name=dim
                if isinstance(device_dict[dim],pd.DatetimeIndex) or \
                    isinstance(device_dict[dim],pd.Series):    
                    coords_dict[dim_name]=device_dict[dim].values
                else:
                    coords_dict[dim_name]=device_dict[dim]
        ds=xr.Dataset()
        ds=ds.assign_coords(coords_dict)
        extra_data={}
        for var in var_names:
            print(var)
            if var=="uniSondes_launchtime":
                print("debugging")
            if not isinstance(device_dict[var],str):
                if isinstance(device_dict[var],dict):
                    temporary_df=pd.DataFrame(
                                    data=np.nan,
                                    columns=device_dict[[*coords_dict.keys()][1]],
                                    index=pd.DatetimeIndex(coords_dict["time"]))
                    merged_df_created=False
                    for key in device_dict[var].keys():
                        if len(device_dict[var][key].shape)>=2:
                            if not instrument=="radar" \
                                or not instrument=="dropsonde":
                                    for col in device_dict[var][key].columns:
                                        #freq=device_dict[[*coords_dict.keys()][1]][i]
                                        temporary_df[col]=device_dict[var]\
                                                            [key][col]
                            merged_df_created=True
                        elif len(device_dict[var][key].shape)==1:
                            if not var[4::] in coords_dict.keys() \
                                and not var in coords_dict.keys():
                                for coord in coords_dict.keys():
                                    if device_dict[var][key].shape[0]==\
                                        len(coords_dict[coord]):
                                        
                                        da_var_name=extra_info_tmp["varname"][var]
                                        temporary_df[coord]=device_dict[var][key]
                                        ds[da_var_name]=xr.DataArray(temporary_df,
                                                  dims=coord)                
                                        
                    if merged_df_created:                    
                        dims=coords_dict.keys()
                        da_var_name=extra_info_tmp["varname"][var]
                  
                        ds[da_var_name]=xr.DataArray(temporary_df,
                                                  dims=dims)                
                        ds[da_var_name].attrs["long_name"]= extra_info_tmp[\
                                                            "variable"].loc[var]
                        ds[da_var_name].attrs["units"]= extra_info_tmp[\
                                                            "units"].loc[var]
                else:
                    if isinstance(device_dict[var],int):
                        continue
                    
                    if len(device_dict[var].shape)>=2:
                        if not instrument=="radar" and not instrument=="dropsondes":
                            da_var_name=extra_info_tmp["varname"][var]+"_mat"
                        else:
                            da_var_name=extra_info_tmp["varname"][var]
                        if not isinstance(da_var_name,str) and not \
                            isinstance(da_var_name,int):
                            if len(da_var_name)>1:
                                da_var_name=da_var_name[-1]
                        ds[da_var_name]=xr.DataArray(
                                        device_dict[var],dims=coords_dict.keys())
                        
                        long_name_att=extra_info_tmp["variable"].loc[var]
                        unit_name_att=extra_info_tmp["units"].loc[var]
                        #######################################################
                        if not isinstance(long_name_att,pd.Series):
                            ds[da_var_name].attrs["long_name"]  = \
                                            long_name_att
                        else:
                            ds[da_var_name].attrs["long_name"]  = \
                                            long_name_att[1]    
                        #######################################################
                        if not isinstance(unit_name_att,pd.Series):
                            ds[da_var_name].attrs["units"]      = \
                                            unit_name_att
                        
                        else:
                            ds[da_var_name].attrs["long_name"]  = \
                                            unit_name_att[1]    
                        
                    elif len(device_dict[var].shape)==1:
                        #if not var[4::] in coords_dict.keys() and not \
                        #    var in coords_dict.keys():
                            
                            for coord in coords_dict.keys():
                                if device_dict[var].shape[0]==\
                                    len(coords_dict[coord]):
                                    
                                    da_var_name=extra_info_tmp["varname"][var]
                                    ds[da_var_name]=xr.DataArray(
                                                    data=device_dict[var].values,
                                                    dims=coord)                
                                    ds[da_var_name].attrs["long_name"]  = \
                                        extra_info_tmp["variable"].loc[var]
                                    ds[da_var_name].attrs["units"]      = \
                                        extra_info_tmp["units"].loc[var]
                    
            else:
                extra_data[var]=device_dict[var]
        print("Dataset assigned")
        
        if instrument=="bahamas":
            import measurement_instruments_ql as Measurement_Instruments_QL
            HALO_Devices_cls=Measurement_Instruments_QL.HALO_Devices(cfg_dict)
            bahamas_cls=Measurement_Instruments_QL.BAHAMAS(HALO_Devices_cls)
            bahamas_nc_vars={bahamas_cls.nc_var_names[k]:k \
                             for k in bahamas_cls.nc_var_names}
            
        else:    
            # If instrument is not bahamas, add geo informations to dataset
            # position 138 only works for current path description
            file_start=pkl_file.rfind("/")+9 #9 because of uniData_ having 8 char
            file_end=pkl_file[-13:] # 13 due to "_YYYYMMDD.pkl"
            bahamas_pkl_fname=pkl_file[0:file_start]+"bahamas"+file_end
            with open(bahamas_pkl_fname,"rb") as pkl:
                print("Add Geo-Reference Data from Bahamas")
                bahamas_dict=pickle.load(pkl)
                bahamas_info=bahamas_dict["bahamas_extra_info"]
                bahamas_info.index=bahamas_info["unify_varname"]
                for geo_var in ["lat","lon","alt"]:
                    bahamas_geo_var="uniBahamas_"+geo_var+"_1d"
                    ds[geo_var]=xr.DataArray(data=bahamas_dict[bahamas_geo_var].T,
                                              dims="time")
                    ds[geo_var].attrs["long_name"]  = \
                                    bahamas_info["variable"].loc[bahamas_geo_var]
                    ds[geo_var].attrs["units"]      = \
                                    bahamas_info["units"].loc[bahamas_geo_var]
            if instrument=="radar":
                import processing_unified_grid as prcsuni
                Radar_uni_prcs=prcsuni.Radar_Processing(cfg_dict)
                # Add radar quality mask    
                if Performance.str2bool(cfg_dict["add_radar_mask_values"]):
                    ds=Radar_uni_prcs.add_mask_values(ds,bahamas_dict,
                                                      coords_dict)
            elif instrument=="radiometer":
                import processing_unified_grid as prcsuni
                Radiometer_uni_prcs=prcsuni.Radiometer_Processing(cfg_dict)
            if not instrument=="dropsondes":
                calib_cfg="calibrate_"+instrument
                if Performance.str2bool(cfg_dict[calib_cfg]):
                    cfg_dict["comment"]="Preliminary Calibrated data. "+\
                                        "Still use with caution."
                #device_dict["corr_comments"]
                
        global_attrs={}
        global_attrs["contact"]=cfg_dict["contact"]
        global_attrs["flight_date"]=cfg_dict["Flight_Dates_used"].values[0]
        global_attrs["flight_number"]=cfg_dict["Flight_Dates_used"].keys()[0]
        global_attrs["mission"]=cfg_dict["campaign"]
        global_attrs["comment"]=cfg_dict["comment"]
        if "corr_comments" in device_dict.keys():
            corrections=[corr[0] for corr in device_dict["corr_comments"].values if len(corr)> 0]
            if len(corrections)>0:
                global_attrs["Corrections"]='-'.join(corrections)
        
        ds.attrs=global_attrs   
        
        # Add missing value information
        ds=self.add_fill_missing(ds,cfg_dict)
        #######################################################################
        # Final Processing
        #######################################################################
        # Radar
        if instrument=="radar":
            ds=Radar_uni_prcs.process_radar_data(ds)
            if Performance.str2bool(cfg_dict["add_radar_mask_values"]):
               ds.attrs["performed_processing"]=ds.attrs["performed_processing"]+\
                   " Radar measurements are provided with masks."
        # Radiometer
        if instrument=="radiometer":
            ds.attrs["performed_processing"]=device_dict["performed_processing"]
            #print("Radiometer ds,",ds)
            if Performance.str2bool(cfg_dict["calibrate_radiometer"]):
                print("Calibrate Radiometer")
                ds=Radiometer_uni_prcs.calibrate_radiometer_TBs(ds)    
            if Performance.str2bool(cfg_dict["remove_radiometer_errors"]):
                ds=Radiometer_uni_prcs.remove_radiometer_errors(ds)
        if instrument=="dropsondes":
            ds.attrs["performed_processing"]="No further processing done."
        #######################################################################
        # Save files as netcdf        
        # Save data
        print("Save ",instrument," data as netCDF4")
        nc_compression=dict(zlib=True,complevel=1,dtype=np.float64)
        nc_encoding= {ds_var:nc_compression for ds_var in ds.variables}
        #outfile_ds["time"]=outfile_ds["time"].astype(np.int64)
        ds.to_netcdf(path=outfile,mode="w",format="NETCDF4",
                          engine="netcdf4",
                          encoding=nc_encoding)
        print("Sucessfully saved: ",outfile)
    
    @staticmethod
    def check_outfile_name_version_for_calibration(cfg_dict,instr):    
        temporary_version=cfg_dict["version"]
        if instr in ["radar","radiometer"]:
            device_calibration="calibrate_"+instr
            if Performance.str2bool(cfg_dict[device_calibration]):
                temporary_version=str(int(int(temporary_version)+1)) 
        return temporary_version
    
    def identify_newest_version(nc_path,for_calibrated_file=False):
        # the general file type of interest depends on calibrated or 
        # uncalibrated files in version number. After that highest
        # subversion_number has to be found.
        if for_calibrated_file: 
            nc_path=nc_path+"_v1*.nc"
        else:
            nc_path=nc_path+"_v0*.nc"
        # List relevant files of processing type (calibrated or uncalibrated)    
        nc_files=glob.glob(nc_path)
        # Identify highest subversion number and access its file
        subversion_list=[]
        for file in nc_files:
            #print(nc_files[0].split('.'))
            subversion_list.append(float(file.split('.')[-2]))
        newest_file_pos=np.argmax(np.array(subversion_list))
        return nc_files[newest_file_pos]
    
