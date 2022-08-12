# -*- coding: utf-8 -*-
"""
Created on Fri May  7 18:26:48 2021

@author: u300737
"""
import numpy as np
import pandas as pd
import xarray as xr

import os
import glob

import pickle

import Performance

from skimage.morphology import rectangle
from skimage.morphology import binary_closing, binary_opening 


def fill_missing_values(ds,cfg_dict):
    for var in ds.keys():
        if isinstance(cfg_dict["missing"],str):
            missing_value=int(cfg_dict["missing"])
        ds[var]=ds[var].fillna(missing_value)
    return ds



class Radiometer_Processing():
    def __init__(self,cfg_dict):
        self.cfg_dict=cfg_dict
        self.radiometer_ds=None
        
    
    # Major function calling several methods below, can also provide quicklooks
    def remove_turn_ascent_descent_data(self,radiometer_df,tb_dict):
        print("Remove turn and ascent/descent data")
        flight=self.cfg_dict["Flight_Dates_used"][0]
        filepath  = self.cfg_dict["device_data_path"]+"all_pickle/"
        pkl_fname = "uniData_bahamas_"+str(flight)+".pkl"
        with open(filepath+pkl_fname,"rb") as pkl_file:
            bahamas_dict=pickle.load(pkl_file)
            
            bahamas_cond=bahamas_dict["uniBahamas_roll_1d"].to_frame(name="Roll")\
                    .join(bahamas_dict["uniBahamas_alt_1d"].to_frame(name="Alt"))
            del bahamas_dict
        
        for var in radiometer_df.keys():
            # Ignore times with roll angle large than roll angle threshold
            roll_thres=float(self.cfg_dict['roll_threshold'])
            roll_index=bahamas_cond[bahamas_cond["Roll"]>roll_thres].index
            radiometer_df[var].loc[roll_index]=np.nan
            # Ignore times below altitude threshold
            alt_thres=float(self.cfg_dict['altitude_threshold'])
            alt_index=bahamas_cond[bahamas_cond["Alt"]<alt_thres].index
            radiometer_df[var].loc[alt_index]=np.nan
        
        # add processing in description
        maneouver_removal_comment=" Strong flight manoeuvers removed." 
        
        if tb_dict["performed_processing"].startswith("No further"):
            tb_dict["performed_processing"]=maneouver_removal_comment
                                        
        else:
            tb_dict["performed_processing"]=\
                tb_dict["performed_processing"]+\
                    maneouver_removal_comment
        return radiometer_df,tb_dict

    def remove_radiometer_errors(self,ds):
        version=self.cfg_dict["version"]+"."+self.cfg_dict["subversion"]
        # Load unified radiometer dataset, check for multiple files
        radiometer_path=self.cfg_dict["campaign_path"]+"Flight_Data/"+\
                    self.cfg_dict["campaign"]+"/all_nc/"
        radiometer_fname="radiometer_"+self.cfg_dict["flight_date_used"]+"_v"+\
                    self.cfg_dict["version"]+"."+self.cfg_dict["subversion"]+"*.nc"
        if len(glob.glob(radiometer_path+radiometer_fname))>1:
            raise FileExistsError("More than one radiometer data file found for",
                    "v",self.cfg_dict["version"]+"."+self.cfg_dict["subversion"],
                    "Please check and remove unneccesary files")
        
        # Radiometer data to be used
        self.radiometer_ds=ds.copy()
        # if self.cfg_dict["quicklooks"] checking figures will be created
        for day in self.cfg_dict["Flight_Dates_used"]:
            print("remove radiometer errors")
            self.remove_TB_errors()
            
            if Performance.str2bool(self.cfg_dict["quicklooks"]):
                #from Error_Identification import Radiometer_Errors
                #Radiometer_Errors=Radiometer_Errors(self.cfg_dict,
                #                                    create_figures=True)
                from Data_Plotter import Quicklook_Plotter,Radiometer_Quicklook
                Quicklooks_Plotter=Quicklook_Plotter(self.cfg_dict)
                Radiometer_Quicklooks=Radiometer_Quicklook(self.cfg_dict)                
                
                Radiometer_Quicklooks.radiometer_tb_dict=self.radiometer_ds
                Radiometer_Quicklooks.plot_radiometer_TBs(day,
                                                      raw_measurements=False)
                for hour in np.unique(self.radiometer_ds["TB"].time.dt.hour):
                    Radiometer_Quicklooks.plot_radiometer_TBs(day,
                                                      raw_measurements=False,
                                                      hourly=hour)
            
                #Radiometer_Quicklooks.plot_radiometer_TBs(day,
                #                                      raw_measurements=True)
            return self.radiometer_ds
                
    def remove_TB_errors(self):
        date_index = self.cfg_dict["Flight_Dates_used"]
        
        has_single_channel_error = False
        # Load the error flag file
        # this only works if run_identify_data_errors has been run
        file_end=".pkl"
        self.error_fpath=radiometer_path=self.cfg_dict["campaign_path"]+\
                            "Flight_Data/"+self.cfg_dict["campaign"]+"/all_pickle/"
        self.error_fname="error_flag_radiometer_"+str(date_index[0])+file_end
        
        # check if the error pkl file exists
        if not os.path.exists(self.error_fpath+self.error_fname):
            # Then create it
            import error_identification as Error_Identification
            
            radiometer_errors=Error_Identification.Radiometer_Errors(self.cfg_dict)
            radiometer_errors.identify_radiometer_errors()
            
        # The error flag file is a nc/csv/pkl file containing all flags
        # Load error index data (generated by looking through the data using
        # Error_identification.py)
        
        import pickle
        with open(self.error_fpath+self.error_fname,"rb") as error_file:
            error_dict=pickle.load(error_file)

        
        # Replace saw tooth values with nan
        # Look for sawtooth flag
        #load raw data
        relevant_error_dict=error_dict[date_index[0]]
        f=0
        # can also include single channel errors 
        for freq in self.radiometer_ds["TB"].uniRadiometer_freq:
            # Explanation for different radiometer modules
            # 1: 183,   f>180
            # 2: 11990, f>=90 & f<180
            # 3: KV,    f<90
            if freq>=180:
                module_name="183"
            elif 90<=freq<180:
                module_name="11990"
            elif freq<90:
                module_name="KV"
                
            sawteeth_series=relevant_error_dict[module_name]["sawteeth"]
            if sawteeth_series[sawteeth_series==1].shape[0]>0:
                print(freq.values,"GHz partly has sawtooth pattern")
                self.radiometer_ds["TB"][:,f]=self.radiometer_ds["TB"][:,f].where(\
                                                sawteeth_series.values==0)
                
            # Replace error values with nan
            # Look for error flag
            error_series=relevant_error_dict[module_name]["errors"]
            if error_series[error_series==1].shape[0]>0:
                print(freq.values," GHz partly shows radiometer module errors")
                self.radiometer_ds["TB"][:,f]=self.radiometer_ds["TB"][:,f].where(\
                                                error_series.values==0)
            #try:
            if "Single_Channels_"+module_name in relevant_error_dict.keys():
                single_channel_errors=\
                    relevant_error_dict["Single_Channels_"+module_name]
                relevant_channel_error=single_channel_errors[\
                                                module_name][\
                                                        str(freq.values)]
                
                if relevant_channel_error[relevant_channel_error==1].shape[0]>0:    
                    #has_single_channel_error = True
                    print(freq.values, "GHz shows specific issues for considered flight")
                    for error in single_channel_errors:
                        self.radiometer_ds["TB"][:,f]=self.radiometer_ds["TB"][:,f].where(\
                                                relevant_channel_error.values==0)
            f+=1
        
        return self.radiometer_ds
   
    def lookup_radiometer_timeoffsets(self):
        if self.cfg_dict["campaign"]=="EUREC4A":
            #######################################################################
            #WF Module
            #                           #RF01        #RF02       #RF03         #RF04        
            #                           200119       200122      200124        200126
            hamp_wf_idx_list=           [[],         [],        [[0,20500]],  [[0,23350]],                
                                        #RF05        RF06        #RF07          #RF08
            #                           200128       200130      200131         200202         200205
                                        [],          [],         [],        [[0,2000],      
                                                                           [2000,4000],
                                                                           [4000,6254],
                                                                           [6254,6270],
                                                                           [6270,-1]],                            
                                        #RF09        #RF10       #RF11         #RF12
            #                           200205       200207      200209        200211
                                        [],         [[0,22400]],   [],            [],     
            #                           #RF13        #RF14       #RF15                  
            #                           200213       200215      200218
                                        [],        [[2033,28756]], []]
            
            #                           #RF01        #RF02       #RF03         #RF04        
            #                           200119       200122      200124        200126
            hamp_wf_ofs_list=           [[-141],      [0],       [-2],         [1],                
                                        #RF05        RF06        #RF07          #RF08
            #                           200128       200130      200131         200202         200205
                                        [0],          [0],       [-1],        [-3,-1,-3,-2,-1],      
                                        #RF09        #RF10       #RF11         #RF12
            #                           200205       200207      200209        200211
                                        [0],         [2],         [0],           [0],     
            #                           #RF13        #RF14       #RF15                  
            #                           200213       200215      200218
                                        [0],          [-2],         [-1]]
            #######################################################################
            # KV Module
            #                           #RF01        #RF02         #RF03         #RF04        
            #                           200119       200122        200124        200126
            hamp_kv_idx_list=           [[],         [],           [],         [[0,23350]],                
                                        #RF05        RF06          #RF07          #RF08
            #                           200128       200130        200131         200202         200205
                                        [],          [],           [],         [[4000,6210]],      
                                        #RF09        #RF10         #RF11         #RF12
            #                           200205       200207        200209        200211
                                        [],       [[21300,23260]], [],          [[0,22900]],     
            #                           #RF13        #RF14         #RF15                  
            #                           200213       200215        200218
                                        [],        [[20483,-1]],   []]
            
            #                           #RF01        #RF02       #RF03         #RF04        
            #                           200119       200122      200124        200126
            hamp_kv_ofs_list=           [[0],         [0],       [-1],         [-2],                
                                        #RF05        RF06        #RF07          #RF08
            #                           200128       200130      200131         200202         200205
                                        [0],         [0],        [0],          [-2],      
                                        #RF09        #RF10       #RF11         #RF12
            #                           200205       200207      200209        200211
                                        [-3],        [-2],       [0],          [-2],     
            #                           #RF13        #RF14       #RF15                  
            #                           200213       200215      200218
                                        [0],         [-1],       []]
            #######################################################################
            # G Module
            #                           #RF01        #RF02       #RF03         #RF04        
            #                           200119       200122      200124        200126
            hamp_g_idx_list=           [[],          [],        [[0,21000]],  [[23700,-1]],                
                                        #RF05        RF06        #RF07          #RF08
            #                           200128       200130      200131         200202         200205
                                        [],          [],        [[21200,-1]],     [],                            
                                        #RF09        #RF10       #RF11         #RF12
            #                           200205       200207      200209        200211
                                        [],          [],          [],         [[21000,23500]],     
            #                           #RF13        #RF14       #RF15                  
            #                           200213       200215      200218
                                        [],           [],         []]
            #######################################################################
            #                           #RF01        #RF02       #RF03         #RF04        
            #                           200119       200122      200124        200126
            hamp_g_ofs_list=           [[0],          [0],       [-7],         [-3],                
                                        #RF05        RF06        #RF07          #RF08
            #                           200128       200130      200131         200202         200205
                                        [0],          [0],       [-3],         [-2],      
                                        #RF09        #RF10       #RF11         #RF12
            #                           200205       200207      200209        200211
                                        [0],         [0],        [-2],         [-5],     
            #                           #RF13        #RF14       #RF15                  
            #                           200213       200215      200218
                                        [0],          [0],        [-4]]
            #######################################################################
        else:
            print("This campaign has no offsets defined")
            self.time_offset_all=pd.DataFrame()
            return None
        
        time_offset_all=pd.DataFrame(data=np.nan,columns=["HAMP-WF_IDX","HAMP-KV_IDX",
                                                          "HAMP-G_IDX","HAMP-WF_OFS",
                                                          "HAMP-KV_OFS","HAMP-G_OFS"],
                                     index=[*self.cfg_dict["Flight_Dates"][\
                                            self.cfg_dict["campaign"]].values()])
        # Merge all into dataframe
        time_offset_all["HAMP-WF_IDX"]  = hamp_wf_idx_list
        time_offset_all["HAMP-WF_OFS"]  = hamp_wf_ofs_list
        time_offset_all["HAMP-KV_IDX"]  = hamp_kv_idx_list
        time_offset_all["HAMP-KV_OFS"]  = hamp_kv_ofs_list
        time_offset_all["HAMP-G_IDX"]   = hamp_g_idx_list
        time_offset_all["HAMP-G_OFS"]   = hamp_g_ofs_list
        
        self.time_offset_all=time_offset_all
        return self.time_offset_all
    
    def shift_time_offsets(self,tb_data,module_freq,shift_comment):
        #modules=["KV","WF","G"]
        #ofs_columns=["HAMP-"+mod+"_OFS" for mod in modules]
        #idx_columns=["HAMP-"+mod+"_IDX" for mod in modules]
        
        
        # Translate frequency to string from dataframe self.time_offset_all
        if module_freq[0]>=180:
            freq_str="HAMP-G"
        elif 90<=module_freq[0]<180:
            freq_str="HAMP-WF"
        elif module_freq[0]< 90:
            freq_str="HAMP-KV"
        else:
            raise Exception("Something went completely wrong, Frequency",
                            module_freq," does not exist")
        # Get relevant offset entries meaning correct module and date
        # module
        offset_arg=freq_str+"_OFS"
        freq_idx=freq_str+"_IDX"
        if (offset_arg in self.time_offset_all) and \
            (freq_idx in self.time_offset_all):
            
            relevant_ofs=self.time_offset_all[[freq_str+"_OFS",freq_str+"_IDX"]]
            # date
            dates=[str(date) for date in self.cfg_dict["Flight_Dates_used"]]
            relevant_ofs=relevant_ofs.loc[dates]
            tb_module_time=tb_data.index
            i=0
            for index_list_item in relevant_ofs[freq_str+"_IDX"]:
                if len(index_list_item)==0:
                    tb_module_time=tb_module_time.shift(
                                    periods=relevant_ofs[freq_str+"_OFS"].iloc[i][0],
                                    freq="s")
                    shift_comment.append("time offsets "+freq_str+": "+\
                                        "[:]"+\
                                        str(relevant_ofs[freq_str+"_OFS"].\
                                              iloc[i][0]))
            
                elif len(index_list_item)>=1:
                    for k in range(len(index_list_item)):
                        if not index_list_item[k][1]+1 == 0:
                            shifted_index=tb_module_time[index_list_item[k][0]:\
                            index_list_item[k][1]+1].shift(
                            periods=relevant_ofs[freq_str+"_OFS"].iloc[i][k],
                                    freq="s")
                        else:
                            shifted_index=tb_module_time[index_list_item[k][0]::].shift(
                            periods=relevant_ofs[freq_str+"_OFS"].iloc[i][k],
                                    freq="s")
                        tb_module_time_list=tb_module_time.values
                        if not index_list_item[k][1]+1 == 0:
                            tb_module_time_list[index_list_item[k][0]:\
                                   index_list_item[k][1]+1]=shifted_index.values
                        else:
                            tb_module_time_list[index_list_item[k][0]::]=shifted_index.values
                        tb_module_time=pd.DatetimeIndex(tb_module_time_list)
                        if relevant_ofs[freq_str+"_OFS"].iloc[i][k]!=0:
                            shift_comment.append("time offsets "+freq_str+": "+\
                                        str(index_list_item[k])+\
                                        str(relevant_ofs[freq_str+"_OFS"].\
                                              iloc[i][k]))
                else:
                    raise Exception("Something went completely wrong")
            
            if relevant_ofs[freq_str+"_OFS"].iloc[i][0]!=0:
                print("TB measurements of ",freq_str,
                      " module are shifted in time")
                
            i+=1
            
            tb_data.index=tb_module_time
        
        
        return tb_data,shift_comment

    def calibrate_radiometer_TBs(self,ds):
        from Measurement_Instruments import HALO_Devices, HAMP
        HALO_Devices_cls=HALO_Devices(self.cfg_dict)
        HAMP_cls=HAMP(HALO_Devices_cls)
        HAMP_cls.access_HAMP_TB_calibration_coeffs()
        calib_coeff_ds=HAMP_cls.tb_calib_coeff_ds
        calib_coeff_da=calib_coeff_ds.sel({"date":\
                        str([*self.cfg_dict["Flight_Dates_used"].values][0])})
        calib_coeff_da=calib_coeff_da.loc[{"frequency":ds.uniRadiometer_freq}]
        ds=ds.sortby("uniRadiometer_freq")
        ds_new=ds.assign(TB=ds["TB"]*calib_coeff_da["slope"].values+\
                         calib_coeff_da["offset"].values)
        
        print("HAMP TB calibration done!")
        return ds_new
                  

class Radar_Processing():
        
    def __init__(self,cfg_dict):
        self.cfg_dict=cfg_dict
    
    def identify_vertical_gaps(self,da):
       """
       da : xr.DataArry or pd.DataFrame of radar values
        

       Returns
       -------
       None.

       """
       # Count number of column values having nan values
       if not isinstance(da,pd.DataFrame):
           da_temporary=pd.DataFrame(data=da.data,index=pd.DatetimeIndex(
                                     np.array(da.time[:])),
                                     columns=np.array(da.height[:]))
           #da_temporary_second=da_temporary.copy()
       else:
           da_temporary=da.copy()
       
       da_temporary=da_temporary.replace(
                       to_replace=float(self.cfg_dict["missing_value"]),
                       value=np.nan)
       
       da_temporary_second=da_temporary.copy()
       # Identify vertical column
       isnan_column_sum=da_temporary.isna().sum(axis=1)
       vertical_isnan_columns=isnan_column_sum[\
                                isnan_column_sum==da_temporary.shape[1]]
       
       # Check if values are existent before or not using diff methods
       da_temporary=da_temporary.replace(
                       to_replace=np.nan,value=0)
       da_temporary_diff=da_temporary.diff(periods=-1,axis=0)
       da_temporary_diff=da_temporary_diff.replace(to_replace=np.nan,
                                                   value=0)
       da_temporary_diff=da_temporary_diff.replace(to_replace=np.inf,
                                                   value=0)
       da_temporary_diff=da_temporary_diff.replace(to_replace=-np.inf,
                                                   value=0)
       
       #da_temporary_diff.index=da_temporary_diff.index.shift(periods=-1)
       diff_nan_index=da_temporary_diff.mean(axis=1)
       #diff_nan_index
       relevant_diff_nan_index=diff_nan_index[diff_nan_index!=0].index
       
       #empty_columns_da=vertical_isnan_columns[]
       gap_index=vertical_isnan_columns.index.intersection(
                       relevant_diff_nan_index)
       return gap_index
    
    def process_radar_data(self,ds):
        # Remove Radar clutter if desired
        ds.attrs["performed_processing"]="No further data processing done."
        if Performance.str2bool(self.cfg_dict["fill_gaps"]):
            ds=self.fill_gaps(ds)
        if Performance.str2bool(self.cfg_dict["remove_clutter"]):
            ds=self.remove_clutter(ds)
        
        if Performance.str2bool(self.cfg_dict["remove_side_lobes"]):
            ds=self.remove_side_lobes(ds)
        if Performance.str2bool(self.cfg_dict["calibrate_radar"]):
            ds=self.calibrate_radar(ds)
                
        if Performance.str2bool(self.cfg_dict["quicklooks"]):
            from Data_Plotter \
                import Quicklook_Plotter,Radar_Quicklook
            Quicklooks_Plotter=Quicklook_Plotter(self.cfg_dict)
            Radar_Quicklooks=Radar_Quicklook(self.cfg_dict)                
            self.processed_radar=ds
            ds=self.apply_radar_flags()
            Radar_Quicklooks.plot_radar_quicklook(ds)
            
        return ds
    
    def fill_gaps(self,ds):
        for var in ds.keys():
            if len(ds[var].shape)==2:
                print(var," gap filling")
                ds[var]=self.da_gap_filling(ds[var])
            
        if ds.attrs["performed_processing"].startswith("No further"):
           ds.attrs["performed_processing"]=" Entire data gap filled."
        else:
            ds.attrs["performed_processing"]=ds.attrs["performed_processing"]+\
                " Entire data gap filled."            
        return ds
    
    def da_gap_filling(self,da):
        da_df=pd.DataFrame(data=da.data,index=pd.DatetimeIndex(
                                     np.array(da.time[:])),
                                     columns=np.array(da.height[:]))
        
        gap_indexes=self.identify_vertical_gaps(da_df)
           
        da_temporary=da_df.copy()
        da_temporary=da_temporary.replace(
                       to_replace=float(self.cfg_dict["missing_value"]),
                       value=np.nan)
        da_temporary=da_temporary.interpolate(method="time",limit=2)
        da_df.loc[gap_indexes]=da_temporary.loc[gap_indexes]
        da.data=np.array(da_df)
        return da
    
    def add_mask_values(self,ds,bahamas_dict,coords_dict):
        """
        

        Parameters
        ----------
        ds : xr.Dataset
            unified radar dataset.
        bahamas_dict : dictionary
            containing all bahamas variables from unified grid
        coord_dict : dict
            coordinates for output file
        
        Returns
        -------
        ds : xr.Dataset
            unified radar dataset with mask values

        """
        mask_path=self.cfg_dict["device_data_path"]+"auxiliary/"
        mask_fname="Radar_Info_Mask_"+\
                            str(self.cfg_dict["Flight_Dates_used"][0])+".csv"
        mask_file=mask_path+mask_fname
        
        #Load CSV-File mask
        radar_mask=pd.read_csv(mask_file)
        radar_mask.index=pd.DatetimeIndex(radar_mask["Unnamed: 0"])
        del radar_mask["Unnamed: 0"]
        mask_values=[]
        for i in range(4):
            mask_values.append(str(radar_mask.iloc[i,-2])+\
                " : "+str(radar_mask.iloc[i,-1]))
        
        # loc radar mask to unified grid
        radar_mask_uni=pd.DataFrame(
                            data=np.nan, 
                            index=pd.DatetimeIndex(\
                                bahamas_dict["uni_time"]),
                            columns=ds["height"])
        radar_mask=radar_mask.iloc[:,0:len(ds["height"])]
        radar_mask_uni.loc[radar_mask.index]=radar_mask
        radar_mask_info={"mask_values":mask_values}
        ds["radar_flag"]=xr.DataArray(data=radar_mask_uni,
                                      dims=coords_dict.keys())
        ds["radar_flag"].attrs=radar_mask_info
        print("Radar mask values added to unified grid")
        return ds
    
    def remove_side_lobes(self,ds):
        """
        

        Parameters
        ----------
        ds : xr.Dataset
            unified radar dataset.

        Returns
        -------
        ds : xr.Dataset
            unified radar dataset with removed side lobes

        """
        
        print("remove side lobes")
        
        # Load unified bahamas dataset
        bahamas_path=self.cfg_dict["campaign_path"]+"Flight_Data/"+\
                    self.cfg_dict["campaign"]+"/all_nc/"
        bahamas_fname="bahamas_"+self.cfg_dict["flight_date_used"]+"_v"+\
            self.cfg_dict["version"]+"."+self.cfg_dict["subversion"]+".nc"
        
        bahamas_ds=xr.open_dataset(bahamas_path+bahamas_fname)
        height=ds["height"]
        roll=bahamas_ds["roll"]
        pitch=bahamas_ds["pitch"]
        alt=bahamas_ds["alt"]
        
        # Calculate height affected by side lobes during turns
        h_side_lobes=alt * (1/np.cos(np.deg2rad(roll))/\
                            np.cos(np.deg2rad(pitch))-1)+60
        h_side_lobes_2d=np.matlib.repmat(h_side_lobes,height.shape[0],1).T
        h_2d=np.matlib.repmat(height,h_side_lobes.shape[0],1)
        geo_mask=h_2d<h_side_lobes_2d
        # Calculate the turns above roll threshold
        roll_ind=abs(roll)>int(self.cfg_dict["roll_threshold"])
        roll_ind_mat=np.matlib.repmat(roll_ind,height.shape[0],1).T
        
        # Combine both to side lobe mask
        side_lobe_mask= geo_mask & roll_ind_mat
        for var in ds.keys():
            if len(ds[var].shape)>1 and not var=="radar_mask":
                print(var)
                ds[var]=ds[var].where(~side_lobe_mask).fillna(\
                                            float(self.cfg_dict["fill_value"]))
        
        ds.attrs["performed_processing"]=ds.attrs["performed_processing"]+\
            " Side lobes removed."
        # Add side lobe information to radar mask
        if Performance.str2bool(self.cfg_dict["add_radar_mask_values"]):
            ds["radar_flag"]=ds["radar_flag"].where(~side_lobe_mask).fillna(5)
            # Add mask value to attributes
            mask_value_list=ds["radar_flag"].mask_values
            mask_value_list.append('5.0 : side lobes removed')
            ds["radar_flag"].attrs["mask_values"]=mask_value_list
        return ds
    
    def morphological_clutter_removal(self,da):
        """
        This functions removes clutter while using binary morphological closing
        and opening from the skimage.morphology library

        Parameters
        ----------
        da : xr.DataArray
            radar data variable.

        Returns
        -------
        None.

        """
        print(da.name)
        da_copy=da.copy()
        period=3
        height_levels=2
        bitmask=(da[:,:]>-120).astype(int)
        #bitmask=bitmask.astype(int)
        
        selem=rectangle(period,height_levels)  #For now closing is only performed in time.
                                               #(height_levels=1)
        bitmask=binary_closing(bitmask,selem)
        bitmask=binary_opening(bitmask,selem)
        varNew=da_copy.where(bitmask).fillna(
                        int(self.cfg_dict["missing_value"]))
        da=varNew
        return da
    
    def calibrate_reflectivity(self,ds):
        prcs_cfg_dict=self.cfg_dict
        import Measurement_Instruments
        HALO_Devices_cls=Measurement_Instruments.HALO_Devices(prcs_cfg_dict)
        Radar_cls=Measurement_Instruments.HALO_Radar(HALO_Devices_cls)
        Radar_cls.show_calibration()
        zg_attrs=ds["Zg"].attrs
        ds["Zg"]=ds["Zg"].where(ds['Zg'] != \
                                float(self.cfg_dict["missing_value"]))
        ds["Zg"]=ds["Zg"]*10**(0.1*Radar_cls.dB_offset)
        ds["Zg"].attrs=zg_attrs
        Radar_cls.processed_radar_ds=ds.copy()
        Radar_cls.calc_dbz_from_z(raw_measurement=False)
        return Radar_cls.processed_radar_ds
    
    def calibrate_radar(self,ds):
        # currently the radar calibration only considers the reflectivity
        # values by an offset according to Ewald (2019).
        ds=self.calibrate_reflectivity(ds)
        return ds
    def remove_clutter(self,ds,method="morphological_closing"):
        """
        Remove the radar clutter by calling a morphological closing method
        Parameters
        ----------
        ds : xr.Dataset
            radar unified grid dataset of all variables.

        Returns
        -------
        ds.

        """
        # Get variable dimension sizes when looping over them
        # Choose only variables having two dimensions
        if method=="morphological_closing":
            print("Morphological Closing of Cloud Mask will be performed",
              " to remove clutter")
            clutter_removal=self.morphological_clutter_removal
        for var in ds.keys():
            if len(ds[var].shape)>=2:
                clutter_clear_array=clutter_removal(ds[var])
                ds[var][:,:]=clutter_clear_array[:,:]
        if ds.attrs["performed_processing"].startswith("No further"):
           ds.attrs["performed_processing"]=" Clutter removed."
        else:
            ds.attrs["performed_processing"]=ds.attrs["performed_processing"]+\
                " Clutter removed."            
        return ds
    
    def apply_radar_flags(self):
        print("Flag the radar data")
        radar_flag=pd.DataFrame(data=np.array(
                                self.processed_radar["radar_flag"][:]),
                                columns=map(str,np.array(
                                    self.processed_radar["height"][:])))
        
        radar_flag.index=pd.DatetimeIndex(np.array(\
                                            self.processed_radar.time[:]))
        radar_flag=radar_flag.replace(to_replace=[1,2,3,4],
                                      value=np.nan)
        if not hasattr(self, "quicklook_vars"):
            self.quicklook_vars=["dBZg","LDRg"]
        for var in self.quicklook_vars:
            self.processed_radar[var]=self.processed_radar[var]+radar_flag
        #reflectivity_factor=reflectivity_factor+radar_flag
        #ldr_factor=ldr_factor+radar_flag
        #else:
        #    radar_flag=pd.DataFrame(data=np.array(
        #                            dataset["data_flag"][:]),
        #                            columns=map(str,
        #                                    np.array(dataset["height"][:])))
        #    radar_flag.index=reflectivity_factor.index
        #    radar_flag=radar_flag.replace(to_replace=[1,2,3,4],
        #                                              value=np.nan)
        #    reflectivity_factor=reflectivity_factor+radar_flag
        #    ldr_factor=ldr_factor+radar_flag
        #if self.processed_radar.attrs["performed_processing"].startswith("No further"):
        #   self.processed_radar.attrs["performed_processing"]=" Clutter removed."
        #else:
        #   self.processed_radar.attrs["performed_processing"]=\
        #       self.processed_radar.attrs["performed_processing"]+\
        #           " Radar data flagged."            
        
        print("Radar data flagged")
        return self.processed_radar        
        

                    