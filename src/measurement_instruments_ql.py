# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 12:04:10 2022

@author: u300737
"""
import os
import glob

import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr

def updt(total, progress):
    """
    Displays or updates a console progress bar.
        
    Original source: https://stackoverflow.com/a/15860757/1391441
    """
    import sys
        
    barLength, status = 20, ""
    progress = float(progress) / float(total)
    if progress >= 1.:
            progress, status = 1, "\r\n"
    block = int(round(barLength * progress))
    text = "\r[{}] {:.0f}% {}".format("#" * block + "-" * (barLength - block), 
                                          round(progress * 100, 0), status)
    sys.stdout.write(text)
    sys.stdout.flush()


class HALO_Devices():
    def __init__(self,cfg_dict):
        self.cfg_dict=cfg_dict
        self.campaign_name=self.cfg_dict["campaign"]
        self.major_data_path=self.cfg_dict["device_data_path"]
        self.halo_devices=["bahamas","bacardi","dropsonde","radiometer","lidar",
                           "radar","smart","specMACS"]
        self.avail_halo_devices={}
        avail_list=[False]*len(self.halo_devices)
        self.avail_devices=dict(zip(self.halo_devices,avail_list))
    def update_major_data_path(self,major_path_name):
        self.major_data_path=major_path_name
    def check_avail_devices(self):
        """
        Checks whether files of measurement data from given devices are present
        for flight of interest

        Returns
        -------
        None.

        """
        for device in self.halo_devices:
            #check first if device file ("directory") is present
            device_path=self.major_data_path+device+"/"
            if not os.path.exists(device_path):
                print(device," is either not mounted on the aircraft or",
                      " not included in the quicklook routine.")
            else:
                # if directory is present look for files of given date (flight)
                # several if clauses are necessary for the different file names
                if device in ["bacardi","bahamas","dropsonde","radar","smart"]:
                    avail_files=glob.glob(device_path+"*"+\
                                          self.cfg_dict["date"]+"*.nc")
                    if len(avail_files)>0:
                        self.avail_devices[device]=True
                        print("Device ",device,"data is available.")
                elif device in ["radiometer"]:
                    # radiometer needs special treatment due to different 
                    # naming and installed radiometer modules operating
                    # separately.
                    avail_KV_files=glob.glob(device_path+"KV/"+\
                                          self.cfg_dict["date"][2:]+"*.nc")
                    avail_11990_files=glob.glob(device_path+"11990/"+\
                                          self.cfg_dict["date"][2:]+"*.nc")
                    avail_183_files=glob.glob(device_path+"183/"+\
                                          self.cfg_dict["date"][2:]+"*.nc")
                    if len(avail_KV_files)!=0:
                        self.avail_devices["radiometer"]=True
                        #self.avail_devices["HAMP"+"_KV"]=True
                    elif len(avail_11990_files)!=0:
                        self.avail_devices["radiometer"]=True
                        #self.avail_devices["HAMP"+"_11990"]=True
                    elif len(avail_183_files)!=0:
                        self.avail_devices["radiometer"]=True
                        #self.avail_devices["HAMP"+"_183"]=True
                    print("Device",device,"data is available.")
                else:
                    print("Device ",device,
                          "is definetly not mounted on HALO.")
                    continue
    @staticmethod   
    def numpydatetime64_to_datetime(npdt_array):
        """
        Converts numpy datetime64 array to a datetime object array.
        Parameters:
        --------
        npdt_array : numpy array of type np.datetime64 or np.datetime64 type
    		Array (1D) or directly a np.datetime64 type variable.
        """
        sec_epochtime = npdt_array.astype(np.timedelta64) /\
                                            np.timedelta64(1, 's')

    	# sec_epochtime can be an array or just a float
        if sec_epochtime.ndim > 0:
            time_dt = np.asarray([dt.datetime.utcfromtimestamp(tt)\
                                  for tt in sec_epochtime])
        else:
            time_dt = dt.datetime.utcfromtimestamp(sec_epochtime)
        return time_dt

    def numpydatetime64_to_reftime(npdt_array, reftime):
        """
        Created by A. Walbroel (IGMK, Cologne)
        Converts numpy datetime64 array to array in seconds 
        since a reftime as type: float. 
        Reftime could be for example: "2017-01-01 00:00:00" (in UTC)

    	Parameters:
        --------
        npdt_array : numpy array of type np.datetime64 or np.datetime64 type
    		Array (1D) or directly a np.datetime64 type variable.
        reftime : str
    		Specification of the reference time in "yyyy-mm-dd HH:MM:SS" (in UTC).
        """
        time_dt = HALO_Devices.numpydatetime64_to_datetime(npdt_array)
        reftime = dt.datetime.strptime(reftime, "%Y-%m-%d %H:%M:%S")
        try:
            sec_epochtime = np.asarray([(dtt - reftime).total_seconds() \
                                    for dtt in time_dt])
        except TypeError:	# then, time_dt is no array
            sec_epochtime = (time_dt - reftime).total_seconds()
        return sec_epochtime
    
    def vectorized_harvesine_distance(s_lat, s_lng, e_lat, e_lng):

        # approximate radius of earth in km
        R = 6373.0

        s_lat = s_lat*np.pi/180.0                      
        s_lng = np.deg2rad(s_lng)     
        e_lat = np.deg2rad(e_lat)                       
        e_lng = np.deg2rad(e_lng)  

        d = np.sin((e_lat - s_lat)/2)**2 +\
            np.cos(s_lat)*np.cos(e_lat) *\
                np.sin((e_lng - s_lng)/2)**2

        return 2 * R * np.arcsin(np.sqrt(d))    
#-----------------------------------------------------------------------------#
class BACARDI(HALO_Devices):
    def __init__(self,HALO_Devices_cls):
        self.cfg_dict=HALO_Devices_cls.cfg_dict
        self.campaign_name=HALO_Devices_cls.campaign_name
        self.major_data_path=HALO_Devices_cls.major_data_path
        self.name="BACARDI"
        self.raw_data_path=self.major_data_path+"/"+self.name.lower()+"/"
        if not os.path.exists(self.raw_data_path):
            os.makedirs(self.raw_data_path)
    
    def open_unprocessed_quicklook_data(self):
        self.bacardi_file=self.name.upper()+\
                            "-QL_broadband-radiation_HALO_"+\
                                self.cfg_dict["date"]+"_"+\
                                    self.cfg_dict["flight"]+".nc"
        B = xr.open_dataset(self.raw_data_path+\
                            self.bacardi_file, decode_times = False)
        # Just temporary open the file and read the relevant variables 
        # required for plotting quicklooks
        
        bacardi_dict={}
        bacardi_dict["time"] = B['TIME'] #B["time"]
        bacardi_dict["alt"]  = B['IRS_ALT']  #B["alt"]
        bacardi_dict["lat"]  = B['IRS_LAT']  #B["lat"]
        bacardi_dict["lon"]  = B['IRS_LON']  #B["lon"]
        
        bacardi_dict["F_down_solar"] = B["FDSC"].data
        bacardi_dict["F_up_solar"]   = B["FUS"].data
        bacardi_dict["F_down_terr"]  = B["FDL"].data
        bacardi_dict["F_up_terr"]    = B["FUL"].data
        bacardi_dict["sza"]          = B["SUN_ALT"].data

        # close file when finished
        B.close()
        self.bacardi_dict=bacardi_dict
    
    def open_processed_LIM_quicklook_data(self):
        
        self.bacardi_file="HALO-AC3_HALO_"+self.name.upper()+\
                          "_BroadbandFluxes_"+self.cfg_dict["date"]+\
                              "_"+self.cfg_dict["flight"]+".nc"
        B = xr.open_dataset(self.raw_data_path+\
                            self.bacardi_file,decode_times = False)
        bacardi_dict={}
        bacardi_dict["time"]    = B["time"]
        bacardi_dict["alt"]     = B["alt"]
        bacardi_dict["lat"]     = B["lat"]
        bacardi_dict["lon"]     = B["lon"]
        
        bacardi_dict["F_down_solar"]    = B['F_down_solar'].data
        bacardi_dict["F_up_solar"]      = B['F_up_solar'].data
        bacardi_dict["F_down_terr"]     = B['F_down_terrestrial'].data
        bacardi_dict["F_up_terr"]       = B['F_up_terrestrial'].data
        bacardi_dict["F_down_solar_sim"]= B["F_down_solar_sim"].data
        bacardi_dict["SZA"]             = B['sza'].data

        B.close()
        self.bacardi_dict=bacardi_dict
    
    def open_raw_quicklook_data(self,LIM_processed=True):
        if not LIM_processed:
            self.open_unprocessed_quicklook_data()
        else:
            try:
                self.open_processed_LIM_quicklook_data()
            except:
                self.open_unprocessed_quicklook_data()
#-----------------------------------------------------------------------------#
class BAHAMAS(HALO_Devices):
    def __init__(self,HALO_Devices_cls):
        # changed after NARVAL-I! use only for NARVAL-II and onward. For
        # NARVAL-I data, use bahamasNetCDFVarTable_n1
        # Nametable to join the different names for Bahamas variables
        #
        # Update 19/01 2017: added list varNamesN1 from file bahamasNetCDFVarTable_n1
        #   to incorporate NARVAL-I variables; here, deleted last entry 'irs_alt'
        #   to fit the other lengths (let's see what problems this will cause...)
        self.cfg_dict=HALO_Devices_cls.cfg_dict
        self.device="bahamas"
        self.device_data_path=self.cfg_dict["device_data_path"]+self.device+"/"
        self.irs_names = ['TIME','ABSHUM','ALPHA','BETA','H','HP','MC',
                          'MIXRATIO','PS','QC','RELHUM','TAS','TAT','TD',
                          'THETA','THETA_V','TV','U','V','W','WA','WS','TS',
                          'TIME','GPS_LAT','GPS_LON','GPS_ALT','GPS_NSV',
                          'GPS_EWV','IRS_VV','IRS_PHI','IRS_THE','IRS_HDG','',
                          '','','IRS_P','IRS_Q','IRS_R','IRS_AXB','IRS_AYB',
                          'IRS_AZB','IRS_AZG','IRS_ATA','GPS_GS','','','',]

        self.igi_names = ['TIME','ABSHUM','ALPHA','BETA','H','HP','MC',
                          'MIXRATIO','PS','QC','RELHUM','TAS','TAT','TD',
                          'THETA','THETA_V','TV','U','V','W','WA','WS','TS',
                          'TIME','IGI_LAT','IGI_LON','IGI_ALT','IGI_NSV',
                          'IGI_EWV','IGI_VV','IGI_ROLL','IGI_PITCH','IGI_YAW',
                          'IGI_RMSX','IGI_RMSY','IGI_RMSZ','IGI_P','IGI_Q',
                          'IGI_R','IGI_AXB','IGI_AYB','IGI_AZB','IGI_AZG',
                          'IGI_ATA','IGI_GS','','','']
        
        self.irs_names_v2=['TIME','ABSHUM','ALPHA','BETA','H','HP','MC',
                            'MIXRATIO','PS','QC','RELHUM','TAS','TAT','TD',
                            'THETA','THETA_V','TV','U','V','W','WA','WS','TS',
                            'TIME','IRS_LAT','IRS_LON','IRS_ALT','IRS_NSV',
                            'IRS_EWV','IRS_VV','IRS_PHI','IRS_THE','IRS_HDG','',
                            '','','IRS_P','IRS_Q','IRS_R','IRS_AXB','IRS_AYB',
                            'IRS_AZB','IRS_AZG','IRS_ATA','IRS_GS','IRS_WS',
                            'IRS_WA','IRS_DA']
        
        self.var_names = ['time','abshum','alpha','beta','h','palt','mc',
                          'mixratio','P','qc','RH','speed_air','TAT','Td',
                          'theta','theta_v','Tv','U','V','W','wa','ws','T',
                          't_sys','lat','lon','alt','nsv','ewv','vv','roll',
                          'pitch','heading','rmsx','rmsy','rmsz','p','q','r',
                          'axb','ayb','azb','azg','ata','speed_gnd','irs_ws',
                          'irs_wa','irs_da']
        
        self.var_names_v1=['utc_time','abshum','alpha','beta','h','palt','mc',
                            'mixratio','P','qc','RH','speed_air','T','Td',
                            'theta','theta_v','Tv','U','V','W','wa','ws','Ts',
                            't_sys','lat','lon','galt','nsv','ewv','vv','roll',
                            'pitch','heading','rmsx','rmsy','rmsz','p','q','r',
                            'axb','ayb','azb','azg','ata','speed_gnd','irs_ws',
                            'irs_wa','irs_da']
        
        self.nc_var_names={'uni_time':'time',
                            'uni_height':'height',
                            'mr_mat':'uniBahamas_mixratio',
                            'mr':'uniBahamas_mixratio_1d',
                            'mr_intFlag':'uniBahamas_mixratio_interpolate_flag',
                            'p_mat':'uniBahamas_P',
                            'p':'uniBahamas_P_1d',
                            'p_intFlag':'uniBahamas_P_interpolate_flag',
                            'rh_mat':'uniBahamas_RH',
                            'rh':'uniBahamas_RH_1d',
                            'rh_intFlag':'uniBahamas_RH_interpolate_flag',
                            'theta_mat':'uniBahamas_theta',
                            'theta':'uniBahamas_theta_1d',
                            'theta_intFlag':'uniBahamas_theta_interpolate_flag',
                            'ta_mat':'uniBahamas_T',
                            'ta':'uniBahamas_T_1d',
                            'ta_intFlag':'uniBahamas_T_interpolate_flag',
                            'u_mat':'uniBahamas_U',
                            'u':'uniBahamas_U_1d',
                            'u_intFlag':'uniBahamas_U_interpolate_flag',
                            'v_mat':'uniBahamas_V',
                            'v':'uniBahamas_V_1d',
                            'v_intFlag':'uniBahamas_V_interpolate_flag',
                            'w_mat':'uniBahamas_W',
                            'w':'uniBahamas_W_1d',
                            'w_intFlag':'uniBahamas_W_interpolate_flag',
                            'altitude_mat':'uniBahamas_alt',
                            'altitude':'uniBahamas_alt_1d',
                            'altitude_intFlag':'uniBahamas_alt_interpolate_flag',
                            'heading_mat':'uniBahamas_heading',
                            'heading':'uniBahamas_heading_1d',
                            'heading_intFlag':'uniBahamas_heading_interpolate_flag',
                            'lat_mat':'uniBahamas_lat',
                            'lat':'uniBahamas_lat_1d',
                            'lat_intFlag':'uniBahamas_lat_interpolate_flag',
                            'lon_mat':'uniBahamas_lon',
                            'lon':'uniBahamas_lon_1d',
                            'lon_intFlag':'uniBahamas_lon_interpolate_flag',
                            'pitch_mat':'uniBahamas_pitch',
                            'pitch':'uniBahamas_pitch_1d',
                            'pitch_intFlag':'uniBahamas_pitch_interpolate_flag',
                            'roll_mat':'uniBahamas_roll',
                            'roll':'uniBahamas_roll_1d',
                            'roll_intFlag':'uniBahamas_roll_interpolate_flag',
                            'gs_mat':'uniBahamas_speed_gnd',
                            'gs':'uniBahamas_speed_gnd_1d',
                            'gs_intFlag':'uniBahamas_speed_gnd_interpolate_flag',                           
                            'vel_mat':'uniBahamas_vv',
                            'vel':'uniBahamas_vv_1d',
                            'vv_intFlag':'uniBahamas_vv_interpolate_flag'}
        
        self.var_name_table = pd.DataFrame()
        self.var_name_table["var_names"]    = self.var_names
        self.var_name_table["irs_names"]    = self.irs_names
        self.var_name_table["igi_names"]    = self.igi_names
        self.var_name_table["irs_names_v2"] = self.irs_names_v2
        self.var_name_table["var_names_v1"] = self.var_names_v1
    
    def open_bahamas_data(self,raw_or_processed="raw"):
        if raw_or_processed =="raw":
            fname           = "*"+str(self.cfg_dict["flight_date_used"])+"*.nc"
            bahamas_path    = self.device_data_path
            bahamas_file=glob.glob(bahamas_path+fname,recursive=True)[0]
            
        elif raw_or_processed =="processed":
            from campaign_netcdf import CPGN_netCDF
            bahamas_path=self.cfg_dict["device_data_path"]+"/all_nc/"
            print("BAHAMAS Path", bahamas_path)
            bahamas_file=CPGN_netCDF.identify_newest_version(bahamas_path,
                                    device="bahamas",for_calibrated_file=True,
                                    date=self.cfg_dict["date"])
            
            #fname= "*"+str(self.cfg_dict["flight_date_used"])
        #print("Open bahamas data in ", self.device_data_path)
        #import os
        #os.chdir(self.device_data_path)
        #print(fname)
        #bahamas dataset
        self.bahamas_ds =   xr.open_dataset(bahamas_file)
    
    def replace_bahamas_varname(self,varName_find,bahamas_vars):
        """
        This function definetly need further adaptations
        
        varName_find : str
            name of the variable to identify that need to be replaced
        bahamas_vars : FrozenDict()
        """

        # function for replacing the bahamas variable name with a corresponding
        # unified one. This is necessary, since bahamas variable names changed
        # often...
        print("Variable name to find: ",varName_find)
        # Load bahamas variable names
        nametable = self.var_name_table
        # Find index of desired variable in list
        index_col=pd.Series()
        
        for col in nametable.columns:
            col_nametable=nametable.loc[:,col]
            pos =  col_nametable[col_nametable==varName_find].index.values
            if not pos.shape[0]==0:
                index_col=index_col.append(pd.Series(pos))
        
        #% Get unique indices
        unique=list(pd.unique(index_col))
        #% Check if found variables form name table are in its first column
        variables= nametable.iloc[unique,:]
        c=variables.isin(dict(bahamas_vars).keys()).astype(int)
        
        #If the variable name is found in more than one column, 
        #select the first one listed
        if c.shape[0]>1:
                c       = c.iloc[0,:]  # this was a series
                unique  = unique[0],
                print('Att: selected first found variable;',
                      'check if this is the right one!')
        
        #% If more than one variable is found, 
        coll={}
        if np.sum(c.values)>1:
            # Loop all bahamas variables
            for var in bahamas_vars:
                # Get column of variables, i.e. the column corresponding to this
                # bahamas name format
                _,coll[var]=np.where(nametable==var)
        
            # Concatenate all columns
            coll=np.hstack(coll.values())
            #% Count how often each column is used
            col_count,_ = np.histogram(coll,np.unique(coll))        
            #% Use the one with the most matching entries
            col_use = np.argmax(col_count)
        
            #% Copy column value to c
            c = col_use;
    
            #% Copy bahamas variable name to output variable
        else:
            c=np.argmax(c)
        varNameUse = nametable.iloc[unique[0],c]
        
        return varNameUse
    
    def lookup_varnames(self,variable_names,version="var_names"):
        """
        Look for variables in different bahamas data versions
        """
        true_names=[]
        if version=="var_names":
            series_names=pd.Series(data=self.var_names,index=self.irs_names_v2)
        for var in variable_names:
              idx=series_names.index.get_loc(var)
              true_names.append(series_names.iloc[idx])
        return true_names
    
    @staticmethod
    def pressure2flightlevel(pressure):
        from metpy.units import units
        from metpy.constants import Rd, g
        """
        Conversion of pressure to height (hft) with hydrostatic equation,
        according to the profile of the 1976 U.S. Standard atmosphere [NOAA1976]_.
        
        Reference:
            H. Kraus, Die Atmosphaere der Erde, Springer, 2001, 470pp.,
            Sections II.1.4. and II.6.1.2.
        
        Parameters
        ----------
        pressure : `pint.Quantity` or `xarray.DataArray`
            Atmospheric pressure
        
        Returns
        -------
        `pint.Quantity` or `xarray.DataArray`
            Corresponding height value(s) (hft)
        
        Notes
        -----
        .. math:: Z = \begin{cases}
                  Z_0 + \frac{T_0 - T_0 \cdot \exp\left(\frac{\Gamma \cdot \
                                R}{g\cdot\log(\frac{p}{p0})}\right)}{\Gamma}
                  &\Gamma \neq 0\\
                  Z_0 - \frac{R \cdot T_0}{g \cdot \log(\frac{p}{p_0})} &\text{else}
                  \end{cases}
        """
        # define standard atmosphere
        _STANDARD_ATMOSPHERE = [
            (0 * units.km, 288.15 * units.K, 101325 * units.Pa, 
                 0.0065 * units.K / units.m),
            (11 * units.km, 216.65 * units.K, 22632.1 * units.Pa, 
                 0 * units.K / units.m),
            (20 * units.km, 216.65 * units.K, 5474.89 * units.Pa, 
                 -0.001 * units.K / units.m),
            (32 * units.km, 228.65 * units.K, 868.019 * units.Pa, 
                 -0.0028 * units.K / units.m),
            (47 * units.km, 270.65 * units.K, 110.906 * units.Pa,
                 0 * units.K / units.m),
            (51 * units.km, 270.65 * units.K, 66.9389 * units.Pa, 
                 0.0028 * units.K / units.m),
            (71 * units.km, 214.65 * units.K, 3.95642 * units.Pa, 
                 float("NaN") * units.K / units.m)
            ]
        _HEIGHT, _TEMPERATURE, _PRESSURE, _TEMPERATURE_GRADIENT = 0, 1, 2, 3


        is_array = hasattr(pressure.magnitude, "__len__")
        if not is_array:
            pressure = [pressure.magnitude] * pressure.units

        # Initialize the return array.
        z = np.full_like(pressure, np.nan) * units.hft

        for i, ((z0, t0, p0, gamma), (z1, t1, p1, _))\
                    in enumerate(zip(_STANDARD_ATMOSPHERE[:-1],
                                     _STANDARD_ATMOSPHERE[1:])):
            p1 = _STANDARD_ATMOSPHERE[i + 1][_PRESSURE]
            indices = (pressure > p1) & (pressure <= p0)
            if i == 0:
                indices |= (pressure >= p0)
            if gamma != 0:
                z[indices] = z0 + 1. / gamma * (t0 - t0 * np.exp(gamma * Rd /\
                                            g * np.log(pressure[indices] / p0)))
            else:
                z[indices] = z0 - (Rd * t0) / g * np.log(pressure[indices] / p0)

        if np.isnan(z).any():
            raise ValueError("flight level to pressure conversion not "
                             "implemented for z > 71km")

        return z if is_array else z[0]
    
    @staticmethod
    def calculate_flight_level(bahamas):
        from metpy.units import units
        
        flv = BAHAMAS.pressure2flightlevel(bahamas.PS.values * units.hPa)
        p_bot, p_top = 101315, 12045
        flv_limits = BAHAMAS.pressure2flightlevel([p_bot, p_top] * units.Pa)
        _pres_maj = np.concatenate([np.arange(top * 10, top, -top) \
                                    for top in (10000, 1000, 100, 10)] + [[10]])
        _pres_min = np.concatenate([np.arange(top * 10, top, -top // 10) \
                                    for top in (10000, 1000, 100, 10)] + [[10]])
        return _pres_min,_pres_maj,flv,flv_limits
    
    @staticmethod
    def add_surface_mask_to_data(bah,cfg_dict,resolution="120s"): 
        """
        

        Parameters
        ----------
        bah       : xr.Dataset
                    Lat,Lon Aircraft Dataset, is mostly BAHAMAS.
        cfg_dict : dict
            configuration dict
        Returns
        -------
        None.

        """
        import Performance
        performance=Performance.performance()
        print("Add surface mask")
        # Get sea_ice mask    
        sea_ice_cls=SEA_ICE(cfg_dict)
        sea_ice_cls.open_sea_ice_ds()
        sea_ice_ds=sea_ice_cls.ds 
        try:
            time_values=bah.TIME[:]
        except:
            time_values=bah.time[:]
        bah_df=pd.DataFrame(data=np.nan,columns=["LAT","LON"],
                             index=pd.DatetimeIndex(np.array(time_values)))
        try:
            bah_df["LAT"]=bah.IRS_LAT.values
        except:
            bah_df["LAT"]=bah.lat.values
        try:
            bah_df["LON"]=bah.IRS_LON.values
        except:
            bah_df["LON"]=bah.lon.values
        #if not 
        bah_df=bah_df.resample(resolution,convention="start").mean()
    
        # Create sea_ice_mask on aircraft
        sea_ice_mask=pd.Series(data=np.nan,index=pd.DatetimeIndex(bah_df.index))
    
        lat_2d=np.array(sea_ice_ds.lat)
        lon_2d=np.array(sea_ice_ds.lon)
        lat_1d=lat_2d.flatten()
        lon_1d=lon_2d.flatten()
        print("check for sea ice concentration")
        for t in range(bah_df.shape[0]):
            if not (np.isnan(bah_df["LAT"].iloc[t])) \
                and not (np.isnan(bah_df["LON"].iloc[t])):
                    #  Calculate differences of aircraft position to 
                    #  land sea mask grid
                    distances=HALO_Devices.vectorized_harvesine_distance(
                                                bah_df["LAT"].iloc[t],
                                                bah_df["LON"].iloc[t],
                                                lat_1d,lon_1d)
            
                    min_geoloc=np.unravel_index(np.argmin(distances,axis=None),
                                        lat_2d.shape)
                    sea_ice_mask.iloc[t]=sea_ice_ds["seaice"][min_geoloc[0],min_geoloc[1]]
                    if bah_df["LAT"].iloc[t]>87:
                        # from upon a specific latitude, the sea ice mask is
                        # not yet provided anymore, but we can certainly 
                        # assume full sea-ice cover.
                        
                        sea_ice_mask.iloc[t]=100
            else:
                pass
            performance.updt(bah_df.shape[0],t)                                                             
        sea_ice_mask=sea_ice_mask/100
        sea_ice_mask=sea_ice_mask.fillna(0)
        print("check for land surface")
        # Now add land mask
        try:
            from ac3airborne.tools import is_land as il
        except:
            print("issues in importing ac3airborne")
            # Manually import shapereader
            import cartopy.io.shapereader as shaper
            import shapely.geometry as shapegeom
            from shapely.ops import unary_union
            from shapely.prepared import prep

            land_shp_fname = shaper.natural_earth(
                            resolution='10m', category='physical', name='land')
            land_geom = unary_union(
                            list(shaper.Reader(land_shp_fname).geometries()))
            land = prep(land_geom)
            def is_land_surface(x, y):
                return land.contains(shapegeom.Point(x, y))


        t=0
        for x, y in zip(bah_df["LON"], bah_df["LAT"]):
            if is_land_surface(x,y):
                sea_ice_mask.iloc[t]=-0.1*int(is_land_surface(x,y))                
            t+=1
            performance.updt(bah_df.shape[0],t)
        bah_df["sea_ice"]=sea_ice_mask.values    
        return bah_df
#-----------------------------------------------------------------------------#
class Dropsondes(HALO_Devices):
    def __init__(self,HALO_Devices_cls):
        #import metpy 
        self.cfg_dict=HALO_Devices_cls.cfg_dict
        self.date=self.cfg_dict["date"]
        self.campaign_name=HALO_Devices_cls.campaign_name
        self.major_data_path=HALO_Devices_cls.major_data_path
        self.name="dropsonde"
        self.device_type="dropsonde"
        self.device_data_path=self.major_data_path+"/"+self.name+"/"
    
    def open_sonde_data_one_var(self,var_name):
        self.sonde_fnames = sorted(glob.glob(self.device_data_path+"*D"+\
                                             self.cfg_dict["date"]+"*"))
        sonde_var_dict={}
        for sonde_file in self.sonde_fnames:
            ds=xr.open_dataset(sonde_file)
            if not var_name in ds.keys():
                raise Exception("This is a wrong variable name given.")
        
            var_series=pd.Series(data=np.array(ds[var_name].values[:]),
                                  index=np.array(ds["gpsalt"][:]))
            time_str=str(ds["launch_time"].values)
            sonde_var_dict[time_str]=var_series
        #self.sonde_dict
        return sonde_var_dict
    
    def open_all_sondes_as_dict(self):
        self.sonde_fnames=sorted(glob.glob(self.device_data_path+"*D"+\
                                           self.cfg_dict["date"]+"*"))
        sonde_dict={}
        s=0
        for sonde_file in self.sonde_fnames:
            ds=xr.open_dataset(sonde_file)
            start_time=str(ds["launch_time"].data)
            for key in ds.keys():
                if s==0:
                    sonde_dict[key]={}
                sonde_dict[key][str(start_time)]=ds[key]
            s+=1
        self.sonde_dict=sonde_dict
        
    def calculate_iwv(self):
        import metpy.calc as mpcalc
        from metpy.units import units
        
        if not hasattr(self,"sonde_dict"):
            self.open_all_sondes_as_dict()
        ### Extra calculated variables
        self.sonde_dict["IWV"]=pd.Series(data=np.nan,
                index=pd.DatetimeIndex(self.sonde_dict["launch_time"].keys()))
        print("Calculate additional meteorological parameters")
        t=0
        for time in self.sonde_dict["launch_time"].keys():
            
            T     = self.sonde_dict['tdry'][time].values * units.degC
            RH    = self.sonde_dict["rh"][time].values * units.percent
            P     = self.sonde_dict["pres"][time].values * units.hPa
            Tdew  = mpcalc.dewpoint_from_relative_humidity(T,RH)
            if t==8:
                print(T,RH,P)
            try:
                pw_value= mpcalc.precipitable_water(P,Tdew)
                self.sonde_dict["IWV"].iloc[t]=pw_value.magnitude/0.99
            
            except:
                pw_value=np.nan
                self.sonde_dict["IWV"].iloc[t]=pw_value
            
            t+=1
    
    def calculate_ivt(self):
        """
        
    
        Parameters
        ----------
        Dropsondes : dict
            The Dropsondes dictionary containing all data on unified grid.
    
        Returns
        -------
        Dropsondes : dict
            Updated Dropsondes dictionary with IVT-keys on unified grid.
    
        """
        import metpy.calc as mpcalc
        from metpy.units import units
        if not hasattr(self,"sonde_dict"):
            self.open_all_sondes_as_dict()
        
        if not "q" in list(self.sonde_dict.keys()):
            t=0
            # Start IVT Calculations
            series_index=pd.DatetimeIndex(self.sonde_dict["launch_time"].keys())
            self.sonde_dict["IVT_u"] = pd.Series(data=np.nan,
                                            index=series_index)
            self.sonde_dict["IVT_v"] = pd.Series(data=np.nan,index=series_index)
            self.sonde_dict["IVT"]   = pd.Series(data=np.nan,index=series_index)
            self.sonde_dict["q"]     = {}
            t=0
            for time in self.sonde_dict["launch_time"].keys():
                
                T     = self.sonde_dict['tdry'][time].values * units.degC
                RH    = self.sonde_dict["rh"][time].values * units.percent
                P     = self.sonde_dict["pres"][time].values * units.hPa
                Tdew  = mpcalc.dewpoint_from_relative_humidity(T,RH)
                MR    = mpcalc.mixing_ratio_from_relative_humidity(P, T, RH)
                wspeed= self.sonde_dict["wspd"][time].values * \
                                                    units.meter / units.second
                wdir= np.deg2rad(self.sonde_dict["wdir"][time].values)
            
                # if q is not already calculated, it has to be computed
                q_hum=MR.magnitude/(MR.magnitude+1)
                #mpcalc.specific_humidity_from_mixing_ratio(MR)
                #q.columns=Dropsondes["Wspeed"].columns
        
                q_series     = pd.Series(data=q_hum,
                              index=P.magnitude*100)
                self.sonde_dict["q"][series_index[t]]=q_series
                q_series= q_series.dropna()
                q_series=q_series.sort_index()
                g= 9.81
                u_metpy,v_metpy= mpcalc.wind_components(wspeed,wdir)
            
                u=pd.Series(data=u_metpy,index=P.magnitude*100)
                v=pd.Series(data=v_metpy,index=P.magnitude*100)
                u=u.loc[q_series.index]
                v=v.loc[q_series.index]
                qu=q_series*u
                qv=q_series*v
                qu=qu.dropna()
                qv=qv.dropna()
                self.sonde_dict["IVT_u"].iloc[t] = 1/g*np.trapz(qu,x=qu.index)
                self.sonde_dict["IVT_v"].iloc[t] = 1/g*np.trapz(qv,x=qv.index)
                self.sonde_dict["IVT"].iloc[t]   = np.sqrt(
                                        self.sonde_dict["IVT_u"].iloc[t]**2+
                                        self.sonde_dict["IVT_v"].iloc[t]**2)
                t+=1
                
    def calc_integral_variables(self,integral_var_list=[""]):
        if "IVT" in integral_var_list:
            self.calculate_ivt()
        if "IWV" in integral_var_list:
            self.calculate_iwv()
            
        
#-----------------------------------------------------------------------------#
class HAMP(HALO_Devices):
    def __init__(self, HALO_Devices_cls,
                 version='raw', **kwargs):
        """
        microwave radiometer onboard HALO (part of HAMP). 
        Time will be given in seconds since 2017-01-01 00:00:00 UTC.
        HALO Devices class containing all devices mounted on HALO during specific 
            campaign. Mostly, HALO_Devices.cfg_dict is used 
            for further specifications.
        
        path : str
			Path of raw HALO HAMP-MWR data. The path must simply link to HALO 
            and then contain the datefurther folders that then link to the 
            HAMP MWR receivers (KV, 11990, 183): Example: path = 
            "/data/obs/campaigns/eurec4a/HALO/" -> contains 
            "./20020205/radiometer/" + ["KV/", "11990/", "183/"].
		
        
        version : str
			Version of the HAMP MWR data. Options available: 'raw'

    	**kwargs:
            urn_DS : bool
		If True, the imported xarray dataset will also 
        be set as a class attribute.
        """


        self.cfg_dict=HALO_Devices_cls.cfg_dict
        self.date=self.cfg_dict["date"]
        self.campaign_name=HALO_Devices_cls.campaign_name
        self.major_data_path=HALO_Devices_cls.major_data_path
        self.name="HAMP"
        self.long_name="HALO Microwave Package"
        
        self.device_type="radiometer"
        self.raw_hamp_path=self.major_data_path
        self.raw_mwr_path=self.major_data_path+self.device_type+"/"
        self.unified_hamp_path=self.major_data_path+"all_nc/"
        
        self.radiometer_modules=["KV","11990","183"]
        
        # init attributes:
        self.freq = dict()
        self.time = dict()
        self.time_npdt = dict()
        self.TB = dict()
        self.flag = dict()
        self.version=version
    
        #if ('return_DS' in kwargs.keys()) and kwargs['return_DS']:
        self.DS = dict()
    
    def open_raw_hamp_data(self):
        tb_data_dict={}
        self.raw_hamp_tb_dict={}
        date=self.cfg_dict["date"]
        # Each hamp module is stored in separate directory, so loop over. 
        for var in self.radiometer_modules:
        
            # List files from specified date
            radiometer_files=glob.glob(self.major_data_path+\
                                   "radiometer/"+var+"/*"+date+"*")
            # Display var name
            print("Open Radiometer Channel: ",var)
        
            # If no files were found, it's probably because the date 
            # is written in yymmdd format in file names instead of yyyymmdd
            if len(radiometer_files)==0:
                #flight_backup=flight    
                date=date[2::]
                radiometer_files=glob.glob(self.major_data_path+\
                                   "radiometer/"+var+"/*"+date+"*")
            
            # Open issue --> if flight reaches onto second UTC date
            # Check if flight ended after 00z
            #uni_time_str=uni_time.date[-1].strftime('%Y%m%d')[2::]
            #if not uni_time_str==flight:
            #    file_name_2=glob.glob(cfg_dict["device_data_path"]+\
            #                       "radiometer/"+var+"/*"+uni_time_str+"*")
            #    radiometer_files.append(file_name_2)
            #flight=flight_backup
        
            # If radiometer files for day exists    
            if not len(radiometer_files)==0:
                # Loop all files from day which should work either way: 
                # if it is only one file and if it contains several files
                for file in radiometer_files:
                    radiometer_ds=xr.open_dataset(file)
                    #try:
                    #    self.freq[module] = DS.frequencies.values			# in GHz
                    #except:
                    #    self.freq[module] = DS.Freq.values
                    
                    tb_data_dict[file]=pd.DataFrame(
                        data=np.array(radiometer_ds["TBs"][:].astype(float)),
                        index=pd.DatetimeIndex(np.array(radiometer_ds.time[:])),
                        columns=np.array(radiometer_ds["Freq"][:]).\
                            astype(float).round(2).astype(str))
                if len(radiometer_files)>1:
                    self.raw_hamp_tb_dict[var]=pd.concat(*tb_data_dict)
                else:
                    self.raw_hamp_tb_dict[var]=tb_data_dict[file]
    
    def open_hamp_raw_data_extended(self,**kwargs):
        if self.version == 'raw':
            reftime = "2017-01-01 00:00:00"		# in UTC
		
            # import data: Identify receivers of the current day:
            self.avail = {module:False for module in self.radiometer_modules}
            radiometer_dict = dict()
            for module in self.radiometer_modules:
                radiometer_dict[module]=[]
                # List files from specified date
                radiometer_files=glob.glob(self.major_data_path+\
                                   "radiometer/"+module+"/*"+self.date+"*.nc")
                # Display var name
                print("Open Radiometer Channel: ",module)
        
                # If no files were found, it's probably because the date 
                # is written in yymmdd format in file names instead of yyyymmdd
                if len(radiometer_files)==0:
                    #flight_backup=flight    
                    date_to_open_file=self.date[2::]
                    radiometer_dict[module]=glob.glob(self.major_data_path+\
                                   "radiometer/"+module+"/*"+\
                                       date_to_open_file+"*")
                else:
                    radiometer_dict[module]=radiometer_files
                if len(radiometer_dict[module])==0:
                    raise FileNotFoundError("No HAMP data found", 
                                            "check file name")
                else:
                    self.avail[module] = True
                    # import data: cycle through receivers and
                    # import when possible:
                    if len(radiometer_dict[module])>1:
                        DS_init = xr.open_mfdataset(radiometer_dict[module], 
                                               concat_dim='time', 
                                               combine='nested')
                        # reduce unwanted dimensions: ---> seems to be not needed
                        #DS['frequencies'] = DS.frequencies[0,:]
					
                    else: 
                        DS_init=xr.open_dataset(radiometer_dict[module][0])     
                    # Sampling of HAMP has been increased to 4 Hz, time is rounded
                    # to seconds
                    if "RSFactor" in DS_init.keys():
                        if DS_init.RSFactor.values!=1:
                            # seconds have to be adjusted for milliseconds
                            sec_array=np.array([0,0.25,0.5,0.75])
                            sec_array_concat=np.tile(sec_array,int(
                                            DS_init.time.shape[0]/\
                                                int(DS_init.RSFactor.values)))
                            pd_secs=pd.to_timedelta(sec_array_concat,unit="s")
                            DS=DS_init.assign_coords(time=("time",
                                                DS_init.time.values+pd_secs))
                            DS.time.attrs=DS_init.time.attrs
                        else:
                            DS=DS_init.copy()
                    else:
                        DS=DS_init.copy()
                    del DS_init                            
                    try:
                        self.freq[module] = DS.frequencies.values			# in GHz
                    except:
                        self.freq[module] = DS.Freq.values
                    self.time_npdt[module] = DS.time.values			# in numpy datetime64
                    self.time[module] = HALO_Devices.numpydatetime64_to_reftime(
                                            DS.time.values, reftime) # in seconds since 2017-01-01 00:00:00 UTC
                                            # Unify variable names by defining class attributes:
                    
                    self.TB[module] = DS.TBs.values					# in K, time x freq
                    try:
                        self.flag[module] = DS.rain_flag.values			# rain flag, no units
                    except:
                        self.flag[module] = DS.RF.values
#                    self.DS[module] = DS

    def open_processed_hamp_data(self,open_calibrated=False,
                                 newest_version=True):
       import campaign_netcdf
       nc_path=self.cfg_dict["device_data_path"]+"all_nc/"
       date=self.cfg_dict["flight_date_used"]
       if newest_version:       
          data_file=campaign_netcdf.CPGN_netCDF.identify_newest_version(
              nc_path,device="radiometer",date=date,
              for_calibrated_file=open_calibrated)
       else:
           data_file=nc_path+"HALO_"+self.cfg_dict["campaign"]+\
               "_"+"radiometer"+"_unified_"+self.cfg_dict["flight"]+"_"+date+\
                   "_v"+str(self.cfg_dict["version"])+"."+\
                       str(self.cfg_dict["subversion"])+".nc"
       print(data_file)
       if not open_calibrated:
           self.processed_hamp_ds=xr.open_dataset(data_file)
       else:
            self.calib_processed_hamp_ds=xr.open_dataset(data_file)
                
    def access_HAMP_TB_calibration_coeffs(self):
       self.calibration_path=self.major_data_path+\
           "HAMP_MWR_TB_clear_sky_offset_correction/"#"calibration_coeffs/"
       #self.calibration_file="clear_sky_sonde_comparison_ALL_J3v0.9.2_radiometer_daily.nc"
       self.calibration_file="HALO-AC3_HALO_HAMP_TB_offset_correction_"+\
                               self.date+".nc"
       self.tb_calib_coeff_ds=xr.open_dataset(self.calibration_path+\
                                         self.calibration_file)
    
    def get_HAMP_TB_calibration_coeffs_of_flight(self):
        self.access_HAMP_TB_calibration_coeffs()
        date=str([*self.cfg_dict["Flight_Dates_used"]][0])
        self.flight_tb_offset_coeff_ds=self.tb_calib_coeff_ds[\
                                        "offset"].sel({"date":date})
        self.flight_tb_slope_coeff_ds=self.tb_calib_coeff_ds[\
                                        "slope"].sel({"date":date})
         
    
#-----------------------------------------------------------------------------#
class RADAR(HALO_Devices):
    def __init__(self,HALO_Devices_cls):
        #super(HALO_Radar,self).__init__()
        self.cfg_dict=HALO_Devices_cls.cfg_dict
        self.campaign_name=HALO_Devices_cls.campaign_name
        self.major_data_path=HALO_Devices_cls.major_data_path
        self.name="mira"
        self.device_type="radar"
        self.raw_radar_path=self.major_data_path+self.device_type+"/"
        self.unified_radar_path=self.major_data_path+"all_nc/"
        self.att_corrected_radar_path=self.major_data_path+self.device_type+\
                                    "_"+self.name+"/"                
    # Data Opening Procedures  
    def list_raw_radar_files_for_flight(self):
        print(self.raw_radar_path+"*"+str(self.date_of_flight)+'*.nc')
        self.radar_fnames = glob.glob(self.raw_radar_path+"*"+\
                                 str(self.date_of_flight)+'*.nc')
        print("RADAR files for flight ",self.flight_of_interest,
              ":",self.radar_fnames)

    def list_att_radar_files_for_flight(self):
        files_to_check=self.att_corrected_radar_path+\
                                        "*"+str(self.date_of_flight)+\
                                        "*v"+str(self.cfg_dict["version"])+"."+\
                                        str(self.cfg_dict["subversion"])+'.nc'
        self.radar_attcorr_fnames=glob.glob(files_to_check)
        
        print("Radar attcorr files for flight",
              self.flight_of_interest,
              ":",self.radar_attcorr_fnames)

    def open_raw_radar_data(self,flight,date,with_dbz=True):
        self.flight_of_interest=flight
        self.date_of_flight=date
        self.list_raw_radar_files_for_flight()
        if len(self.radar_fnames)==1:
            self.raw_radar_ds=xr.open_dataset(self.radar_fnames[0])
            #xr.decode_cf()
        else:
            self.raw_radar_ds=xr.open_mfdataset(self.radar_fnames)
        self.raw_radar_ds=self.adapt_time_stamp(self.raw_radar_ds)
        if with_dbz:
           self.calc_dbz_from_z() 
        #self.raw_radar_ds=self.decode_time(self.raw_radar_ds)

    # Old version
    def load_attitude_corrected_data(self):
        self.flight_of_interest=self.cfg_dict["flight"]
        self.date_of_flight=self.cfg_dict["date"]
        self.list_att_radar_files_for_flight()
        if len(self.radar_attcorr_fnames)==1:
            self.attcorr_radar_ds=xr.open_dataset(self.radar_attcorr_fnames[0])
        else:
            self.attcorr_radar_ds=xr.open_mfdataset(self.radar_attcorr_fnames)
    
    def open_attitude_corrected_data(self):
        import campaign_time as Campaign_Time #import specify_dates_to_use as specify_dates

        ### ----- radar needs special treatment ------------------------###
        cmpgn_time=Campaign_Time.Campaign_Time(self.cfg_dict["campaign"],
                                               self.cfg_dict["date"])
        flightdates_use = cmpgn_time.specify_dates_to_use(self.cfg_dict)
        attitude_correction_by_bahamas=True
        try:
            self.load_attitude_corrected_data()
            radar_is_attitude_corrected=True
        except:
            radar_is_attitude_corrected=False
            import radarattitude        
            if not attitude_correction_by_bahamas:
                # Use SMART data to correct the attitude by the radar    
                print("Correct the radar attitude by SMART")
            else:
                print("Correct the radar attitude by BAHAMAS")        
        
                radarattitude.run_att_correction(
                                    self.cfg_dict["Flight_Dates_used"],
                                    self.cfg_dict,use_smart=\
                                        not attitude_correction_by_bahamas)
                
                radar_is_attitude_corrected=True
            self.load_attitude_corrected_data()    
                
    def open_processed_radar_data(self,newest_version=True,
                                  reflectivity_is_calibrated=True):
       import campaign_netcdf
       nc_path=self.cfg_dict["device_data_path"]+"all_nc/"#"radar_"+\
                   	#str([*self.cfg_dict["Flight_Dates_used"]][0])
       if newest_version:       
          data_file=campaign_netcdf.CPGN_netCDF.identify_newest_version(nc_path,
              date=self.cfg_dict["flight_date_used"],
              for_calibrated_file=reflectivity_is_calibrated)
       if not reflectivity_is_calibrated:
           self.processed_radar_ds=xr.open_dataset(data_file)
       else:
            self.calib_processed_radar_ds=xr.open_dataset(data_file)
    
    def open_version_specific_processed_radar_data(self,version="undefined",
                                                   for_calibrated_file=False):
        import campaign_netcdf as Campaign_netCDF
        try:
            used_flight_date=str([*self.cfg_dict["Flight_Dates_used"]][0])
        except:
            used_flight_date=str(self.cfg_dict["flight_date_used"])
        nc_path=self.cfg_dict["device_data_path"]+"all_nc/"#"#radar_"+\
        #used_flight_date
        print("Look for Radar data in ",nc_path)
        default_data_file=Campaign_netCDF.CPGN_netCDF.identify_newest_version(
            nc_path,date=used_flight_date,
            for_calibrated_file=for_calibrated_file)
        if version=="undefined":
            # the newest version will be used
            data_file=default_data_file
        else:
            data_file_l = list(default_data_file)
            data_file_l[-6:-3] = list(version)
            data_file = "".join(data_file_l)
            # change the name of the data_File to be read
            #data_file[-6:-4]=version
        try:
            print("Opened specific version:",data_file)
            processed_ds=xr.open_dataset(data_file)
        
        except FileNotFoundError as error:
            print(error)
            print("The version you have chosen is not existent already.",
                                "the last newest version will be addded.")
            processed_ds=xr.open_dataset(default_data_file)
            
        return processed_ds
    #-------------------------------------------------------------------------#
    # Time stamp adjustments 
    def adapt_time_stamp(self,ds):
        if not np.issubdtype(ds.time.values.dtype,np.datetime64) or\
            not str(ds.time.values.dtype).startswith('datetime64[ns]'):
            # The raw radar time stamp has a complexe format
            attrs= {'units': str(ds.time.long_name)}
            # {'units': str(ds.time.units),
            # 'long_name': str(ds.time.long_name)}
        
            temporary_ds=xr.Dataset({'time':('time',np.array(ds.time[:]),attrs)})
            temporary_ds=xr.decode_cf(temporary_ds)
            ds["time"]=temporary_ds["time"]
        return ds

    def decode_time(self,ds):
        # from old Measurement_Instruments
        if not np.issubdtype(ds.time.values.dtype,
                             np.datetime64):
            
            attrs= {'units': str(ds.time.long_name)}
        
            temporary_ds=xr.Dataset({'time':\
                        ('time',np.array(ds.time[:]),attrs)})
            temporary_ds=xr.decode_cf(temporary_ds)
            ds["time"]=temporary_ds["time"]
            del temporary_ds
        
        else:
            pass
        
        return ds
    
    def calc_dbz_from_z(self,raw_measurement=True):
        if raw_measurement:
            ds=self.raw_radar_ds
        else:
            ds=self.processed_radar_ds
            
        dBZg = np.array(10*np.emath.log10(ds['Zg'][:]))
        dBZg = dBZg.astype(np.float32)
        #% Only keep real part of array (imaginary numbers were created when
        #% taking log10 of -Inf: log10(-Inf) = Inf +      1.36437635384184i)
        dBZg = np.real(dBZg)
        #% And convert positive infinity back to negative infinity
        dBZg = np.where(dBZg==np.inf,-np.inf,dBZg)
        if "dBZg" in ds.variables:
            dBZg_attrs=ds["dBZg"].attrs
        
        if raw_measurement:
            self.raw_radar_ds["dBZg"]=xr.DataArray(data=dBZg.T,
                                               dims=ds.dims)
        else:
            self.processed_radar_ds["dBZg"]=xr.DataArray(data=dBZg,
                                               dims=ds.dims)
            if "dBZg" in ds.variables:
                self.processed_radar_ds["dBZg"].attrs=dBZg_attrs
            # Fill values back again
            self.processed_radar_ds["Zg"]=self.processed_radar_ds["Zg"].\
                        fillna(float(self.cfg_dict["missing_value"]))
            self.processed_radar_ds["dBZg"]=self.processed_radar_ds["dBZg"].\
                        fillna(float(self.cfg_dict["missing_value"]))
    
    def calc_db_from_ldr(self,raw_measurement=True):
        if raw_measurement:
            ds=self.raw_radar_ds
        else:
            ds=self.processed_radar_ds
        
        LDRg = np.array(10*np.emath.log10(ds['LDRg'][:]))
        LDRg = np.real(LDRg)
        LDRg=np.where(LDRg==np.inf,-np.inf,LDRg)
        if "LDRg" in ds.variables:
            LDRg_attrs=ds["LDRg"].attrs
        if raw_measurement:
            self.raw_radar_ds["LDRg"]=xr.DataArray(data=LDRg.T,dims=ds.dims)
        else:
            self.processed_radar_ds["LDRg"]=xr.DataArray(data=LDRg.T,
                                                         dims=ds.dims)
            if "LDRg" in ds.variables:
                self.processed_radar_ds["LDRg"].attrs=LDRg_attrs
            # Fill values back again
            self.processed_radar_ds["LDRg"]=self.processed_radar_ds["LDRg"].\
                        fillna(float(self.cfg_dict["missing_value"]))
            self.processed_radar_ds["LDRg"]=self.processed_radar_ds["LDRg"].\
                        fillna(float(self.cfg_dict["missing_value"]))
    
    #-------------------------------------------------------------------------#
    # Calibration
    def get_calibration(self):
        flight=self.cfg_dict["Flight_Dates_used"].index[0]
        ## --> for specific campaigns
        if (self.cfg_dict["campaign"]=="NAWDEX") or\
            (self.cfg_dict["campaign"]=="NARVAL-II"):
                #self.cfg_dict["campaign"]="NAWDEX"
                self.dB_offset = 8.3 # Ewald  radar calibration
        elif self.cfg_dict["campaign"]=="EUREC4A":
            self.dB_offset=-1.7 #?
        elif self.cfg_dict["campaign"]=="HALO_AC3":
            dB_offsets={"RF01":-2.1,
                        "RF09":-2.1,
                        "RF12":-1.2,
                        "RF13":-1.6,
                        "RF16":-1.1,
                        "RF17":-1.3,
                        "mean":-1.6}
            if flight in [*dB_offsets.keys()]:
                self.dB_offset=dB_offsets[flight] # determined by F. Ewald
            else:
                self.dB_offset=dB_offsets["mean"]
    
    def show_calibration(self):
        self.get_calibration()
        print("The raw reflectivity has a offset of",self.dB_offset,"dB",
              " according to Ewald (2019)")
        print("This offset has to be added to Zg by:",
              "Zg_calib=Zg_raw*10**(0.1*self.dB_offset)")
    def get_gaseous_attenuation(self):
        flight=self.cfg_dict["flight"]
        aux_path=self.cfg_dict["device_data_path"]+"/auxiliary/"
        try:
            gas_file=glob.glob(aux_path+"*"+flight+"*.nc")
        except:
            gas_file=glob.glob(aux_path+"*"+flight[0]+"*.nc")
        self.gas_ds=xr.open_dataset(gas_file[0],
                                    engine="netcdf4")
    
    def correct_for_gaseous_attenuation(self):
        self.get_gaseous_attenuation()
        att_coeffs=self.gas_ds["Gaseous_twowayatt"]
        att_coeffs=att_coeffs.sel({"alt":slice(12000,0)})
        att_coeffs=att_coeffs.sortby("alt")
        # Gaseous attenuation has to be interpolated onto unified dataset
        att_coeffs_df=pd.DataFrame(
            data=np.array(att_coeffs.values[:]),
            index=pd.DatetimeIndex(att_coeffs.time[:]),
            columns=np.array(att_coeffs["alt"].values[:]))
        
        # Time interpolation
        att_coeffs_df=att_coeffs_df.iloc[1::,:].resample("1s").mean()
        # Gaseous attenuation is given Z, not dBZ
        att_coeffs_dbz=pd.DataFrame(
            data=np.array(10*np.emath.log10(att_coeffs_df.values[:])),
            index=att_coeffs_df.index,
            columns=np.array(att_coeffs_df.columns))
        # Spatial linear interpolation onto unified height grid
        alt_values=att_coeffs_dbz.columns #old heights roughly (15 m)
        new_alt_values=self.radar_ds["height"].values # unified 30 m resolution
        #
        regrid_att_coeffs_z=np.array(
            [np.interp(new_alt_values, alt_values, att_coeffs_df.iloc[i,:])\
             for i in range(att_coeffs_df.shape[0])])
        regrid_att_coeffs=np.array(
            [np.interp(new_alt_values, alt_values, att_coeffs_dbz.iloc[i,:])\
                            for i in range(att_coeffs_dbz.shape[0])])
        # make dataframe for later xr.DataArray out of it
        regrid_att_coeffs=pd.DataFrame(data=regrid_att_coeffs,
                              index=att_coeffs_df.index,columns=new_alt_values)
        regrid_att_coeffs_z=pd.DataFrame(data=regrid_att_coeffs_z,
                index=att_coeffs_df.index,columns=new_alt_values)
        #alt_values=att_coeffs_dbz.columns


        self.radar_ds["Gas_Attenuation"]=xr.DataArray(
                        data=regrid_att_coeffs_z,dims=self.radar_ds.dims)
        self.radar_ds["dBZg_old"]=self.radar_ds["dBZg"].copy()
        self.radar_ds["dBZg"]=self.radar_ds["dBZg_old"]+\
            self.radar_ds["Gas_Attenuation"]
        print("Radar reflectivty dBZg corrected for gaseous attenuation")
        self.radar_ds["Zg_old"]=self.radar_ds["Zg"].copy()
        self.radar_ds["Zg"]=self.radar_ds["Gas_Attenuation"]*self.radar_ds["Zg"]
        print("Radar reflectivity Zg corrected for gaseous attenuation")
    # Melting layer detection and precipitation phase
    @staticmethod
    def find_melting_layer(radar_dict,vertical_value_to_use="max"):
        """
        This routine identifies the melting layer using the radar LDR to then
        distinguish between precipitation types. 
        The ML detection is based on Austen (2023), who developped this 
        method in his Bachelor thesis.

        Parameters
        ----------
        radar_dict : TYPE
            DESCRIPTION.
        vertical_value_to_use : TYPE, optional
            DESCRIPTION. The default is "max".

        Returns
        -------
        ldr_mlayer_height : TYPE
            DESCRIPTION.
        low_ldr_df : TYPE
            DESCRIPTION.
        ldr_cutted_df : TYPE
            DESCRIPTION.
        mlayer_mask : TYPE
            DESCRIPTION.

        """
        ldr_threshold=-17
        lower_height_thres=5
        maximum_height=2000
        maximum_gradient=60
        # based on Austen et al. 2023    
        height=np.array(radar_dict["height"][:])
        ldr_df=pd.DataFrame(data=np.array(radar_dict["LDRg"][:]),
                     columns=height,
                     index=pd.DatetimeIndex(
                         np.array(radar_dict["time"])))
    
        ldr_cutted_df=ldr_df.copy()
    
        low_ldr_df=ldr_cutted_df.iloc[:,lower_height_thres:70]
        #-------------------------------------------------------------------------#
        # LDR threshold
        low_ldr_df[low_ldr_df<ldr_threshold]=np.nan
    
        # which value to use if a vertical column is above ldr_threshold
        # my method was the maximum value which always shifts the bright band above
        # version Austen is 
        if vertical_value_to_use=="max":    
            ldr_mlayer_height=low_ldr_df.idxmax(axis=1)
        elif vertical_value_to_use=="lowest": # Austen et al. 2023
            #this is a bad method but performs well
            mask_of_ldr_values=low_ldr_df/low_ldr_df
            ldr_mlayer_height=mask_of_ldr_values.idxmin(axis=1)
        else:
            Exception("you have to deside for one of the options (max,lowest)")
        #-------------------------------------------------------------------------#
        # LDR should always lie below maximum height defined above
        ldr_mlayer_height[ldr_mlayer_height>maximum_height]=np.nan
        
        #-------------------------------------------------------------------------#
        # Gradient criteria (continuity), it is less strong than in Austen et al.
        ldr_ml_height_gradient=ldr_mlayer_height.diff()
        strong_gradient=ldr_ml_height_gradient[\
                            abs(ldr_ml_height_gradient)>maximum_gradient]
        
        # set value to nan for too strong gradients
        ldr_mlayer_height.loc[strong_gradient.index]=np.nan
        #-------------------------------------------------------------------------#
        # 5s rolling mean
        ldr_mlayer_height=ldr_mlayer_height.rolling("5s",min_periods=5).mean()   
        #-------------------------------------------------------------------------#
        # Melting layer mask
        mlayer_mask=pd.Series(data=np.zeros(ldr_mlayer_height.shape[0]),
                              index=ldr_mlayer_height.index)
        mlayer_mask[~ldr_mlayer_height.isnull()]+=1
        #-------------------------------------------------------------------------#
        # max 10 s gap filling via interpolation
        ldr_mlayer_height=ldr_mlayer_height.interpolate(
            method="polynomial",order=5,limit=10,limit_area="inside",
            limit_direction="both")
        # Extrapolate
        ldr_mlayer_height=ldr_mlayer_height.interpolate(method="polynomial",order=5,
            limit_area="outside",limit=10,fill_value="extrapolate")
        # where mlayer mask was zero, so no melting layer
        # but new interpolation now yields values
        ## Reapply the upper altitude threshold but now set to 2000
        # LDR should always lie below maximum height defined above
        ldr_mlayer_height[ldr_mlayer_height>maximum_height]=2000
        
        # period is uncertain
        condition_1=mlayer_mask==0
        condition_2=~ldr_mlayer_height.isnull()
        both_conditions= condition_1 & condition_2
        mlayer_mask[both_conditions]=2
        #
        mlayer_mask[ldr_mlayer_height.between(0,270,inclusive="right")]=2
        #------------------------------------------------------------------------- #
        #
        return ldr_mlayer_height,low_ldr_df,ldr_cutted_df,mlayer_mask
    
    @staticmethod
    def classify_precipitation_type(radar_dict, bb_height,bb_mask):
        """
        This routine uses the detected melting layer to then separate and 
        classify the precipitation phases from the HALO radar near the surface.
        This routine is based on Austen (2023),  who developped this 
        method in his Bachelor thesis.

        
        Parameters
        ----------
        radar_dict : TYPE
            DESCRIPTION.
        bb_height : TYPE
            DESCRIPTION.
        bb_mask : TYPE
            DESCRIPTION.

        Returns
        -------
        precip_type_series : TYPE
            DESCRIPTION.
        sfc_zg_series : TYPE
            DESCRIPTION.

        """
        surface_Zg=radar_dict["Zg"][:,4]
        surface_Zg=surface_Zg.where(surface_Zg!=-888.)
        sfc_zg_series=pd.Series(data=np.array(surface_Zg[:]),
                                index=pd.DatetimeIndex(
                                   np.array(surface_Zg.time[:])))
        surface_type=pd.Series(data=np.array(radar_dict["radar_flag"].values[:,0]),
                               index=sfc_zg_series.index)
        
        precip_type_series=pd.Series(data=np.nan,
                                     index=pd.DatetimeIndex(
                                         np.array(surface_Zg.time[:])))
        precip_type_series[sfc_zg_series.isnull()]=0
        precip_type_series[~sfc_zg_series.isnull()]=1.0 # snow
        precip_type_series.loc[bb_mask==1.0]=2.0 # rain
        precip_type_series.loc[bb_mask==2.0]=3.0 # uncertain    
        precip_type_series[surface_type==-0.1]=-1.0
        # take as last conditions do not anymore look for clear rain defined signals
        # but maybe it is not important once applied to the rain reflectivities
        return precip_type_series,sfc_zg_series
    # Precipitation rates from catalogue of Z-R/S relationships
    @staticmethod
    def get_rain_rate(zg_series):
        rain_rate=pd.DataFrame(data=np.nan, index=zg_series.index,
                               columns=["norris","palmer","chandra"])
        rain_rate["norris"]   = 1/466*zg_series**(1/1.47)
        rain_rate["palmer"]   = 1/200*zg_series**(1/1.6)
        rain_rate["chandra"]  = 1/177*zg_series**(1/1.11)
        rain_rate["mean_rain"]= rain_rate.mean(axis=1)
        return rain_rate
    @staticmethod
    def get_snow_rate(zg_series):
        snow_rate=pd.DataFrame(data=np.nan,index=zg_series.index,
                              columns=["schoger","matrosov","heymsfield"])
        snow_rate["schoger"]   = 1/77.67*zg_series**(1/1.22)
        snow_rate["matrosov"]  = 1/56*zg_series**(1/1.2)
        snow_rate["heymsfield"]= 1/10.13*zg_series**(1/1.92)
        snow_rate["mean_snow"] = snow_rate.mean(axis=1) 
        return snow_rate
    @staticmethod
    def take_correct_precipitation_rates(zg_series_dict,surface_mask,bb_mask,z_for_snow="Z_e"):
        precipitation_rate=pd.DataFrame(data=np.nan,
                                        index=zg_series_dict["zg"].index,
                               columns=["r_norris","r_palmer","r_chandra",
                                       "s_schoger","s_matrosov","s_heymsfield",
                                       "mean_snow","mean_rain","mean_unc",
                                       "surface","precip_phase"])
        default_rain_rate=RADAR.get_rain_rate(zg_series_dict["zg"])
        if z_for_snow=="Z_e":
            z_snow=zg_series_dict["ze"]
        else:
            z_snow=zg_series_dict["zg"]
            
        default_snow_rate=RADAR.get_snow_rate(z_snow)
        #Snow rate
        precipitation_rate["s_schoger"].loc[bb_mask==1]=default_snow_rate["schoger"]
        precipitation_rate["s_matrosov"].loc[bb_mask==1]=default_snow_rate["matrosov"]
        precipitation_rate["s_heymsfield"].loc[bb_mask==1]=default_snow_rate["heymsfield"]
        precipitation_rate["mean_snow"].loc[bb_mask==1]=default_snow_rate["mean_snow"]
        #Rain rate
        precipitation_rate["r_norris"].loc[bb_mask==2]=default_rain_rate["norris"]
        precipitation_rate["r_palmer"].loc[bb_mask==2]=default_rain_rate["palmer"]
        precipitation_rate["r_chandra"].loc[bb_mask==2]=default_rain_rate["chandra"]
        precipitation_rate["mean_rain"].loc[bb_mask==2]=default_rain_rate["mean_rain"]
        #---------------------------------------------------------------------#
        # for uncertain tak all
        precipitation_rate["mean_mixed"]=0.0
        precipitation_rate["max_mixed"]=0.0
        precipitation_rate["min_mixed"]=0.0
        #Snow rate
        unc_rate_mean_snow=default_snow_rate.loc[bb_mask==3]["mean_snow"]
        #Rain rate
        unc_rate_mean_rain=default_rain_rate.loc[bb_mask==3]["mean_rain"]
        precipitation_rate["mean_mixed"].loc[bb_mask==3]=\
            (unc_rate_mean_snow+unc_rate_mean_rain)/2
        precipitation_rate["max_mixed"].loc[bb_mask==3]=default_snow_rate["heymsfield"]
        precipitation_rate["min_mixed"].loc[bb_mask==3]=default_rain_rate["norris"]
        #---------------------------------------------------------------------#
        precipitation_rate["surface"]=surface_mask.values
        precipitation_rate["precip_phase"].loc[bb_mask==1]="snow"
        precipitation_rate["precip_phase"].loc[bb_mask==2]="rain"
        precipitation_rate["precip_phase"].loc[bb_mask==3]="uncertain"
        precipitation_rate["precip_phase"].loc[bb_mask==-1]="land"
        return precipitation_rate    
    @staticmethod
    def mimic_attenuation(processed_radar,surface_mask,precip_type_series,
                          attenuation_value=4):
        # Create higher radar reflectivities 
        strong_radar=processed_radar.copy()
        # Add 4 dBZ that can be expected due to radar attentuation (gaseous and hydrometeor)
        strong_radar["dBZg"]=strong_radar["dBZg"]+attenuation_value 
        strong_radar["Zg"]=10**(1/10*strong_radar["dBZg"])
        strong_z_series_dict={}
        strong_z_series_dict["zg"]=pd.DataFrame(strong_radar["Zg"][:,4],
                index=pd.DatetimeIndex(np.array(strong_radar["Zg"].time[:])))
        #calculate precipitation rate for stronger z
        strong_precip_rate=RADAR.take_correct_precipitation_rates(
            strong_z_series_dict,surface_mask,precip_type_series,z_for_snow="Zg")
        strong_precip_rate=strong_precip_rate.fillna(0)
        strong_precip_rate["rate"]=strong_precip_rate["mean_snow"]+\
            strong_precip_rate["mean_rain"]+strong_precip_rate["mean_mixed"]
        return strong_precip_rate
    
    @staticmethod
    def calc_radar_cfad(df,
                    reflectivity_bins=np.linspace(-60,60,121),
                    ):
        
        """
        Parameters
        ----------
        df : pd.DataFrame 
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
        #if raw_measurements:
        #df=input_data
        #else:
        #    pass
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
        print("this take a while")
            
        for height in df.columns:
            # Start grouping by pd.cut and pd.value_counts
            bin_groups=pd.cut(df[height],reflectivity_bins)
            height_hist=pd.value_counts(bin_groups).sort_index()        
            
            #Assign counted bins to histogram dataframe
            cfad_hist.iloc[i,:]=height_hist.values/df.shape[0]#(cfad_hist.shape[0]*cfad_hist.shape[1])
            cfad_hist_absolute.iloc[i,:]=height_hist.values
            updt(len(df.columns),i)
            i+=1
            
        cfad_hist_dict["relative"]=cfad_hist
        cfad_hist_dict["absolute"]=cfad_hist_absolute
        # Finished, return histogram    
        return cfad_hist_dict
#-----------------------------------------------------------------------------#
class SMART(HALO_Devices):
    def __init__(self,HALO_Devices_cls):
        self.cfg_dict=HALO_Devices_cls.cfg_dict
        self.device="smart"
        self.device_data_path=self.cfg_dict["device_data_path"]+self.device+"/"
    #---------------------------------------------------------------------
    #    OLD format from hamp_processing
    #    def open_data(self):
    #        file_name=self.cfg_dict["campaign"]+"_HALO_SMART_IMS_ql_"+\
    #                    self.cfg_dict["date"]+".nc"
    #        self.ds=xr.open_dataset(self.device_data_path+file_name)
    #        self.file=self.device_data_path+file_name
    #---------------------------------------------------------------------
    def open_ims_data(self):
        file_name=self.cfg_dict["campaign"]+"_HALO_SMART_IMS_ql_"+\
                    self.cfg_dict["date"]+".nc"
        self.ims_file=self.device_data_path+file_name
        self.ds=xr.open_dataset(self.ims_file)
    
    def open_irradiance_data(self):
        file_name="HALO-AC3_HALO_SMART_spectral_irradiance_Fdw_ql_"+\
                self.cfg_dict["date"]+"_"+self.cfg_dict["flight"]+".nc"
        self.irs_file=self.device_data_path+file_name
        self.ds=xr.open_dataset(self.irs_file)
        
#-----------------------------------------------------------------------------#
class SEA_ICE():
    def __init__(self,cfg_dict):
        self.cfg_dict=cfg_dict
    def open_sea_ice_ds(self):
        self.sea_ice_local_path=self.cfg_dict["device_data_path"]+"sea_ice/"
        if not os.path.exists(self.sea_ice_local_path):
            os.makedirs(self.sea_ice_local_path)
        
        sea_ice_date=str(self.cfg_dict["date"])
        try:
            # prinicipially this should work for everyone using the ac3airborne 
            # module
            import ac3airborne
            from ac3airborne.tools import get_amsr2_seaice
            ds_sea_ice=get_amsr2_seaice.get_amsr2_seaice("")
        except: # if not files have to be downloaded manually from :
                # https://atmos.meteo.uni-koeln.de/~mech/amsr2_seaice_concentration/halo-ac3/
                # and moved to the according HALO-(AC)3 Sea ice local path
                sea_ice_file_start="asi-AMSR2-n6250-"
                sea_ice_file_end="-v5.4.nc"
                
                self.sea_ice_file=sea_ice_file_start+sea_ice_date+\
                                        sea_ice_file_end
                self.ds=xr.open_dataset(self.sea_ice_local_path+\
                                          self.sea_ice_file)
                
                

class POLAR_Devices():
    def __init__(self,cfg_dict,
                 major_path=""):
        self.cfg_dict=cfg_dict
        self.campaign_name=self.cfg_dict["campaign"]
        if not major_path=="":
            self.major_data_path=major_path
        else:
            self.major_data_path=os.getcwd()+\
                "/Flight_Data/"+cfg_dict["campaign"]+"/"
        self.polar_joint_devices=["gps_ins"]
        self.avail_halo_devices={}
        avail_list=[False]*len(self.polar_joint_devices)
        self.avail_devices=dict(zip(self.polar_joint_devices,avail_list))
    
class GPS_INS(POLAR_Devices):
    def __init__(self,POLAR_Devices):
        self.cfg_dict=POLAR_Devices.cfg_dict
        self.major_data_path=POLAR_Devices.major_data_path
        self.campaign_name=POLAR_Devices.campaign_name
        #campaign_name
    def open_aircraft_gps_position(self,single_date=None,
                                   used_polar_aircraft="P5"):
        
        gps_data_path=self.major_data_path+\
                        used_polar_aircraft.lower()+"_gps_ins/"
        if single_date==None:
            polar_gps_files=glob.glob(gps_data_path+"*.nc")
            if used_polar_aircraft=="P5":
                self.P5_GPS={}
                for file in polar_gps_files:
                    date=file[-16:-8]
                    self.P5_GPS[date]=xr.open_dataset(file)
            elif used_polar_aircraft=="P6":
                self.P6_GPS={}
                for file in polar_gps_files:
                    date=file[-16:-8]
                    self.P6_GPS[date]=xr.open_dataset(file)
            else:
                raise Exception("You have chosen an aircraft",
                                " that is not P5 or P6 polar aircraft.")
                    #p5_xr.openmfg_dataset(gps_data_path+"*.nc")

class POLAR5(POLAR_Devices):
    def __init__(self,cfg_dict):
        print("This is the P5 Aircraft class")
        
"""
This is based on HD but class from AW will be used    
class HAMP(HALO_Devices):
    def __init__(self,HALO_Devices_cls):
        #super(HALO_Devices,self).__init__(cfg_dict)
        self.cfg_dict=HALO_Devices_cls.cfg_dict
        self.campaign_name=HALO_Devices_cls.campaign_name
        self.major_data_path=HALO_Devices_cls.major_data_path
        self.name="HAMP"
        self.long_name="HALO Microwave Package"
        self.device_type="radiometer"
        self.raw_hamp_path=self.major_data_path
        self.unified_hamp_path=self.major_data_path+"all_nc/"
        self.radiometer_vars=["KV","11990","183"]
    def open_raw_hamp_data(self,date):
        tb_data_dict={}
        self.raw_hamp_tb_dict={}
        # Each hamp module is stored in separate directory, so loop over. 
        for var in self.radiometer_vars:
        
            # List files from specified date
            radiometer_files=glob.glob(self.major_data_path+\
                                   "radiometer/"+var+"/*"+date+"*")
            # Display var name
            print("Open Radiometer Channel: ",var)
        
            # If no files were found, it's probably because the date 
            # is written in yymmdd format in file names instead of yyyymmdd
            if len(radiometer_files)==0:
                #flight_backup=flight    
                date=date[2::]
                radiometer_files=glob.glob(self.major_data_path+\
                                   "radiometer/"+var+"/*"+date+"*")
            
            # Open issue --> if flight reaches onto second UTC date
            # Check if flight ended after 00z
            #uni_time_str=uni_time.date[-1].strftime('%Y%m%d')[2::]
            #if not uni_time_str==flight:
            #    file_name_2=glob.glob(cfg_dict["device_data_path"]+\
            #                       "radiometer/"+var+"/*"+uni_time_str+"*")
            #    radiometer_files.append(file_name_2)
            #flight=flight_backup
        
            # If radiometer files for day exists    
            if not len(radiometer_files)==0:
                # Loop all files from day which should work either way: 
                # if it is only one file and if it contains several files
                for file in radiometer_files:
                    radiometer_ds=xr.open_dataset(file)
                    tb_data_dict[file]=pd.DataFrame(
                        data=np.array(radiometer_ds["TBs"][:].astype(float)),
                        index=pd.DatetimeIndex(np.array(radiometer_ds.time[:])),
                        columns=np.array(radiometer_ds["frequencies"][:]).\
                            astype(float).round(2).astype(str))
                if len(radiometer_files)>1:
                    self.raw_hamp_tb_dict[var]=pd.concat(*tb_data_dict)
                else:
                    self.raw_hamp_tb_dict[var]=tb_data_dict[file]
    def open_processed_hamp_data(self,cfg_dict,
                                 open_calibrated=False,
                                 newest_version=True):
       import Campaign_netCDF
       nc_path=cfg_dict["device_data_path"]+"all_nc/radiometer_"+\
       str([*cfg_dict["Flight_Dates_used"]][0])
       #sys.exit()
       if newest_version:       
          data_file=Campaign_netCDF.CPGN_netCDF.identify_newest_version(
              nc_path,for_calibrated_file=open_calibrated)
       if not open_calibrated:
           self.processed_hamp_ds=xr.open_dataset(data_file)
       else:
            self.calib_processed_hamp_ds=xr.open_dataset(data_file)
            
    def access_HAMP_TB_calibration_coeffs(self):
       self.calibration_path=self.major_data_path+"calibration_coeffs/"
       self.calibration_file="clear_sky_sonde_comparison_ALL_J3v0.9.2_radiometer_daily.nc"
       self.tb_calib_coeff_ds=xr.open_dataset(self.calibration_path+\
                                         self.calibration_file)
      
    def get_HAMP_TB_calibration_coeffs_of_flight(self):
        self.access_HAMP_TB_calibration_coeffs()
        date=str([*self.cfg_dict["Flight_Dates_used"]][0])
        self.flight_tb_offset_coeff_ds=self.tb_calib_coeff_ds[\
                                        "offset"].sel({"date":date})
        self.flight_tb_slope_coeff_ds=self.tb_calib_coeff_ds[\
                                        "slope"].sel({"date":date})
"""

"""
date="20200205"
flight="RF09"
prcs_cfg_dict={"campaign":"EUREC4A",
                "campaign_path":"C://Users/u300737/Desktop/PhD_UHH_WIMI/Work/GIT_Repository/hamp_processing_py/hamp_processing_python/Flight_Data/EUREC4A/",
              "Flight_Dates_used":{date:flight},
              "device_data_path":"C://Users/u300737/Desktop/PhD_UHH_WIMI/Work/GIT_Repository/hamp_processing_py/hamp_processing_python/Flight_Data/EUREC4A/"}
HALO_Devices_cls=HALO_Devices(prcs_cfg_dict)
#HAMP_cls=HAMP(HALO_Devices_cls)
Radar_cls=HALO_Radar(HALO_Devices_cls)
Radar_cls.open_attitude_corrected_data(flight, date)

#Radar_cls.open_raw_radar_data(flight, date)
# #HAMP_cls.open_processed_hamp_data(prcs_cfg_dict)
# #HAMP_cls.open_processed_hamp_data(prcs_cfg_dict,open_calibrated=True)#
#Radar_cls.open_processed_radar_data(reflectivity_is_calibrated=False)
#Radar_cls.open_processed_radar_data(reflectivity_is_calibrated=True)
# calib_ds=Radar_cls.calib_processed_radar_ds
# uncalib_ds=Radar_cls.processed_radar_ds
# calib_ds=HAMP_cls.calib_processed_hamp_ds
# from Data_Plotter import Quicklook_Plotter
import Data_Plotter
Quick_Plotter=Data_Plotter.Quicklook_Plotter(prcs_cfg_dict)
Radar_Quicklook=Data_Plotter.Radar_Quicklook(prcs_cfg_dict)
#Radar_Quicklook.processed_
Radar_Quicklook.plot_radar_quicklook(Radar_cls.attcorr_radar_ds,
                                     flight_report=True)
#plot_raw_radar_quicklook(Radar_cls.raw_radar_ds)

#radar_ds=Radar_cls.processed_radar_ds
#calib_radar_ds=
#Radar_Quicklook.processed_radar=Radar_cls.calib_processed_radar_ds
#print("Open radar mask")
#processed_radar=Data_Plotter.replace_fill_and_missing_values_to_nan(
#    Radar_Quicklook.processed_radar,["radar_flag"])
#processed_radar_flag=np.array(processed_radar["radar_flag"].copy()[:])
#import matplotlib.pyplot as plt
#fig=plt.figure(figsize=(12,8))
#ax1=fig.add_subplot(111)
#ax1.pcolor(processed_radar_flag.T)

#Radar_Quicklook.processed_radar_quicklook(hourly=np.nan,is_calibrated=True,with_masks=False)
#radar_mask=pd.DataFrame(data=np.array(
#    Radar_cls.calib_processed_radar_ds["radar_flag"][:]).astype(int),
#    index=pd.DatetimeIndex(np.array(
#    Radar_cls.calib_processed_radar_ds.time[:])),
#        columns=np.array(
#    Radar_cls.calib_processed_radar_ds["height"][:]))
#radar_mask[radar_mask==-888]=np.nan
             
#Plotting
        
#C1=ax1.pcolor(radar_mask.index,np.array(radar_mask.columns[:]),
#              radar_mask.values.T)
        
#ax1.pcolor(radar_mask.values[:].T)
#Radar_Quicklook.processed_radar_quicklook(hourly=np.nan,
#                                          is_calibrated=True)
# Radiometer_Quicklook=Data_Plotter.Radiometer_Quicklook(prcs_cfg_dict)
# #Radiometer_Quicklook.radiometer_tb_dict=HAMP_cls.raw_hamp_tb_dict
# Radiometer_Quicklook.plot_radiometer_TB_calibration_comparison()
#Radar_Quicklook
#uncalib_ds.sortby("uniRadiometer_freq")
#calib_coeff_da=calib_coeff_da.loc[{"frequency":uncalib_ds.uniRadiometer_freq}]
#calib_ds=uncalib_ds.assign(TB=uncalib_ds["TB"]*calib_coeff_da["slope"].values+\
#                        calib_coeff_da["offset"].values)

        
"""
#    print("Freq",freq.data,"GHz cannot be calibrated")
#import matplotlib.pyplot as plt
#import pandas as pd
#test=pd.DataFrame(columns=["calib","uncalib"])
#test["uncalib"]=uncalib_ds["TB"].sel({"uniRadiometer_freq":54.94}).data
#test["calib"]=calib_ds["TB"].sel({"uniRadiometer_freq":54.94}).data
#calib_ds["TB"]=calib_ds["TB"]+20
#plt.plot(uncalib_ds["TB"].sel({"uniRadiometer_freq":22.24}))
#plt.plot(calib_ds["TB"].sel({"uniRadiometer_freq":22.24}))

"""
#plt.plot(calib_ds["TB"].sel({"uniRadiometer_freq":54.94}))
#plt.ylim([254,256])
#HAMP_cls.open_processed_hamp_data(prcs_cfg_dict,open_calibrated=True)

#print(HAMP_cls.processed_hamp_ds)
#print(HAMP_cls.calib_processed_hamp_ds)
#Radar_cls=HALO_Radar(HALO_Devices_cls)
#HAMP_cls.open_raw_hamp_data(date)
#print(HAMP_cls.tb_calib_coeff_ds)
#HAMP_cls.plot_HAMP_TB_calibration_coeffs_of_flight()
#raw_radar_ds=Radar_cls.open_raw_radar_data(flight, date)
# Open exemplaric file if several exist for one flight
#raw_radar_ds = xr.open_dataset(radar_fnames[0])
#attrs= {'units': str(raw_radar_ds.time.long_name)}
#temporary_ds=xr.Dataset({'time':('time',np.array(raw_radar_ds.time[:]),attrs)})
#temporary_ds=xr.decode_cf(temporary_ds)   #

#dBZg = np.array(10*np.emath.log10(raw_radar_ds['Zg'][:]))
#dBZg = dBZg.astype(np.float32)
#% Only keep real part of array (imaginary numbers were created when
#% taking log10 of -Inf: log10(-Inf) = Inf +      1.36437635384184i)
#dBZg = np.real(dBZg)
#% And convert positive infinity back to negative infinity
#dBZg = np.where(dBZg==np.inf,-np.inf,dBZg)
#raw_radar_ds["dBZg"]=xr.DataArray(data=dBZg.T,dims=raw_radar_ds.dims)
#raw_radar_ds["time"]=temporary_ds["time"]
#raw_radar_ds
#"""
