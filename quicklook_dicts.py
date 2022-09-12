# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:30:29 2022

@author: u300737
"""

def get_prcs_cfg_dict(flight,date,campaign,campaign_path,
                          additional_entries_dict={}):
        prcs_cfg_dict={
              "flight":flight,
              "date":date,
              "t1":date,
              "t2":date,
              "campaign":campaign,
              "campaign_path":campaign_path,
              "Flight_Dates_used":{date:flight},
              "device_data_path":campaign_path,
              "flight_date_used":date,
              # Needed for HAMP and Radar
              "correct_attitude":True,          # default False
              "version":0,
              "subversion":8,
              "missing_value":-888,
              "altitude_threshold":4800,
              "roll_threshold":5}
        if len(additional_entries_dict.keys())>0:
            for key,value in additional_entries_dict.items():
                prcs_cfg_dict[key]=value
        return prcs_cfg_dict
def get_data_handling_attr_dicts(entries_to_change={}):
        datasets={}
        datasets["bacardi"]=["bacardi_dict"]
        datasets["bahamas"]=[""]
        datasets["radar"]=["attcorr_radar_ds"]
        datasets["smart"]=["ds"]
        if "datasets" in entries_to_change.keys():
            for key,value in entries_to_change["datasets"]:
                datasets[key]=value
        data_reader={}
        data_reader["bacardi"]="open_raw_quicklook_data"
        data_reader["bahamas"]="open_bahamas_data"
        data_reader["hamp"]="open_hamp_raw_data"
        data_reader["radar"]="open_attitude_corrected_data"
        data_reader["smart"]="open_irradiance_data"
        if "data_reader" in entries_to_change.keys():
            for key,value in entries_to_change["data_reader"]:
                data_reader[key]=value
        
        return datasets, data_reader
    
def get_plotting_handling_attrs_dict(entries_to_change={}):
        plot_handler={}
        plot_handler["bacardi"]=["plot_bacardi_quicklook"]
        plot_handler["bahamas"]=["plot_bahamas_movement_quicklook",
                             "plot_flight_map_with_sea_ice_conc",
                             "plot_bahamas_meteo_quicklook"]
        plot_handler["hamp"]=["plot_HAMP_TB_quicklook"]
        plot_handler["radar"]=["plot_radar_quicklook","plot_single_radar_cfad"]
        plot_handler["smart"]=["plot_smart_irradiance"]
        if "plot_handler" in entries_to_change.keys():
            for key,value in entries_to_change["plot_handler"]:
                plot_handler[key]=value
        
        plot_cls_args={"bacardi":[],
                   "bahamas":["data_cls.bahamas_ds"],
                   "hamp":[],
                   "radar":[],
                   "smart":[]}
        if "plot_cls_args" in entries_to_change.keys():
            for key,value in entries_to_change["plot_cls_args"]:
                plot_cls_args[key]=value
    
        plot_fcts_args={"bacardi":[["bacardi_dict",],],
                    "bahamas":[[],[],["HALO_Devices_cls",True]],
                    "radar":[["attcorr_radar_ds",True,False, True],[]],
                    "hamp":[["data_cls","HALO_Devices_cls"]],
                    "smart":[["ds"]]}
        if "plot_fct_args" in entries_to_change.keys():
            for key,value in entries_to_change["plot_fct_args"]:
                plot_fcts_args[key]=value

        return plot_handler, plot_cls_args,plot_fcts_args
