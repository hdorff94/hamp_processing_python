# -*- coding: utf-8 -*-
"""
This is the major routine to run the hamp processing

Created on Fri Jan 29 08:09:31 2021

@author: u300737
"""
def main(campaign_name,rf,
         instruments_to_unify=[#"bahamas",
                               "radar",
                               #"radiometer",
                               #"dropsondes", 
                               ] # default bahamas, dropsondes, radar, radiometer.
         ):
    import os
    import glob
    import sys
    import init_paths
    # Define paths
    working_path=init_paths.main()
    #path_name="C:\\Users\\u300737\\Desktop\\Desktop_alter_Rechner\\"+\
    #                "PhD_UHH_WIMI\\"
    actual_working_path=os.getcwd()
    airborne_data_importer_path=actual_working_path+"/../../Work/GIT_Repository/"+\
                                "hamp_processing_py/"+\
                                    "hamp_processing_python/" 
    # This is also the major path where your data will be stored
    airborne_processing_module_path=actual_working_path+"/src/"
    airborne_plotting_module_path=actual_working_path+"/plotting/"

    # Insert paths to import modules
    sys.path.insert(1,airborne_processing_module_path)
    sys.path.insert(2,airborne_plotting_module_path)
    sys.path.insert(3,airborne_data_importer_path)
    
    import config_handler
    import campaign_time
    
    import performance
    
    import numpy as np
    import pandas as pd
    import xarray as xr
    
    import radar_attitude
    import radar_masks
    import unified_grid as unigrid
    
    try:
        import Flight_Campaign as Campaign
    except:
        print("Module Flight Campaign is not listed in the path",
              "Flights need to be defined manually.")

    Flight_Dates={}
    Flight_Dates["EUREC4A"]={"RF01":"20200119","RF02":"20200122",
                                 "RF03":"20200124","RF04":"20200126",
                                 "RF05":"20200128","RF06":"20200130",
                                 "RF07":"20200131","RF08":"20200202",
                                 "RF09":"20200205","RF10":"20200207",
                                 "RF11":"20200209","RF12":"20200211",
                                 "RF13":"20200213","RF14":"20200215",
                                 "RF15":"20200218"}
        
    Flight_Dates["HALO_AC3"]={"RF00":"20220225",
                              "RF01":"20220311", # this is the transfer flight
                              "RF02":"20220312",
                              "RF03":"20220313",
                              "RF04":"20220314",
                              "RF05":"20220315",
                              "RF06":"20220316",
                              "RF07":"20220320",
                              "RF08":"20220321",
                              "RF09":"20220328",
                              "RF10":"20220329",
                              "RF11":"20220330",
                              "RF12":"20220401",
                              "RF13":"20220404",
                              "RF14":"20220407",
                              "RF15":"20220408",
                              "RF16":"20220410",
                              "RF17":"20220411",
                              "RF18":"20220412"
                              }
        
    #%% allocate configurator
    cfg=config_handler.Configuration(major_path=airborne_data_importer_path)
    processing_cfg_name="unified_grid_cfg"    
    major_cfg_name="major_cfg"
    
    #Output file name prefix
    # The usual file name will follow the format: 
    # <instrument>_<date>_v<version-number>.nc
    # An additional file name prefix can be specified here (e.g. for EUREC4A),
    # if no prefix is necessary, set to empty string ('')
    # filenameprefix = 'EUREC4A_HALO_';
    # filenameprefix = ''
    
    campaign=campaign_name
    filenameprefix = campaign+'_HALO_'
    #%% comments for data files
    # Specify comment to be included into data files
    comment = 'Preliminary data! Uncalibrated Data.'+\
                ' Only use for preliminary work!'
    # Specify contact information
    contact = 'henning.dorff@uni-hamburg.de'
    
    #%% load configurations
    configurations=cfg.return_default_config_dict(major_cfg_name,
                                    processing_cfg_name,
                                    campaign,comment=comment,
                                    contact=contact)
    
    #%% specify time frame for data conversion
    flight=rf
    # % Start date
    start_date =Flight_Dates[campaign][flight]  
    # % End date
    end_date = Flight_Dates[campaign][flight]
    
    #%% define processing steps
    #  Set version information and processing steps in config file cfg.
    #  Missing value, set value for missing value (pixels with no measured signal). 
    #  This should be different from NaN, since NaN is used as fill value 
    #  (pixels where no measurements were conducted)#
    #  Set threshold for altitude to discard radiometer data
    #  Set threshold for roll angle to discard radiometer data
    cfg.add_entries_to_config_object(processing_cfg_name,
                            {"t1":start_date,"t2":end_date,
                             "date":start_date,"flight_date_used":start_date,
                             "correct_attitude":False,         #0.1 default False
                                                              # as otherwise it
                                                              # is recalculated 
                                                              # every time 
                                                              # although already
                                                              # existent
                             "unify_Grid":True,               #0.2 default True
                             
                             "fill_gaps":True,                # 0.3
                             "remove_clutter":True,           # 0.4 default True
                             "remove_side_lobes":True,        # 0.5 default True
                             "remove_radiometer_errors":True, # default True
                             "add_radarmask":True,            # 0.6 default True
                             "add_radar_mask_values":True,    # if false mask 
                                                                       # is not added to the data
                                      
                             "version":2,
                             "subversion":7,
                             "quicklooks":False,               # default True
                             "missing_value":-888,
                             "fill_value": np.nan,
                             "altitude_threshold":4800,
                             "roll_threshold":5})
    
    #%% define instruments to unify
    cfg.add_entries_to_config_object(processing_cfg_name,
                        {"instruments_to_unify":instruments_to_unify})
    
    #%% specify processing steps (masking criteria,calibration)
    cfg.add_entries_to_config_object(processing_cfg_name,
                                     {"land_mask":1,
                                      "noise_mask":1,
                                      "calibration_mask":1,
                                      "surface_mask":1,
                                      "seasurface_mask":1,
                                      "num_RangeGates_for_sfc":4})
    
    cfg.add_entries_to_config_object(processing_cfg_name,
                                     {"calibrate_radiometer":True, # 1.x
                                      "calibrate_radar":True})     # 1.x
    
    processing_config_file=cfg.load_config_file(processing_cfg_name)
    
    processing_config_file["Input"]["data_path"]=\
                                processing_config_file["Input"][\
                                    "campaign_path"]+"Flight_Data/"
    processing_config_file["Input"]["device_data_path"]=\
                                processing_config_file["Input"][\
                                    "data_path"]+campaign+"/"
                                    
    prcs_cfg_dict=dict(processing_config_file["Input"])    
    #   Specify the relevant flight dates for the period of start and end date
    #   given above
    Campaign_Time_cls=campaign_time.Campaign_Time(campaign,start_date)
    flightdates_use = Campaign_Time_cls.specify_dates_to_use(prcs_cfg_dict);
    
    # Used for later processing
    prcs_cfg_dict["campaign"]=campaign#[*Flight_Dates][0]
    prcs_cfg_dict["Flight_Dates"]=Flight_Dates
    prcs_cfg_dict["Flight_Dates_used"]=flightdates_use
    
    # % Check structure of folders for data files
    #still missing, will be based on checkfolderstructure(getPathPrefix,flightdates_use)
    #%% raw data plotting
    from measurement_instruments_ql import HALO_Devices, RADAR, HAMP
    date=start_date#flightdates_use.values[0]
    
    HALO_Devices_cls=HALO_Devices(prcs_cfg_dict)
    Radar_cls=RADAR(HALO_Devices_cls)
    HAMP_cls=HAMP(HALO_Devices_cls)
    # Open raw data
    Radar_cls.open_raw_radar_data(flight,date)
    raw_radar_ds=Radar_cls.raw_radar_ds            
    
    import halodataplot as halo_data_plotter
    
    Quick_Plotter=halo_data_plotter.Quicklook_Plotter(prcs_cfg_dict)
    Radiometer_Quicklook=halo_data_plotter.Radiometer_Quicklook(prcs_cfg_dict)
    Radar_Quicklook=halo_data_plotter.Radar_Quicklook(prcs_cfg_dict) 

    perform_raw_quicklooks=False
    if perform_raw_quicklooks:
        Radar_Quicklook.plot_raw_radar_quicklook(raw_radar_ds)
        # CFAD plotting requires radar reflectivity as dataframe and 
        # then routine plot_single_radar_cfad also calculates the cfad 
        # by status method "calc_radar_cfad" in Data_Plotter
        raw_radar_reflectivity=pd.DataFrame(
                                    data=np.array(raw_radar_ds["dBZg"].T[:]),
                                    index=np.array(raw_radar_ds["time"]),
                                    columns=np.array(raw_radar_ds["range"][:]))
        #Radar_Quicklook.plot_radar_clutter_comparison(clutter_removal_version="0.2")
        #Radar_Quicklook.plot_single_radar_cfad(raw_radar_reflectivity)
       
    #%% processing
    print("=========================== Processing ===============================")
    if "radar" in instruments_to_unify:
            if performance.str2bool(prcs_cfg_dict["correct_attitude"]):
                # Correct radar data for aircraft attitude
                print("Correct the radar attitude")
                radar_attitude.run_att_correction(flightdates_use, prcs_cfg_dict)
    
            if not performance.str2bool(prcs_cfg_dict["correct_attitude"]):
                prcs_cfg_dict["radar_outDir"]=\
                    prcs_cfg_dict["device_data_path"]+"radar_mira/"
                # Even if explicitly desired to not attitude correct radarfiles,
                # it is checked here, whether the corrected-file already exists

                for flight in flightdates_use:
                    if len(glob.glob(prcs_cfg_dict["radar_outDir"]+"*"+\
                                     str(flight)+"*.nc"))>=1:
                        print("Flight is already attitude-corrected,",
                              " so skip this step")
            
                    else:
                        new_flightdates_use=pd.Series(flight,
                                  index=flightdates_use[\
                                        flightdates_use==int(flight)].index)
                        radar_attitude.run_att_correction(
                                new_flightdates_use,prcs_cfg_dict)
    
            if performance.str2bool(prcs_cfg_dict["add_radarmask"]):
                # Create radar info mask
                radar_masks.run_make_masks(flightdates_use, prcs_cfg_dict)
    
    if performance.str2bool(prcs_cfg_dict["unify_grid"]):
            # Unify data from bahamas, dropsondes,
            # radar, radiometer onto common grid
            unigrid.run_unify_grid(flightdates_use,prcs_cfg_dict)
    
    return None

if __name__=="__main__":
   campaign="HALO_AC3"
   research_flights_to_process=[
                               "RF01",
                               #"RF02",
                               #"RF03",
                               #"RF04",
                               #"RF05",
                               #"RF06",
                               #"RF07",
                               #"RF08",#,
                               #"RF09",
                               #"RF10",#
                               #"RF11",
                               #"RF12",
                               #"RF13",
                               #"RF14",
                               #"RF15",
                               #"RF16",
                               #"RF17",
                               #"RF18"
                                ]
   
   for research_flight in research_flights_to_process:
       main(campaign,research_flight)