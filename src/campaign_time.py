# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 11:09:27 2021

@author: u300737
"""
import numpy as np
import pandas as pd

class Campaign_Time():
    def __init__(self,campaign_name,date):
        self.campaign=campaign_name
        self.date=date
        self.bahamas_tOffsets=   {}      # time offset in seconds
        # These values have been estimated by eye using assess_radar_data.m
        # Numbers represent indices in raw measurement data and are not yet
        # converted to times
        
        # Note individual indices or intervals in the form ii:jj
        

        #%% Errors

        self.bahamas_tOffsets["NARVAL-I"]={'20131210':-2,
                                           '20131211':-2,
                                           '20131212':-2,
                                           '20131214':-2,
                                           '20131215':-1,
                                           '20131216':-1,
                                           '20131219': 0,
                                           '20131220':-2}
        
        self.bahamas_tOffsets["NARVAL-North"]={
                                               '20140107':-2,
                                               '20140109':-2,
                                               '20140112':-1,
                                               '20140118':-19,
                                               '20140120':-2,
                                               '20140121':-2,
                                               '20140122':-1}
        
        self.bahamas_tOffsets["NARVAL-II"]  = {'20160808': 0,
                                               '20160810': 0,
                                               '20160812': 0,
                                               '20160815': 0,
                                               '20160817': 0,
                                               '20160819': 1,
                                               '20160822': 0}
        
        self.bahamas_tOffsets["NAWDEX"]     = {'20160917': 0,
                                               '20160921': 0,
                                               '20160923': 0,
                                               '20160926': 0,
                                               '20160927': 1,
                                               '20161001': 0,
                                               '20161006': 0,
                                               '20161009': 1,
                                               '20161010': 0,
                                               '20161013': 0,
                                               '20161014': 0,
                                               '20161015': 1,
                                               '20161018': 0,
                                               }
        
        self.bahamas_tOffsets["EUREC4A"]    = {'20200119': 0,
                                               '20200124': 0,
                                               '20200126': 0,
                                               '20200128': 0,
                                               '20200130': 0,
                                               '20200131': 0,
                                               '20200202': 0,
                                               '20200205': 0,
                                               '20200207': 0,
                                               '20200209': 0,
                                               '20200211': 0,
                                               '20200213': 0,
                                               '20200215': 0,
                                               '20200218': 0,}
        self.bahamas_tOffsets["HALO_AC3"]  = {"20200225" : 0,
                                              "20220311":0,
                                              "20220312":0,
                                              "20220313":0}
          
    def look_up_bahamas_time_offs(self):
        #Get index of date
        try:
            index = self.bahamas_tOffsets[self.campaign][self.date]
        except:
            index=[]
        #If date is not in list, assume offset of 0 seconds, else, copy to output
        #% variable
        if not type(index)==int:
            timeOffset = 0
        else:
            timeOffset = index
        return timeOffset

    def get_flight_campaign_flight_dates(self,cfg_dict):
        """
        
        
        Parameters
        ----------
        cfg_dict : dict
            Configuration dictionary containing all information 
            for processing/plotting.

        Returns
        -------
        cfg_dict : dict
            same configuration file now with key entry for Flight_Dates listing all
            flights of the campaign.

        """
        if cfg_dict["campaign"]=="EUREC4A":
            cfg_dict["Flight_Dates"]={"RF01":"20200119","RF02":"20200122",
                             "RF03":"20200124","RF04":"20200126",
                             "RF05":"20200128","RF06":"20200130",
                             "RF07":"20200131","RF08":"20200202",
                             "RF09":"20200205","RF10":"20200207",
                             "RF11":"20200209","RF12":"20200211",
                             "RF13":"20200213","RF14":"20200215",
                             "RF15":"20200218"}
        
        elif cfg_dict["campaign"]=="NAWDEX":
            cfg_dict["Flight_Dates"]={"RF01":"20160917",
                                      "RF02":"20160921",
                                      "RF03":"20160923",
                                      "RF04":"20160926",
                                      "RF05":"20160927",
                                      "RF06":"20161001",
                                      "RF07":"20161006",
                                      "RF08":"20161009",
                                      "RF09":"20161010",
                                      "RF10":"20161013",
                                      "RF11":"20161014",
                                      "RF12":"20161015",
                                      "RF13":"20161018"}
        
        elif cfg_dict["campaign"]=="HALO_AC3":
            cfg_dict["Flight_Dates"]={"RF00":"20220225",
                                      "RF01":"20220311",
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
                                      "RF18":"20220412",
                                      "RF19":"20220414"}
        else:
            raise Exception("The given campaign ",cfg_dict["campaign"],
                            "is either not defined or not yet included")
        return cfg_dict
    
    def specify_dates_to_use(self,cfg_dict):
        """
        This function specify the flights of interest in between the start 
        and end date. 
    
    
        Parameters
        ----------
        cfg_dict
        
        Returns
        -------
        interested_flights.

        """
        start=cfg_dict["t1"]
        end=cfg_dict["t2"]
        cfg_dict=self.get_flight_campaign_flight_dates(cfg_dict)
        Flight_Dates=cfg_dict["Flight_Dates"]
        index_arg=list(Flight_Dates.keys())
        data_arg=np.array(list(Flight_Dates.values())).astype(int)
        flight_series=pd.Series(data=data_arg,
                                index=np.array(index_arg))
        print(flight_series)
        flights_used=flight_series.between(int(start),int(end),inclusive=True)
        interested_flights=flight_series[flights_used]
    
        return interested_flights


### Additional Functions
def sdn_2_dt(sdn):
        time_dt = pd.to_datetime(np.array(sdn)-719529, unit='D')
        return time_dt

    