# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 11:25:15 2021

@author: u300737
"""
import glob
import os 
import sys

import numpy as np
import pandas as pd
import xarray as xr

import pickle

class Radiometer_Errors():
    def __init__(self,cfg_dict,create_figures=False,
                 calc=True,overview=True,check=False):
        self.radiometer_modules=["183","11990","KV"]
        self.cfg_dict=cfg_dict
        #%% Set parameters
        #Set if figures should be produced
        # figures = True  if you want to look through figures from 
        # individual flights and note error and saw tooth occurrences.
        # Note error and saw tooth interval indices in file
        # 'radiometer_errors_lookup.npy
        self.create_figures = create_figures
        
        #Set if error time steps should be calculated from indices
        self.calc = calc
        
        # Set if overview figure should be produced
        # if you want to produce overview figure
        self.overview = overview
        
        # Set if you want to check the results of error removal, i.e 
        # if you want to check you identified errors
        self.check = check
        # Set campaign to analyse, string with campaign name
        self.campaign = self.cfg_dict["campaign"]
    
    def radiometer_single_channel_errors_lookup(self):
        # These values have been estimated by eye using main() of 
        # Error_Identification.py. Numbers represent indices in raw measurement
        # data and are not yet converted to times
        # Note individual indices or intervals in the form [ii jj]
                    # key frequency,     date as key for sub_dict,  indices
        single_channel_errors = {}
        #G
        single_channel_errors['23.84']={'20200213':[[24154,24170],
                                                    [50284,50309],
                                                    [37134,37154],
                                                    [111297,111303],
                                                    [111250,111260],
                                                    [125644,125659]],
                                        '20200122':[[115702,115759],
                                                    [119739,119779]],
                                        '20200211':[[22206,22239]],
                                        }
        
        
        single_channel_errors['90.0']={'20200213':[[37322,37336],
                                                   [50495,50507]],
                                       '20200207':[[94441,94448]]}
        #KV
        single_channel_errors['190.81']={'20200130':[[44437,63006]],
                                         '20200131':[[0,78999]],
                                         '20200202':[[0,49115]],
                                         '20200205':[[0,53433]]}
        
        self.single_channel_errors=single_channel_errors
    
    def radiometer_errors_lookup(self):
        # These values have been estimated by eye using assess_radiometer_data.m
        # Numbers represent indices in raw measurement data and are not yet
        # converted to times
        # Note individual indices or intervals in the form [ii jj]

        #%% Errors
        errors={}
        #183
        errors['183'] = {'20131210':[],
                       '20131211':[],
                       '20131212':[],
                       '20131214':[],
                       '20131215':[],
                       '20131216':[],
                       '20131219':[],
                       '20131220':[],
                       '20140107':[],
                       '20140109':[],
                       '20140112':[],
                       '20140118':[],
                       '20140120':[],
                       '20140121':[],
                       '20140122':[[3062,3686], 4470],
                       # '20160808',{[]};...
                       # '20160810',{[1 107]};...
                       # '20160812',{[1 15]};...
                       # '20160815',{[1 449]};...
                       # '20160817',{[1 6217]};...
                       # '20160819',{[1 9074],16096,[16545 16554],[18088 18091]};...
                       # '20160822',{[1 12797],20375};...
                       # '20160824',{[1 10849]};...
                       # '20160826',{[1 12192],[18193 18209],[19740 19749]};...
                       # '20160830',{[1 9548]};...
                       # '20160917',{[]};...
                       # '20160921',{[]};...
                       # '20160923',{[]};...
                       # '20160926',{[1 39]};...
                       # '20160927',{[1 2]};...
                       # '20161001',{[1 315]};...
                       # '20161006',{[1 741],3626};...
                       # '20161009',{[]};...
                       # '20161010',{[1 1445]};...
                       # '20161013',{[]};...
                       # '20161014',{[]};...
                       # '20161015',{[]};...
                       # '20161018',{[]};...
                       '20200119':[[1,60]],
                       '20200122':[],
                       '20200124':[[1,3834],
                                   [97936,99937],
                                   [125260,127013],
                                   [127768,129441]],
                       '20200126':[[1,358],
                                   [3516,4325],
                                   [14720,16533],
                                   [34828,34870]],
                       '20200128':[],
                       '20200130':[],
                       '20200131':[[37226,38165],
                                   [50218,50979],
                                   [60604,62715],
                                   [78996,81075],
                                   [83514,84365]],
                       '20200202':[[1,5642]],
                       '20200205':[53434,56829],
                       '20200207':[[1,6673],
                                   [29821,30010],
                                   [30306,30859],
                                   [31501,31988],
                                   [84913,86333],
                                   [88232,88616]],
                       '20200209': [],
                       '20200211': [[26145,27147],
                                   [42141,43527],
                                   [44798,45865],
                                   [64214,64407],
                                   [73630,73965],
                                   [96760,98944]],
                       '20200213': [],
                       '20200215': [],
                       '20200218': [[76618,78863]],
                       # HALO-(AC)3
                       '20220311': [],
                       '20220312': [],
                       '20220313': [],
                       '20220314': [],
                       '20220315': [],
                       '20220316': [],
                       '20220320': [],
                       '20220321': []}
        
        #11990
        errors["11990"]={
                '20131210':[4693, 11103],#;... %, [1 2],[17215 17216]};...
                '20131211':[7227, 16263],
                # '20131212',{[]};...
                # '20131214',{1564, 17361};...
                # '20131215',{[1 2]};...
                # '20131216',{2944, 11701, 20465, 24855};... % , [13892 13893]
                # '20131219',{8375, 15154};... % [3835 3836], , [10644 10645]
                # '20131220',{13512, 15766};... % [11263 11264],
                # '20140107',{1579, 9846};... % [9379 9380],
                # '20140109',{4978, 9633, 18926};...
                # '20140112',{820};...
                # '20140118',{[]};...
                # '20140120',{4697};...
                # '20140121',{[]};...
                # '20140122',{11806};...
                # '20160808',{[]};...
                # '20160810',{[]};...
                # '20160812',{[]};...
                # '20160815',{4355};...
                # '20160817',{18700,23730};...
                # '20160819',{[]};...
                # '20160822',{7263};...
                # '20160824',{4123};...
                # '20160826',{[]};...
                # '20160830',{[]};...
                # '20160917',{[]};...
                # '20160921',{[]};...
                # '20160923',{[1 3]};...
                # '20160926',{[1 34]};...
                # '20160927',{[1 144]};...
                # '20161001',{[1 1979]};...
                # '20161006',{[1 93]};...
                # '20161009',{[]};...
                # '20161010',{[1 703]};...
                # '20161013',{[1 678],16132};...
                # '20161014',{[1 4513],6995};...
                # '20161015',{[1 927],9964,14945,17438};...
                # '20161018',{[]};...
                # '20200119', {[]};...
                
                # EUREC4A
                '20200122':[[124529,124538],[126453,126457]],
                '20200124':[],
                '20200126':[],
                '20200128':[],
                '20200130':[],
                '20200131':[],
                '20200202':[],
                '20200205':[],
                '20200207':[],
                '20200209':[],
                '20200211':[[26200,26210], [96085,96101]],
                '20200213':[],
                '20200215':[],
                '20200218':[],
                # HALO-(AC)3
                '20220311': [],
                '20220312': [],
                '20220313': [],
                '20220314': [],
                '20220315': [],
                '20220316': [],
                '20220320': [],
                '20220321': []}

        #%KV
        errors["KV"] = {'20131210':[5054, 6059, 7057, 8037],
                '20131211':[693, 2761, [3773,3776], 4797],
                #'20131212',{1, 4559};... % , [7532 7533]
                # '20131214',{[1 2], [718 835], [2214 2295], [2416 2424], 3713, 4798, 5874};...
                # '20131215',{4122};...
                # '20131216',{135, 2103};...
                # '20131219',{11296};...
                # '20131220',{[]};...
                # '20140107',{904};...
                # '20140109',{[]};...
                # '20140112',{319};...
                # '20140118',{[]};...
                # '20140120',{[]};...
                # '20140121',{[]};...
                # '20140122',{1236, 7170};...
                # '20160808',{9541,13912,22678};...
                # '20160810',{[1668 1712],1961,4136,6302,14977};...
                # '20160812',{[]};...
                # '20160815',{5562,12035,14202};...
                # '20160817',{2762,11429};...
                # '20160819',{8900,17611};...
                # '20160822',{6347,5837,12330,14483,16642};...
                # '20160824',{[]};...
                # '20160826',{[]};...
                # '20160830',{25135,25168};...
                # '20160917',{[]};...
                # '20160921',{[]};...
                # '20160923',{[]};...
                # '20160926',{[1 110]};...
                # '20160927',{[1 85]};...
                # '20161001',{[1 880]};...
                # '20161006',{[1 2],[2189 4625],10954};...
                # '20161009',{10017};...
                # '20161010',{[1 2959],5394,16311};...
                # '20161013',{[1 2441]};...
                # '20161014',{[1 1530]};...
                # '20161015',{[1 493]};...
                # '20161018',{[1 1658]};...
                # EUREC4A
                '20200119':[],
                '20200122':[[1,8893]],
                '20200124':[[1,16830]],
                '20200126':[[1,10820]],
                '20200128':[[1,11780]],
                '20200130':[[1,10490]],
                '20200131':[[1,6973]],
                '20200202':[],
                '20200205':[[1,8553]],
                '20200207':[],
                '20200209':[[1,14200], [137400,137600]],
                '20200211':[[1,7038], [92202,92213],[130000,130384]],
                '20200213':[[1,9065]],
                '20200215':[[1,12380]],
                '20200218':[[1,3868],[113185,128415]],
                # HALO-(AC)3
                '20220311': [],
                '20220312': [],
                '20220313': [],
                '20220314': [],
                '20220315': [],
                '20220316': [],
                '20220320': [],
                '20220321': []}
        
        #Assign errors to class
        self.errors=errors
        #%% Saw tooth
        sawtooth={}

        sawtooth['183'] = {
            # '20131210',{[]};...
            # '20131211',{[2345 16798]};...
            # '20131212',{[2042 23063]};...
            # '20131214',{[838 10379]};...
            # '20131215',{[4710 13202]};...
            # '20131216',{[5933 14153]};...
            # '20131219',{[]};...
            # '20131220',{[1760 10449]};...
            # '20140107',{[]};...
            # '20140109',{[]};...
            # '20140112',{[3019 3254]};...
            # '20140118',{[]};...
            # '20140120',{[]};...
            # '20140121',{[]};...
            # '20140122',{[]};...
            # '20160808',{[1 6450]};...
            # '20160810',{[108 7649]};...
            # '20160812',{[16 8309]};...
            # '20160815',{[450 5603]};...
            # '20160817',{[6218 12205]};...
            # '20160819',{[9075 15713]};...
            # '20160822',{[12798 17395]};...
            # '20160824',{[10850 17868]};...
            # '20160826',{[12193 17808]};...
            # '20160830',{[9549 14735]};...
            # '20160917',{[]};...
            # '20160921',{[]};...
            # '20160923',{[]};...
            # '20160926',{[]};...
            # '20160927',{[3 546]};...
            # '20161001',{[316 1029]};...
            # '20161006',{[742 3625]};...
            # '20161009',{[]};...
            # '20161010',{[1446 2887]};...
            # '20161013',{[]};...
            # '20161014',{[1728 3130]};...
            # '20161015',{[]};...
            # '20161018',{[]};...
            '20200119': [61,16230],
            '20200122':[],
            '20200124':[],
            '20200126':[[34791,39636]],
            '20200128':[],#};... % 34870 39497
            '20200130':[],
            '20200131':[],
            '20200202':[],
            '20200205':[],
            '20200207':[],
            '20200209':[],
            '20200211':[],
            '20200213':[[1,6165]],
            '20200215':[],
            '20200218':[[1,5680]],
            # HALO-(AC)3
            '20220311': [],
            '20220312': [],
            '20220313': [],
            '20220314': [],
            '20220315': [],
            '20220316': [],
            '20220320': [],
            '20220321': []}

        sawtooth['11990'] = {'20131210':[],
            '20131211':[],
            '20131212':[],
            '20131214':[],
            # '20131215',{[]};...
            # '20131216',{[]};...
            # '20131219',{[]};...
            # '20131220',{[]};...
            # '20140107',{[]};...
            # '20140109',{[]};...
            # '20140112',{[]};...
            # '20140118',{[]};...
            # '20140120',{[]};...
            # '20140121',{[]};...
            # '20140122',{[]};...
            # '20160808',{[]};...
            # '20160810',{[]};...
            # '20160812',{[]};...
            # '20160815',{[]};...
            # '20160817',{[]};...
            # '20160819',{[]};...
            # '20160822',{[]};...
            # '20160824',{[]};...
            # '20160826',{[]};...
            # '20160830',{[]};...
            # '20160917',{[]};...
            # '20160921',{[]};...
            # '20160923',{[]};...
            # '20160926',{[]};...
            # '20160927',{[]};...
            # '20161001',{[]};...
            # '20161006',{[]};...
            # '20161009',{[]};...
            # '20161010',{[]};...
            # '20161013',{[]};...
            # '20161014',{[]};...
            # '20161015',{[]};...
            # '20161018',{[]};...
            '20200119':[],
            '20200122':[],
            '20200124':[],
            '20200126':[],
            '20200128':[],
            '20200130':[],
            '20200131':[],
            '20200202':[],
            '20200205':[],
            '20200207':[],
            '20200209':[],
            '20200211':[],
            '20200213':[],
            '20200215':[],
            '20200218':[],
             # HALO-(AC)3
            '20220311': [],
            '20220312': [],
            '20220313': [],
            '20220314': [],
            '20220315': [],
            '20220316': [],
            '20220320': [],
            '20220321': []}

        sawtooth['KV'] = {'20131210':[],
            '20131211':[],
            '20131212':[],
            # '20131214',{[]};...
            # '20131215',{[]};...
            # '20131216',{[]};...
            # '20131219',{[]};...
            # '20131220',{[]};...
            # '20140107',{[]};...
            # '20140109',{[]};...
            # '20140112',{[]};...
            # '20140118',{[]};...
            # '20140120',{[]};...
            # '20140121',{[]};...
            # '20140122',{[]};...
            # '20160808',{[]};...
            # '20160810',{[]};...
            # '20160812',{[]};...
            # '20160815',{[]};...
            # '20160817',{[]};...
            # '20160819',{[]};...
            # '20160822',{[]};...
            # '20160824',{[]};...
            # '20160826',{[]};...
            # '20160830',{[]};...
            # '20160917',{[]};...
            # '20160921',{[]};...
            # '20160923',{[]};...
            # '20160926',{[]};...
            # '20160927',{[]};...
            # '20161001',{[]};...
            # '20161006',{[]};...
            # '20161009',{[]};...
            # '20161010',{[]};...
            # '20161013',{[]};...
            # '20161014',{[]};...
            # '20161015',{[]};...
            # '20161018',{[]};...
            '20200119':[],
            '20200122':[],
            '20200124':[],
            '20200126':[],
            '20200128':[],
            '20200130':[],
            '20200131':[],
            '20200202':[],
            '20200205':[],
            '20200207':[],
            '20200209':[],
            '20200211':[],
            '20200213':[],
            '20200215':[],
            '20200218':[],
            # HALO-(AC)3
            '20220311': [],
            '20220312': [],
            '20220313': [],
            '20220314': [],
            '20220315': [],
            '20220316': [],
            '20220320': [],
            '20220321': []}
        # Assign sawtooth_table to class
        self.sawtooth=sawtooth

    
    def assess_radiometer_data(self):
        # assess_radiometer_data
        # use this to assess radiometer measurement errors
        # Syntax:  [output1,output2] = function_name(input1,input2,input3)
        #
        # Inputs:
        #       
        # See also: 
       
        #   Author: Dr. Heike Konow
        #   Meteorological Institute, Hamburg University
        #   email address: heike.konow@uni-hamburg.de
        #   Website: http://www.mi.uni-hamburg.de/
        #   June 2017; Last revision: April 2020
        # 
        #   Python Code Version
        #   Author: Henning Dorff
        #   Meteorological Institute, Hamburg University
        #   email adress: henning.dorff@uni-hamburg.de
        #   website: http://www.mi.uni-hamburg.de/
        #   July 2021
       
        #------------- BEGIN CODE --------------

        # Get dates and folder paths

        # Get used dates from campaign
        used_dates = self.cfg_dict["Flight_Dates_used"]
        # % Set path to data
        base_folder = self.cfg_dict["campaign_path"]+"Flight_Data/"+\
                        self.cfg_dict["campaign"]
        radiometer_path=base_folder+"/radiometer/"
        self.raw_radiometer_path=radiometer_path
        
        bahamas_path=base_folder+"/bahamas/"
        
        radiometer_strings = ['183', '11990', 'KV']
        channel_paths=[radiometer_path+"/"+freq+"/" \
                       for freq in radiometer_strings]
        
        major_plotting_path=base_folder+"/plots/"
        radiometer_fig_path=major_plotting_path+"radiometer_errors/"
            
        if not os.path.exists(major_plotting_path):
            os.mkdir(major_plotting_path)
            if not os.path.exists(radiometer_fig_path):
                os.mkdir(radiometer_fig_path)
                print("Radiometer Figure Path created as:", 
                      radiometer_fig_path)
        self.radiometer_fig_path=radiometer_fig_path
            
        if self.create_figures:
            for date in used_dates:
                #Preallocate radiometer dict
                self.radiometer_tb_dict=dict.fromkeys(radiometer_strings)
                # Loop radiometers
                f=0
                for freq in self.radiometer_tb_dict.keys():
                    # Get file path
                    filepath = glob.glob(channel_paths[f]+str(date)[2:]+"*.nc")
                    
                    # Check if files in directory
                    if not len(filepath)==0: # not a optimal solution
                        # Read data
                        
                        radiometer_ds=xr.open_dataset(filepath[0])
                        self.radiometer_tb_dict[freq]=pd.DataFrame(
                        data=np.array(radiometer_ds["TBs"][:].astype(float)),
                        index=pd.DatetimeIndex(np.array(radiometer_ds.time[:])),
                        columns=np.array(radiometer_ds["frequencies"][:])\
                                    .round(2).astype(str))
                        print("Radiometer ",freq,"Channel read")
                    f+=1
                # Plot radiometer TBs 
                self.plot_radiometer_TBs(date,raw_measurements=True)    
    
    def identify_radiometer_errors(self):
        # Old args
        #,do_figures=False,calc=False,
        # overview=True,check=False):
        
        # To assess quality of radiometer data, use the script below. 
        # 1. In a first step set figures=true to go through all flights and 
        # identify error indices. 
        # These should be noted in the file radiometer_errors_lookup
        # 2. Set calc=true to calculate percentages of errors
        # 3. Set overview=true for overview figure


        #%% Set parameters
        
        ##Set if figures should be produced
        #self.figures = do_figures
        ##Set if error time steps should be calculated from indices
        #self.calc = calc
        ## Set if overview figure should be produced
        #self.overview = overview
        ##Set if you want to check the results of error removal
        #self.check = check
        ## Set campaign to analyse
        #self.campaign = self.cfg_dict["campaign"]
        
        #%% asssess radiometer data and identify errors

        self.assess_radiometer_data()
            
        #%% Convert intervals to error flags

        # This only works if you have analysed the errors in the steps above
        # and noted the intervals in radiometer_errors_lookup.m 
        

        # Convert to flag for radiometer
        self.convert_radiometer_error_times()

        # Convert to flag for radiometer for errors in individual channels
        self.convert_radiometer_error_times_single_channel()
    
    ###########################################################################
    def convert_radiometer_error_times(self):
        # Load error indices
        self.radiometer_errors_lookup()
        self.uni_path= self.cfg_dict["campaign_path"]+"Flight_Data/"+\
                        self.cfg_dict["campaign"]+"/all_pickle/"
        #self.radiometer_path=self.cfg_dict["campaign_path"]+"Flight_Data/"+\
        #                        self.cfg_dict["campaign"]+"all_nc/"
        """
        Check if flag npy file exists, otherwise create and or add entries
        """
        #self.uni_path= self.cfg_dict["campaign_path"]+"Flight_Data/"+\
        #                self.cfg_dict["campaign"]+"all_pickle/"
        
        flag_file    = "error_flag_radiometer_"+\
            str(self.cfg_dict["Flight_Dates_used"][0])+".pkl"
       
        if os.path.exists(self.uni_path+flag_file):
            with open(self.uni_path+flag_file,"rb") as flag_f:
                flag_dict= pickle.load(flag_f)
        else:
            flag_dict={}
        
            for date in self.cfg_dict["Flight_Dates_used"]:
                flag_dict[date]={}
            
        # Loop radiometer modules
        for date in self.cfg_dict["Flight_Dates_used"]:
            for module in self.radiometer_modules:
                if module not in flag_dict[date].keys():
                    flag_dict[date][module]={}
                # Display instrument
                print(module,"-Module")
                if module=="KV":
                    pass
                #% Loop flight dates
                for day in [*self.cfg_dict["Flight_Dates_used"]]:
    
                    #% Get path to latest version of processed radiometer file
                    file_construction=self.raw_radiometer_path+\
                                             str(module)+"/*"+str(day)[2:]+"*.NC"
                    file_list_day= glob.glob(file_construction)
                    time_raw_list=[] # List of xr.DataArrays
                    k=0
                    for file in file_list_day:
                        #% Preallocate array
                        #% Loop all found files
                        # Read time from original files
                        
                        time_raw_list.append(xr.open_dataset(file).time)
                        k+=1
                    ## Concatenate
                    if len(time_raw_list) > 1:
                        time_raw = pd.DatetimeIndex(np.array(
                                                xr.concat(time_raw_list)))
                    else:
                        time_raw= pd.DatetimeIndex(np.array(time_raw_list[0]))
                
            
                    errors_day = self.errors[module][str(day)]
                    sawteeth_day = self.sawtooth[module][str(day)]
    
                    # Read time from unified data
                    self.uni_file="uniData_radiometer_"+str(day)+".pkl"
                    with open(self.uni_path+self.uni_file,"rb") as uni_file:
                        uni_dict=pickle.load(uni_file)
                        
                    time_uni = uni_dict["uni_time"]
                    #----> to be changed  #% Convert time to serial date number
                    # Create flags and fill with zeros
                    time_error_flag = pd.Series(data=0,
                                      index=pd.DatetimeIndex(time_uni))
                    time_saw_flag = pd.Series(data=0,
                                      index=pd.DatetimeIndex(time_uni))
    
                    # If errors array is not empty
                    if len(errors_day)!=0:
                        # Loop all errors for current date
                        for error in errors_day:
                            # Loc error indices to variable
                            # If only one index is given
                    
                            if len(error)==1: 
                                raw_time_error=time_raw.iloc[error[0]]
                                #  Make sure that error index time 
                                #  is after first time step from uni time
                                #  Find according time interval indices in
                                #  unified grid
                                series_raw_time=pd.Series(data=np.nan, 
                                                          index=raw_time_error)
                                series_raw_time=series_raw_time.truncate(
                                                        before=time_uni[0],
                                                        after=time_uni[-1])
                                raw_time_error=series_raw_time.index
                                time_error_flag.loc[raw_time_error]=1
                            
                            elif len(error)>1:
                                raw_time_error=time_raw[error[0]:error[1]]
                                series_raw_time=pd.Series(data=np.nan, 
                                                          index=raw_time_error)
                                series_raw_time=series_raw_time.truncate(
                                                        before=time_uni[0],
                                                        after=time_uni[-1])
                                raw_time_error=series_raw_time.index
                                time_error_flag.loc[raw_time_error]=1
                                
                            else:
                                continue
            
                    # If sawtooth array is not empty
                    if len(sawteeth_day)!=0:
        
                        # Loop all errors for current date
                        for saw_tooth in sawteeth_day:
                            #Loc error indices to variable
                            if len(saw_tooth)==0:
                                continue
                            #% If only one index is given
                            elif len(saw_tooth)==1:
                                raw_time_tooth=time_raw.iloc[saw_tooth[0]]
                                time_saw_flag.loc[raw_time_tooth]=1
                            else:
                                raw_time_tooth=time_raw.iloc[\
                                                saw_tooth[0]:saw_tooth[1]]
                                # Set flag to 1
                                time_saw_flag.loc[raw_time_tooth] = 1
                
                module_flag_df=pd.DataFrame(data=np.nan,
                                            index=time_saw_flag.index,
                                            columns=["errors","sawteeth"])
                module_flag_df["errors"]=time_error_flag
                module_flag_df["sawteeth"]=time_saw_flag
                flag_dict[date][module]=module_flag_df    
        
        # Save flags to pickle file
        with open(self.uni_path+flag_file,"wb") as flag_f:
            pickle.dump(flag_dict,flag_f,protocol=-1)
    ###########################################################################    
    def convert_radiometer_error_times_single_channel(self):
        # Load error indices
        self.radiometer_single_channel_errors_lookup()
        self.uni_path= self.cfg_dict["campaign_path"]+"Flight_Data/"+\
                        self.cfg_dict["campaign"]+"/all_pickle/"
        """
        Check if flag pkl file exists, otherwise create and or add entries
        """
        
        flag_file    = "error_flag_radiometer_"+\
            str(self.cfg_dict["Flight_Dates_used"][0])+".pkl"
       
        if os.path.exists(self.uni_path+flag_file):
            with open(self.uni_path+flag_file,"rb") as flag_f:
                flag_dict= pickle.load(flag_f)
        else:
            flag_dict={}
            for date in self.cfg_dict["Flight_Dates_used"]:
                flag_dict[date]={}
        
        #radiometer_frequencies=pd.DataFrame()
        # Loop radiometer modules
        channel_dict={}
        for module in self.radiometer_modules:
            # Display instrument
            print(module,"-Module")
            if module=="KV":
                pass
            #% Loop flight dates
            for date in [*self.cfg_dict["Flight_Dates_used"]]:
                if module not in flag_dict[date].keys():
                    flag_dict[date][module]={}
                #module_channel_df=pd.DataFrame()
            
                #% Get path to latest version of processed radiometer file
                file_construction=self.raw_radiometer_path+\
                                         str(module)+"/*"+str(date)[2:]+"*.NC"
                file_list_day= glob.glob(file_construction)
                if len(file_list_day)>0:
                    time_raw_list=[] # List of xr.DataArrays
                    k=0
                    for file in file_list_day:
                        #read radiometer frequencies
                        if k==0:
                            try:
                                module_freqs=xr.open_dataset(file).frequencies
                            except:
                                module_freqs=xr.open_dataset(file).Freq
                        #% Preallocate array
                        #% Loop all found files
                        # Read time from original files
                        
                        time_raw_list.append(xr.open_dataset(file).time)
                        k+=1
                    ## Concatenate
                    if len(time_raw_list) > 1:
                        time_raw = pd.DatetimeIndex(np.array(
                                            xr.concat(time_raw_list)))
                    else:
                        time_raw= pd.DatetimeIndex(np.array(time_raw_list[0]))
                
                # Read time from unified data
                self.uni_file="uniData_radiometer_"+str(date)+".pkl"
                with open(self.uni_path+self.uni_file,"rb") as uni_file:
                    uni_dict=pickle.load(uni_file)
                    
                time_uni = uni_dict["uni_time"]
                #----> to be changed  #% Convert time to serial date number
                
                # Load frequency error dict
                frequency_errors=self.single_channel_errors
                
                channel_flag_df=pd.DataFrame(data=0,
                                    columns=module_freqs.round(2).astype(str),
                                    index=pd.DatetimeIndex(time_uni))
            
                # Loop over all freqs o module
                for freq in module_freqs.data:
                    # Create flags and fill with zeros
                    time_channel_flag = pd.Series(data=0,
                                  index=pd.DatetimeIndex(time_uni))
                
                    print("Evaluate module freq:",freq)
                    # if the given frequency is in error dict
                    if str(freq.round(2)) in frequency_errors.keys():
                        incorrect_freq=str(freq.round(2))
                        # Check if date is existent for given frequency
                        if str(date) in frequency_errors[incorrect_freq].keys():
                            print("Errors for ",freq,"GHz were found")
                            # loop all errors of given frequency 
                            for channel_error in frequency_errors[\
                                                        incorrect_freq][str(date)]:
                                # Loc error indices to variable
                                # If only one index is given
                
                                if len(channel_error)==1: 
                                    raw_time_error=time_raw.iloc[channel_error[0]]
                                    #  Make sure that error index time 
                                    #  is after first time step from uni time
                                    #  Find according time interval indices in
                                    #  unified grid
                                    series_raw_time=pd.Series(data=np.nan, 
                                                      index=raw_time_error)
                                    series_raw_time=series_raw_time.truncate(
                                                    before=time_uni[0],
                                                    after=time_uni[-1])
                                    raw_time_error=series_raw_time.index
                                    time_channel_flag.loc[raw_time_error]=1
                        
                                elif len(channel_error)>1:
                                    raw_time_error=time_raw[channel_error[0]:\
                                                            channel_error[1]]
                                    series_raw_time=pd.Series(data=np.nan, 
                                                      index=raw_time_error)
                                    series_raw_time=series_raw_time.truncate(
                                                    before=time_uni[0],
                                                    after=time_uni[-1])
                                    raw_time_error=series_raw_time.index
                                    time_channel_flag.loc[raw_time_error]=1
                            
                                else:
                                    continue
            
                channel_flag_df[incorrect_freq]=time_channel_flag.values
            channel_dict[module]=channel_flag_df
        flag_dict[date]["Single_Channels_"+module]=channel_dict
        # Save flags to pickle file
        with open(self.uni_path+flag_file,"wb") as flag_f:
            pickle.dump(flag_dict,flag_f,protocol=-1)
            print("single channel errors added to ",self.uni_path+flag_file)
      

class Radar_Errors():
    def __init__(self,cfg_dict):
        self.cfg_dict=cfg_dict
    def convert_radar_error_times(self):
        #% convert to flag for radar
        #convertRadarErrorTimes(campaign)
        pass
    
def main(function_configurated=False,
         prcs_cfg_dict=None):
    
    if not function_configurated:
        import config_handler
        from campaign_time import specify_dates_to_use as specify_dates

        Flight_Dates={}
        Flight_Dates["EUREC4A"]={"RF01":"20200119","RF02":"20200122",
                             "RF03":"20200124","RF04":"20200126",
                             "RF05":"20200128","RF06":"20200130",
                             "RF07":"20200131","RF08":"20200202",
                             "RF09":"20200205","RF10":"20200207",
                             "RF11":"20200209","RF12":"20200211",
                             "RF13":"20200213","RF14":"20200215",
                             "RF15":"20200218"}
        #%%
        # load config files
        cfg=config_handler.Configuration(major_path=os.getcwd())
        #major_cfg_name="major_cfg"
        processing_cfg_name="unified_grid_cfg"    
        major_cfg_name="major_cfg"

        #Output file name prefix
        # The usual file name will follow the format: 
        # <instrument>_<date>_v<version-number>.nc
        # An additional file name prefix can be specified here (e.g. for EUREC4A),

        # if no prefix is necessary, set to empty string ('')
        # #filenameprefix = 'EUREC4A_HALO_';
        #     filenameprefix = ''

        campaign="EUREC4A"
        filenameprefix = campaign+'_HALO_'
        #%%
        print("=================== Configuration ============================")
        major_cfg=cfg.open_or_create_config_file(arg=1,name=major_cfg_name,
                                         campaign=campaign)
        data_cfg=cfg.open_or_create_config_file(arg=1,name=processing_cfg_name,
                                        campaign=campaign)
        
        # Comments for data files
        # Specify comment to be included into data files
        comment = 'Preliminary data! Uncalibrated Data.'+\
                    ' Only use for preliminary work!'
        # Specify contact information
        contact = 'henning.dorff@uni-hamburg.de'
    
        # Check if config-File exists and if not create the relevant first one
        if cfg.check_if_config_file_exists(major_cfg_name):
            major_config_file=cfg.load_config_file(cfg.major_path,
                                                   major_cfg_name)
        else:
            cfg.create_new_config_file(file_name=major_cfg_name+".ini")
        if cfg.check_if_config_file_exists(processing_cfg_name):
            processing_config_file=cfg.load_config_file(cfg.major_path,
                                                        processing_cfg_name)
        else:
            cfg.create_new_config_file(file_name=processing_cfg_name+".ini")
    
        if sys.platform.startswith("win"):
            system_is_windows=True
        else:
            system_is_windows=False

        if system_is_windows:
            if not major_config_file["Input"]["system"]=="windows":
                windows_paths={
                    "system":"windows",
                    "campaign_path":os.getcwd()+"/"    
                    }
                windows_paths["save_path"]=windows_paths["campaign_path"]+\
                                                "Save_path/"
        
        
            cfg.add_entries_to_config_object(major_cfg_name,
                                             windows_paths)
            cfg.add_entries_to_config_object(major_cfg_name,
                                             {"Comment":comment,
                                              "Contact":contact})

        if not processing_config_file["Input"]["system"]=="windows":
            windows_paths={
                "system":"windows",
                "campaign_path":os.getcwd()+"/"    
                }
            windows_paths["save_path"]=windows_paths["campaign_path"]+"Save_path/"
        
        
            cfg.add_entries_to_config_object(processing_cfg_name,windows_paths)
            cfg.add_entries_to_config_object(processing_cfg_name,
                                         {"Comment":comment,
                                          "Contact":contact})

        # %% Specify time frame for data conversion
        # % Start date
        start_date = '20200131';  
        # % End date
        end_date = '20200201';

        #%%Define processing steps
        #  Set version information
        #  Missing value
        #       set value for missing value (pixels with no measured signal). 
        #       This should be different from NaN, 
        #       since NaN is used as fill value 
        #       (pixels where no measurements were conducted)#
        #  Set threshold for altitude to discard radiometer data
        #  Set threshold for roll angle to discard radiometer data
        cfg.add_entries_to_config_object(processing_cfg_name,
                                 {"t1":start_date,
                                  "t2":end_date,
                                  "flight_date_used":start_date,
                                  "correct_attitude":False,
                                  "add_radarmask":False,
                                  "unify_Grid":True,
                                  "quicklooks":True,
                                  "remove_clutter":True,
                                  "remove_side_lobes":True,
                                  "remove_radiometer_errors":True,
                                  "version":0,
                                  "subversion":8,
                                  "missing_value":-888,
                                  "fill_value": np.nan,
                                  "altitude_threshold":4800,
                                  "add_radar_mask_values":True,
                                  "roll_threshold":5})
        #%% Define masking criteria when adding radar mask
        cfg.add_entries_to_config_object(processing_cfg_name,
                                 {"land_mask":1,
                                  "noise_mask":1,
                                  "calibration_mask":1,
                                  "surface_mask":1,
                                  "seasurface_mask":1,
                                  "num_RangeGates_for_sfc":4})

        processing_config_file=cfg.load_config_file(os.getcwd(),
                                                    processing_cfg_name)
    
        processing_config_file["Input"]["data_path"]=processing_config_file["Input"]\
                                                ["campaign_path"]+"Flight_Data/"
        processing_config_file["Input"]["device_data_path"]=processing_config_file["Input"]\
                                                ["data_path"]+"EUREC4A/"
    
        prcs_cfg_dict=dict(processing_config_file["Input"])    

        #%% Relevant flight dates
        #   Specify the relevant flight dates for the period of start
        #   and end date given above

        flightdates_use = specify_dates(prcs_cfg_dict["t1"],
                                prcs_cfg_dict["t2"],
                                Flight_Dates);

    else:
        pass
    # Used for later processing
    prcs_cfg_dict["campaign"]=[*Flight_Dates][0]
    prcs_cfg_dict["Flight_Dates"]=Flight_Dates
    prcs_cfg_dict["Flight_Dates_used"]=flightdates_use

    radiometer_errors=Radiometer_Errors(prcs_cfg_dict)
    radiometer_errors.identify_radiometer_errors()
    

if __name__=="__main__":
    main(function_configurated=False,
         prcs_cfg_dict=None)    
