# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 16:40:41 2020

@author: u300737
"""
import sys
import os
from configparser import ConfigParser
from termcolor import colored

class Configuration():
    def __init__(self,major_path):
       self.major_path=major_path 
    
    def create_new_config_file(self,file_name="data_config",campaign=''):
        
        #Get the configparser object
        config_object= ConfigParser()
        if sys.platform.startswith("win"):
            system_is_windows=True
        else:
            system_is_windows=False

        config_object["Input"]={
                "system":"linux",
                "campaign":campaign,
                "campaign_path":"/scratch/uni/u237/users/hdorff/",
                "filenameprefix" : campaign+'_HALO_',
                }
        config_object["Input"]["save_path"]="/home/zmaw/u300737/PhD/Work/"+\
                config_object["Input"]["campaign"]
        major_config_file=self.load_config_file(file_name)
        
        if system_is_windows:
            if not major_config_file["Input"]["system"]=="windows":
                windows_paths={
                    "system":"windows",
                "campaign_path":self.major_path+"/"    
                }
                windows_paths["save_path"]=windows_paths["campaign_path"]+"Save_path/"
        
        
#                self.add_entries_to_config_object(major_cfg_name,windows_paths)
#                self.add_entries_to_config_object(major_cfg_name,{"Comment":comment,
#                                                         "Contact":contact})
            
        file_name=self.major_path+file_name+".ini"
        with open(file_name,'w') as conf:
            config_object.write(conf)
            print("Config-File ",colored(file_name,"red"),
                  "is created!")
    
    def check_if_config_file_exists(self,name):
        if not os.path.isfile(name):
            self.create_new_config_file(file_name=name)
            file_exists=True
        else:
            print("Config-file",name+".ini"," already exists")
        return True
    
    def load_config_file(self,name):
        config_object = ConfigParser()
        file_name=self.major_path+"/"+name+".ini"
        print(file_name)
        config_object.read(file_name)
        return config_object
    def open_or_create_config_file(self,arg=2,name="major_cfg_file",campaign=''):
        if arg==1:
            print("Create Config_file")
            self.create_new_config_file(file_name=name,campaign=campaign)
        elif arg==2:
            print("Load config file")
            self.load_config_file(self.path,name)
        return self
    def add_entries_to_config_object(self,config_file_name,entry_dict):
        """
    
        Parameters
        ----------
        config_file_name: DICT
            file name of the config-file
        entry_dict : DICT
            dictionary of entries to add in the config file.
        Returns
        -------
        None
    
        """    
        config_object_old= ConfigParser()
        config_object_old.read(self.major_path+config_file_name+".ini")
        
        # add dictionary entries to config_object
        for key in entry_dict.keys():
            config_object_old["Input"][key]=str(entry_dict[key])
        config_object_new=config_object_old
        # write config_objects into data_config file    
        with open(self.major_path+config_file_name+".ini",'w') as conf:
            config_object_new.write(conf)
        print("Entries: ",entry_dict.keys(),
              "have added to or changed in the config file")
    
        return None
    
    def del_entries_from_config_object(self,entry_dict):
        """
    
        Parameters
        ----------
        entry_dict : DICT
            dictionary of entries to delete from the config file.
        Returns
        -------
        None.
    
        """    
        config_object_old= ConfigParser()
        config_object_old.read("data_config.ini")
        
        # dictionary entries to be deleted from config_object
        for key in entry_dict.keys():
            del config_object_old[key]
        
        # write config_objects into data_config file    
        with open("data_config.ini",'w') as conf:
            config_object= ConfigParser()
            config_object.write(conf)
        print("Entries: ",entry_dict.keys(),"have added to the config file")
        return None
    
    def adapt_config_file_to_system(self,is_windows):
        """
        This function adapts the config_file to
        the system one is working on.
    
        Parameters
        ----------
        is_windows : TYPE
            DESCRIPTION.
    
        Returns
        -------
        None.
    
        """
        return None
    
    
    def print_config_file(self,major,keys):
        """
        Parameters
        ----------
        keys : DICT.KEYS
        
        Simply returns the desired keys of config_file for quickcheck.    
    
        Returns
        -------
        dictionary : dict
            object containing all entries in config_file
    
        """
        config_file=ConfigParser()
        config_file.read("data_config.ini")
        dictionary=config_file[major][keys]
        print("The defined specifications of the config file for ",
              colored(major,'red'),"[",colored(keys,'green'),"] "
              "are:",colored(dictionary,"magenta"))
        return dictionary
    
    def return_default_config_dict(self,major_cfg_name,processing_cfg_name,
                                   campaign,comment="nothing",
                                   contact="max.mustermann@uni-hausen.de"):
        print("=================== Configuration ============================")
        major_cfg=self.open_or_create_config_file(arg=1,name=major_cfg_name,
                                         campaign=campaign)
        data_cfg=self.open_or_create_config_file(arg=1,name=processing_cfg_name,
                                        campaign=campaign)

        


        # Check if config-File exists and if not create the relevant first one
        if self.check_if_config_file_exists(major_cfg_name):
            major_config_file=self.load_config_file(major_cfg_name)
        else:
            self.create_new_config_file(file_name=major_cfg_name+".ini")
        
        if self.check_if_config_file_exists(processing_cfg_name):
            processing_config_file=self.load_config_file(processing_cfg_name)
        else:
            self.create_new_config_file(file_name=processing_cfg_name+".ini")
        
        if sys.platform.startswith("win"):
            system_is_windows=True
        else:
            system_is_windows=False

        if system_is_windows:
            if not major_config_file["Input"]["system"]=="windows":
                windows_paths={
                    "system":"windows",
                "campaign_path":self.major_path+"/"    
                }
                windows_paths["save_path"]=windows_paths["campaign_path"]+"Save_path/"
        
        
                self.add_entries_to_config_object(major_cfg_name,windows_paths)
                self.add_entries_to_config_object(major_cfg_name,{"Comment":comment,
                                                         "Contact":contact})

        if not processing_config_file["Input"]["system"]=="windows":
            windows_paths={
                "system":"windows",
                "campaign_path":self.major_path+"/"}
            windows_paths["save_path"]=windows_paths["campaign_path"]+"Save_path/"
        
        
        self.add_entries_to_config_object(processing_cfg_name,windows_paths)
        self.add_entries_to_config_object(processing_cfg_name,
                                         {"Comment":comment,
                                          "Contact":contact})

        return self