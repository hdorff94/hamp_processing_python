# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 19:30:41 2020

@author: u300737
"""

#General functions
class performance():
    def __init__(self):
        pass
    
    def updt(self,total, progress):
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
    
def str2bool(s):
    return s.lower() in ("yes","true","t",1,"1")
    
#    def round_partial(self,data,resolution):
#        """
#        value may be a pandas series
#        """
#        partial_rounded=round(data/resolution)*resolution
#       
#        return partial_rounded
