import pandas as pd
import numpy as np


from pandas_profiling import ProfileReport   #pip install pandas_profiling 

##############################################
# Column 별 Basic Statistics 

##############################################

def Stat_profiling(fileroot,saveoption=None) :   #read csv file, report save as json or html
    
    f = open(fileroot, 'r', encoding="UTF-8")
    content = f.read()
    profile = ProfileReport(content, title="Profiling Report")
    
    # if saveoption == None:
    #     print(profile)
    # elif saveoption == 'json':
    #     profile.to_file(fileroot".json")

    # elif saveoption == 'html':
    #     profile.to_file(".html")

    return content