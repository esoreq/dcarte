import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
import sys
# This line allows you to clone the repository and skip installing dcarte
from scipy.stats import circmean,circstd
from dcarte.utils import (between_time,
                          time_to_angles)
import dcarte
from dcarte.local import LocalDataset

def mean_time(x): return pd.Series(circmean(x,high=360),name='mean')
def std_time(x): return pd.Series(circstd(x,high=360),name='std')


def activity_dailies(obj):
    """Activity_dailies creates a daily summary across key locations
    
    Args:
        obj ([LocalDataset]): [description]

    Returns:
        pd.DataFrame: a pandas dataframe with patient_id 
    """
    df = obj.datasets['motion']
    drop = ['bed_in','Back door','Front door']
    activity_metrics = (df.
        query('location_name not in @drop').
        assign(activity=True).
        set_index('start_date').
        groupby(['patient_id','location_name']).
        resample('1D',offset='12h').
        activity.
        count().
        swaplevel(-2,-1).
        unstack())
    activity_metrics['Total'] = activity_metrics.sum(axis=1)
    activity_metrics= activity_metrics.query('Total > 0.0')
    return activity_metrics

def activity_weeklies(obj):
    """activity_weeklies creates a weekly summary across key locations based on activity_dailies
    
    Args:
        obj ([LocalDataset]): [description]

    Returns:
        pd.DataFrame: a pandas dataframe with patient_id 
    """
    df = obj.datasets['activity_dailies']
    activity_weeklies = (df.
                        reset_index().
                        groupby('patient_id').
                        resample('1W',on='start_date').
                        agg({col:['mean','std'] for col in  df.columns}))
    return activity_weeklies

def sleep_dailies(obj):
    """sleep_dailies creates a daily summary across key features from the sleep mat
    
    Args:
        obj ([LocalDataset]): [description]

    Returns:
        pd.DataFrame: a pandas dataframe with patient_id 
    """
    df = obj.datasets['sleep']
    sleep_metrics = between_time(df,'start_date','17:00','11:00').copy()
    sleep_metrics['time'] = sleep_metrics['start_date']
    sleep_metrics['snoring'] = sleep_metrics['snoring'].astype(float)
    sleep_metrics = (sleep_metrics.set_index('start_date').
                        groupby('patient_id').
                        resample('1D',offset='12h').
                        agg({'heart_rate':['mean'],
                            'respiratory_rate':['mean'],
                            'snoring':['sum'],
                            'time':['first','last','count']}))
    sleep_metrics.columns = ['hr','br','snr','ttb','wup','tib']
    sleep_metrics['tib'] = sleep_metrics['tib']/60
    sleep_metrics['tob'] = (sleep_metrics.wup - sleep_metrics.ttb).dt.total_seconds()/60**2 - sleep_metrics.tib
    sleep_metrics.ttb = sleep_metrics.ttb.dt.time.apply(time_to_angles)
    sleep_metrics.wup = sleep_metrics.wup.dt.time.apply(time_to_angles)
    sleep_metrics.columns = ['Heart rate','Breathing rate','Snoring','Time to bed','Wake up time','Time in bed','Time out of bed']
    sleep_metrics = sleep_metrics.dropna(subset=['Heart rate','Breathing rate','Time to bed','Wake up time'])
    return sleep_metrics

def sleep_weeklies(obj):
    """sleep_weeklies creates a weekly summary across key features based on sleep_dailies
    
    Args:
        obj ([LocalDataset]): [description]

    Returns:
        pd.DataFrame: a pandas dataframe with patient_id 
    """
    df = obj.datasets['sleep_dailies']
    df_ = (df.
            reset_index().
            groupby('patient_id').
            resample('1W',on='start_date').
            agg({'Heart rate': ['mean','std'],
                'Breathing rate': ['mean','std'],
                'Snoring': ['mean','std'],
                'Time to bed': [mean_time,std_time],
                'Wake up time':   [mean_time,std_time],
                'Time in bed': ['mean','std'],
                'Time out of bed': ['mean','std']}))
    return df_

def physiology_dailies(obj):
    """physiology_dailies creates a daily summary across key features from the dialy vital signs
    
    Args:
        obj ([LocalDataset]): [description]

    Returns:
        pd.DataFrame: a pandas dataframe with patient_id 
    """
    df = obj.datasets['physiology']
    factors = ['raw_heart_rate','raw_body_weight','raw_body_mass_index',
            'raw_body_temperature','diastolic_bp','systolic_bp','raw_total_body_fat']
    daily_physiology = df.query("source in @factors")
    daily_physiology = (daily_physiology.reset_index(drop=True).
                            groupby(['patient_id','source']).
                            resample('1D',on='start_date',offset='12h').
                            agg({'value':'mean'}).
                            swaplevel(-2,-1).
                            unstack().
                            droplevel(0,axis=1))
    daily_physiology = daily_physiology[factors][daily_physiology[factors].isnull().sum(axis=1)<len(factors)]
    daily_physiology.columns = ['Heart rate','Body_weight','BMI','Temperature','Diastolic_BP','Systolic_BP', 'Body_Fat']
    
    return daily_physiology

def physiology_weeklies(obj):
    """physiology_weeklies creates a weekly summary across key features based on physiology_dailies
    
    Args:
        obj ([LocalDataset]): [description]

    Returns:
        pd.DataFrame: a pandas dataframe with patient_id 
    """
    df = obj.datasets['physiology_dailies']
    physiology_weeklies = (df.
                           reset_index().
                           groupby('patient_id').
                           resample('1W',on='start_date').
                           agg({col:['mean','std'] for col in  df.columns}))
    return physiology_weeklies

def process_light(obj):
    Habitat = obj.datasets['Habitat']
    Light = Habitat.query('source == "raw_light"')
    Light = (Light.groupby(['patient_id', 'location_name']).
                resample('15T', on='start_date').
                mean().fillna(method='ffill')).reset_index()
    Light.insert(3, 'date', Light.start_date.dt.date)
    Light.insert(4, 'time', Light.start_date.apply(time_to_angles))
    return Light

def process_temperature(obj):
    Habitat = obj.datasets['Habitat']
    temperature = Habitat.query('source != "raw_light"')
    temperature = (temperature.groupby(['patient_id', 'location_name']).
                resample('15T', on='start_date').
                mean().fillna(method='ffill')).reset_index()
    temperature.insert(3, 'date', temperature.start_date.dt.date)
    temperature.insert(4, 'time', temperature.start_date.apply(time_to_angles))
    return temperature

def create_weekly_profile():
    module = "weekly_profile"
    data_type = 'profile'
    LocalDataset('activity_dailies',{'motion': dcarte.load('motion','base')},
                 ['activity_dailies'],data_type , module,[['motion','base']])
    LocalDataset('activity_weeklies',{'activity_dailies': dcarte.load('activity_dailies','profile')},
                 ['activity_weeklies'],data_type , module,[['activity_dailies','profile']])
    LocalDataset('sleep_dailies',{'sleep': dcarte.load('sleep','base')},
                 ['sleep_dailies'],data_type , module,[['sleep','base']])
    LocalDataset('sleep_weeklies',{'sleep_dailies': dcarte.load('sleep_dailies','profile')},
                 ['sleep_weeklies'],data_type , module,[['sleep_dailies','profile']])
    LocalDataset('physiology_dailies',{'physiology': dcarte.load('physiology','base')},
                 ['physiology_dailies'],data_type , module,[['physiology','base']])
    LocalDataset('physiology_weeklies',{'physiology_dailies': dcarte.load('physiology_dailies','profile')},
                 ['physiology_weeklies'],data_type , module,[['physiology_dailies','profile']])
    LocalDataset('light',{'Habitat': dcarte.load('Habitat','base')},
                 ['process_light'],data_type , module,[['Habitat','base']])
    LocalDataset('temperature',{'Habitat': dcarte.load('Habitat','base')},
                 ['process_temperature'],data_type , module,[['Habitat','base']])
    
if __name__ == "__main__":
    create_weekly_profile()  