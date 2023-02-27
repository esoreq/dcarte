import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
import sys
# This line allows you to clone the repository and skip installing dcarte
import datetime as dt
from scipy.stats import circmean,circstd
from dcarte.utils import (between_time,
                          mine_transition,
                          time_to_angles)
import dcarte
from dcarte.local import LocalDataset

def mean_time(x): return pd.Series(circmean(x,high=360),name='mean')
def std_time(x): return pd.Series(circstd(x,high=360),name='std')


def process_activity_dailies(obj):
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

def process_activity_weeklies(obj):
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

def mine_vital_signs(sleep):
    sleep_vitals = sleep.groupby(['patient_id','period_segments']).agg(      
            start_date = ('start_date','min'),
            end_date = ('start_date','max'),
            minutes_snoring = ('snoring','sum'),
            heart_rate = ('heart_rate','mean'),
            hr_min = ('heart_rate','min'),
            hr_max = ('heart_rate','max'),
            respiratory_rate = ('respiratory_rate','mean'),
            rr_min = ('respiratory_rate','min'),
            rr_max = ('respiratory_rate','max'),
            period_obs = ('respiratory_rate','count'),
        ).dropna()
    return sleep_vitals

def segment_periods(df,gap_period=1):
    segments = (df.start_date.shift(-1) - df.end_date).shift() > pd.Timedelta(hours=gap_period)
    return df.assign( period_segments = segments.cumsum())

def mine_sleep_states(sleep):
    # sleep.state = sleep.state.replace({'LIGHT':'OTHER','REM':'OTHER'})
    start_end = sleep.groupby(['patient_id','period_segments']).agg(start_date = ('start_date','min'),end_date = ('start_date','max'))
    sleep_states = (sleep
                    .assign(duration=1)
                    .groupby(['patient_id','period_segments','state'])
                    .duration
                    .sum()
                    /60).unstack() 
    sleep_states = pd.concat([sleep_states,start_end],axis=1).dropna()
    return sleep_states

def mine_bed_habits(bed_occupancy):
    df = (bed_occupancy.groupby(['patient_id','period_segments']).
             agg(start_date=('start_date', 'min'), 
                 end_date=('end_date', 'max'), 
                 awake_events = ('transition',lambda x: x.count()-1),
                 time_in_bed=('dur', lambda x: x.sum()/(60*60))))   
    df = df.assign(bed_time_period=(df.end_date - df.start_date)/np.timedelta64(1, 'h'))
    df = df.assign(time_out_of_bed=(df.bed_time_period - df.time_in_bed))
    df = map_daily_period_type(df).dropna()
    df.end_date = df.end_date-pd.Timedelta(minutes=1)
    return df

def map_daily_period_type(habits, start_time = dt.time(8), end_time = dt.time(20)):
    expr = " ".join(["start_time <= habits.start_date.dt.time",
                  " and habits.end_date.dt.time < end_time"
                  " and (habits.start_date.dt.date == habits.end_date.dt.date)"])
    habits['period_type'] = pd.eval(expr).map({False:'Nocturnal',True:'Diurnal'})
    return habits

def process_sleep_dailies(obj):
    """sleep_dailies creates a daily summary across key features from the sleep mat
    
    Args:
        obj ([LocalDataset]): [description]

    Returns:
        pd.DataFrame: a pandas dataframe with patient_id 
    """
    sleep = obj.datasets['sleep']
    bed_occupancy = obj.datasets['bed_occupancy']
    bo_transition = bed_occupancy.groupby('patient_id').apply(mine_transition, value = 'location_name')
    bed_in = bo_transition.query('transition == "Bed_in>Bed_out"')
    bed_in = bed_in.groupby('patient_id').apply(segment_periods)
    habits = mine_bed_habits(bed_in)
    mapper = habits.reset_index().set_index(['patient_id','start_date']).period_segments.to_dict()
    sleep = sleep.assign( period_segments = sleep.set_index(['patient_id','start_date']).index.map(mapper))
    sleep.period_segments = ((sleep.period_segments+1)>0).cumsum()
    sleep_states = mine_sleep_states(sleep)
    sleep_vitals = mine_vital_signs(sleep)
    keys = ['patient_id','start_date','end_date']
    sleep_states_ = sleep_states.reset_index().drop(columns='period_segments').set_index(keys)
    habits_ = habits.reset_index().drop(columns='period_segments').set_index(keys)
    sleep_vitals_ = sleep_vitals.reset_index().drop(columns='period_segments').set_index(keys)
    sleep_periods = sleep_vitals_.join(habits_).join(sleep_states_).round(2)
    sleep_periods = sleep_periods.dropna(subset=['time_in_bed','DEEP','hr_max'])
    sleep_metrics = resample_sleep_metrics(sleep_periods)

    diurnal_habits = resample_sleep_metrics(sleep_periods,'Diurnal')
    diurnal_habits = diurnal_habits.assign(nap_ibp = diurnal_habits.time_in_bed)
    sleep_metrics = pd.merge(sleep_metrics,diurnal_habits.nap_ibp,how='left',left_index=True, right_index=True)
    sleep_metrics.nap_ibp = sleep_metrics.nap_ibp.fillna(0)
    sleep_metrics = sleep_metrics.assign(time_to_bed = (sleep_metrics.start_time.dt.time.apply(time_to_angles)+180)%360,
                                         wake_up_time = sleep_metrics.end_time.dt.time.apply(time_to_angles))
    return sleep_metrics


def resample_sleep_metrics(sleep_periods,period_type:str="Nocturnal"):
    habits = sleep_periods.query('period_type == @period_type').drop(columns=['period_type'])
    habits = (habits.
              reset_index().
              assign(datetime = habits.reset_index().start_date).
              set_index('datetime').
              groupby('patient_id').
              resample('1D',offset='12h').agg(
                start_time = ('start_date', 'min'),
                end_time = ('end_date', 'max'),
                nb_awakenings = ('awake_events' ,lambda x: x if x.shape[0]==1 else x.sum()+x.shape[0]-1),
                time_in_bed = ('time_in_bed' ,'sum'),
                period_obs = ('period_obs' ,'sum'),
                minutes_snoring = ('minutes_snoring' ,'sum'),
                heart_rate = ('heart_rate', 'mean'),
                hr_min = ('hr_min' ,'min'),
                hr_max = ('hr_max' ,'max'),
                respiratory_rate = ('respiratory_rate' ,'mean'),
                rr_min = ('rr_min' ,'min'),
                rr_max = ('rr_max' ,'max'),
                AWAKE = ('AWAKE' ,'sum'),
                DEEP = ('DEEP' ,'sum'),
                REM = ('REM' ,'sum'),
                LIGHT = ('LIGHT' ,'sum')
              ).dropna())
    habits = habits.assign(bed_time_period=(habits.end_time - habits.start_time)/np.timedelta64(1, 'h'))
    habits = habits.assign(time_out_of_bed=(habits.bed_time_period - habits.time_in_bed))
    return habits


def get_awake_events(x):
    return (x.diff() > pd.Timedelta(minutes=1)).sum()


def process_sleep_model(obj):
    """sleep_model creates a ready to use feature space for the sdi model 
    
    Args:
        obj ([LocalDataset]): [description]

    Returns:
        pd.DataFrame: a pandas dataframe with patient_id 
    """
    df = obj.datasets['sleep_dailies']
    mapper = {  'time_in_bed':'ibp', 
                'minutes_snoring':'snoring_time', 
                'heart_rate':'hr_average', 
                'respiratory_rate':'rr_average',
                'time_out_of_bed':'obp', 
                'time_to_bed':'To_bed',
                'wake_up_time':'Arise'
                }
    df = df.rename(columns=mapper)            
    df = df.assign(
        deep_ratio = (df.DEEP)/df.ibp,
        awake_ratio = (df.AWAKE)/df.ibp,
        rem_ratio = (df.REM)/df.ibp,
        light_ratio = (df.LIGHT)/df.ibp,
        exit_ratio = (df.nb_awakenings/(df.ibp+df.obp)),
        night = df.index.get_level_values(1).date   
    )

    return df
    

def process_physiology_dailies(obj):
    """physiology_dailies creates a daily summary across key features from the dialy vital signs
    
    Args:
        obj ([LocalDataset]): [description]

    Returns:
        pd.DataFrame: a pandas dataframe with patient_id 
    """
    dp = obj.datasets['physiology']
    ds = obj.datasets['sleep_dailies']
    factors = ['raw_heart_rate','raw_body_weight','raw_body_mass_index','raw_oxygen_saturation',
               'raw_body_temperature','systolic_bp','diastolic_bp','raw_total_body_fat']
    daily_physiology = dp.query("source in @factors")
    daily_physiology = (daily_physiology.reset_index(drop=True).
                        groupby(['patient_id', 'source']).
                        resample('1D', on='start_date', offset='12h').
                        agg({'value': 'mean'}).
                        swaplevel(-2, -1).
                        unstack().
                        droplevel(0, axis=1))
    daily_physiology = daily_physiology[factors]
    daily_physiology.columns = ['Heart_rate','Weight','BMI','Oxygen_Saturation','Temperature','Systolic_BP','Diastolic_BP', 'Body_Fat']
    ds = (ds.reset_index()[['patient_id','start_date','respiratory_rate','heart_rate']]
              .groupby(['patient_id']).
               resample('1D', on='start_date', offset='12h').
               agg(RR_rest = ('respiratory_rate' , 'mean'),
                   HR_rest = ('heart_rate' , 'mean')))
    daily_physiology = daily_physiology.join(ds)
    
    return daily_physiology

def process_physiology_weeklies(obj):
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
    mapping = {'bathroom1': 'Bathroom', 
        'WC1': 'Bathroom',
        'kitchen': 'Kitchen',
        'hallway': 'Hallway',
        'corridor1': 'Hallway',
        'dining room': 'Lounge',
        'living room': 'Lounge',
        'lounge': 'Lounge',
        'study': 'Lounge',
        'office': 'Lounge',
        'conservatory': 'Lounge',
        'bedroom1': 'Bedroom',
        'main door':'Front door',
        'front door': 'Front door',
        'back door': 'Back door'}  

    Light.location_name = Light.location_name.astype(str).replace(mapping)
    Light = Light.query('["garage", "secondary"] not in location_name')
    Light = (Light.groupby(['patient_id', 'location_name']).
                resample('15T', on='start_date').
                mean().fillna(method='ffill')).reset_index()
    Light.insert(3, 'date', Light.start_date.dt.date)
    Light.insert(4, 'time', Light.start_date.apply(time_to_angles))
    return Light

def process_temperature(obj):
    Habitat = obj.datasets['Habitat']
    temperature = Habitat.query('source != "raw_light"')
    
    
    mapping = {'bathroom1': 'Bathroom', 
                'WC1': 'Bathroom',
                'kitchen': 'Kitchen',
                'hallway': 'Hallway',
                'corridor1': 'Hallway',
                'dining room': 'Lounge',
                'living room': 'Lounge',
                'lounge': 'Lounge',
                'study': 'Lounge',
                'office': 'Lounge',
                'conservatory': 'Lounge',
                'bedroom1': 'Bedroom',
                'main door':'Front door',
                'front door': 'Front door',
                'back door': 'Back door'}  

    temperature.location_name = temperature.location_name.astype(str).replace(mapping)
    temperature = temperature.query('["garage", "secondary"] not in location_name')
    temperature = (temperature.groupby(['patient_id', 'location_name']).
                resample('15T', on='start_date').
                mean().fillna(method='ffill')).reset_index()
    temperature.insert(3, 'date', temperature.start_date.dt.date)
    temperature.insert(4, 'time', temperature.start_date.apply(time_to_angles))
    return temperature

def create_weekly_profile():
    module_path = __file__
    # since = '2022-02-10'
    # until = '2022-02-20'
    module = "weekly_profile"
    domain = 'profile'
    parent_datasets = { 'activity_dailies':[['motion','base']], 
                        'activity_weeklies':[['activity_dailies','profile']], 
                        'sleep_dailies':[['sleep','base'],['bed_occupancy','base']], 
                        'physiology_dailies':[['physiology','base'],['sleep_dailies','profile']], 
                        'light':[['Habitat','base']], 
                        'temperature':[['Habitat','base']],
                        'sleep_model':[['sleep_dailies','profile']]}
    for dataset in parent_datasets.keys():
        p_datasets = {d[0]:dcarte.load(*d) for d in parent_datasets[dataset]} 
        LocalDataset(dataset_name = dataset,
                        datasets = p_datasets,
                        pipeline = [f'process_{dataset.lower()}'],
                        domain = domain,
                        module = module,
                        module_path = module_path,
                        dependencies = parent_datasets[dataset])
    
if __name__ == "__main__":
    create_weekly_profile()  