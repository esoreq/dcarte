import pandas as pd
import os
import sys
import numpy as np
import dcarte
from dcarte.utils import process_transition, localize_time
from dcarte.local import LocalDataset
from dcarte.config import get_config
import dcarte


def map_devices(df, dt):
    dt.type = dt.type.str.replace('"|\[|\]', '',regex=True)\
                     .str.replace(' ', '_')
    mapping = dt.set_index('id').type.to_dict()
    df.device_type = df.device_type.replace(mapping).astype('category')
    return df


def process_physiology(self):
    _d1 = map_devices(self.datasets['vital_signs'],
                      self.datasets['device_types'])
    _d2 = map_devices(self.datasets['blood_pressure'],
                      self.datasets['device_types'])
    bp_types = np.array(['diastolic_value', 'systolic_value'])
    df = [_d1]

    for bp_type in bp_types:
        cols = _d2.columns[~_d2.columns.str.startswith(bp_type)]
        _tmp = _d2[cols].copy()
        _tmp.source = bp_types[bp_types != bp_type][0].replace('_value', '_bp')
        _tmp.columns = _d1.columns
        df.append(_tmp)
    df = pd.concat(df)
    dtypes = {'device_type': 'category', 'patient_id': 'category',
              'home_id': 'category', 'unit': 'category',
              'source': 'category', 'value': 'float'}
    df = df.astype(dtypes).reset_index(drop=True)
    df = localize_time(df,['start_date'])
    
    return df


def process_habitat(self):
    df = map_devices(self.datasets['environmental'],
                      self.datasets['device_types'])
    df = df[df.location_name != ''].reset_index(drop=True)
    dtypes = {'home_id': 'category', 'location_id': 'category',
              'unit': 'category', 'location_name': 'category',
              'source': 'category', 'value': 'float'}
    df = df.astype(dtypes)
    df = localize_time(df,['start_date'])
    return df


def process_doors(self):
    groupby = ['patient_id','location_name']
    datetime = 'start_date'
    value = 'value'
    covariates = ['location_name']
    df = self.datasets['door']
    df = localize_time(df,['start_date'])
    doors = process_transition(df,
                               groupby, 
                               datetime, 
                               value, 
                               covariates)
    return doors.drop('location_name',1).reset_index()


def process_entryway(self):
    df = self.datasets['doors'].copy()
    df = df[df.location_name.isin(['front door', 'back door', 'main door'])]
    df.location_name = df.location_name.astype('str')\
                           .str.replace('main door', 'front door')\
                           .astype('category')
    df = df[df.transition == 'opened>closed']     

    return df.reset_index(drop=True)


def process_kitchen(self):
    # TODO: add sensor type as a column
    activity = self.datasets['activity']
    activity = localize_time(activity,['start_date'])
    doors = self.datasets['doors']
    appliances = map_devices(self.datasets['appliances'],
                             self.datasets['device_types'])
    doors = doors[(doors.location_name == 'fridge door') &
                  (doors.transition == 'opened>closed')]
    activity = activity[activity.location_name.isin(['kitchen',
                                                     'dining room'])]
    activity.insert(0,'device_type','pir')
    doors.insert(0,'device_type','door')
    appliances.value = appliances.value.astype(str)\
                                 .replace({'microwave-use': 'oven-use',
                                 'toaster-use': 'oven-use', 'iron-use': np.nan,
                                 'multi-use socket-use': np.nan})
    doors = doors[['patient_id', 'start_date', 'device_type','location_name']]\
                   .rename({'location_name': 'activity'}, axis=1)   
    activity = activity[['patient_id', 'start_date', 'device_type', 'location_name']]\
                        .rename({'location_name': 'activity'}, axis=1)       
    appliances = appliances[['patient_id', 'start_date', 'device_type', 'value']]\
                            .rename({'value': 'activity'}, axis=1) 
    kitchen = pd.concat([doors, activity, appliances.dropna()])   
    kitchen = kitchen[kitchen.patient_id != ''].sort_values(['patient_id', 'start_date']) 
    kitchen = process_transition(kitchen, ['patient_id'], 'start_date', 'activity', ['device_type'])
    return kitchen.reset_index(drop=True)


def process_motion(self):
    activity = self.datasets['activity']
    activity = localize_time(activity,['start_date'])
    entryway = self.datasets['entryway']
    bed_occupancy = self.datasets['sleep_event'].rename({'value':'location_name'},axis=1)
    fact = ['patient_id','location_name', 'start_date']
    motion = pd.concat([activity[fact], entryway[fact], bed_occupancy[fact]]).\
                        sort_values(['patient_id', 'start_date'])

    mapping = {'bathroom1': 'Bathroom', 
               'WC1': 'Bathroom',
               'kitchen': 'Kitchen',
               'hallway': 'Hallway',
               'corridor1': 'Hallway',
               'dining room': 'Lounge',
               'living room': 'Lounge',
               'lounge': 'Lounge',
               'bedroom1': 'Bedroom',
               'front door': 'Front door',
               'back door': 'Back door'}                    
    motion.location_name = motion.location_name.replace(mapping)

    motion = motion[~motion.location_name.isin(['office', 
                                                'conservatory', 
                                                'study', 
                                                'cellar'])]    
    return motion.reset_index(drop=True)

def process_transitions(self):
    motion = self.datasets['motion']
    motion = process_transition(motion, ['patient_id'], 'start_date', 'location_name')
    return motion.reset_index()


def process_sleep(self):
    sleep_mat = self.datasets['sleep_mat']    
    sleep_mat = localize_time(sleep_mat,['start_date'])
    return sleep_mat


def create_base_datasets():
    domain = 'base'
    module = 'base'
    parent_datasets = { 'Doors':[['door','raw']], 
                        'Entryway':[['doors','base']], 
                        'Habitat':[['environmental','raw'],
                                   ['device_types','lookup']], 
                        'Kitchen':[['appliances','raw'],
                                   ['doors','base'],
                                   ['activity','raw'],
                                   ['device_types','lookup']], 
                        'Motion':[['activity','raw'],
                                  ['entryway','base'],
                                  ['sleep_event','raw']], 
                        'Physiology':[['vital_signs','raw'],
                                      ['blood_pressure','raw'],
                                      ['device_types','lookup']],
                        'Sleep':[['sleep_mat','raw']], 
                        'Transitions':[['motion','base']]}
    
    
    for dataset in ['Doors', 
                    'Entryway', 
                    'Habitat', 
                    'Kitchen', 
                    'Motion', 
                    'Physiology',
                    'Sleep', 
                    'Transitions']:
        
        p_datasets = {d[0]:dcarte.load(*d) for d in parent_datasets[dataset]} 
        
        LocalDataset(dataset_name = dataset,
                     datasets = p_datasets,
                     pipeline = [f'process_{dataset.lower()}'],
                     domain = domain,
                     module = module,
                     dependencies = parent_datasets[dataset])
    
if __name__ == "__main__":
    create_base_datasets()  