import pandas as pd
import os
import sys
import dcarte
from dcarte.utils import process_transition, localize_time,load_csv_from_zip
from dcarte.local import LocalDataset
from dcarte.config import get_config
import zipfile
import dcarte

TIHM = '/external/tihm_dri/'

def process_observation(obj,TIHM = '/external/tihm_dri/'):
    cfg = get_config()
    tihmdri_zip = zipfile.ZipFile(f'{cfg["data_folder"]}{TIHM}tihmdri.zip')
    tihm15_zip = zipfile.ZipFile(f'{cfg["data_folder"]}{TIHM}tihm15.zip')
    tihm15_df = load_csv_from_zip(tihm15_zip,'Observations.csv')
    tihmdri_df = load_csv_from_zip(tihmdri_zip,'Observations.csv')
    df = pd.concat([tihm15_df,tihmdri_df])
    return df

def process_device_type(obj):
    cfg = get_config()
    tihmdri_zip = zipfile.ZipFile(f'{cfg["data_folder"]}{TIHM}tihmdri.zip')
    Observation_type = pd.read_csv(tihmdri_zip.open('Observation-type.csv'))
    return Observation_type


def process_motion(obj):
    df = obj.datasets['observation']
    df = (df[['subject','location','datetimeObserved','type']].
            query('type == "272149007-2"').
            dropna(subset=['location']).
            drop(columns='type'))
    df.columns = ['patient_id', 'location_name', 'start_date']
    df = localize_time(df,['start_date'])
    df = df.drop_duplicates().reset_index()
    return df

def process_doors(self):
    df = self.datasets['observation']
    columns = ['subject','location','datetimeObserved','valueState','type']
    df = (df[columns].
            query('type == "224751004"').
            dropna(subset=['location']).
            drop(columns='type'))
    df.columns = ['patient_id', 'location_name', 'start_date','value']
    df.value = df.value.replace({'Close':'closed','Open':'opened'})
    groupby = ['patient_id','location_name']
    datetime = 'start_date'
    value = 'value'
    covariates = None
    df = localize_time(df,['start_date'])
    doors = process_transition( df,
                                groupby, 
                                datetime, 
                                value, 
                                covariates)
    doors = doors.reset_index()
    doors.location_name = doors.location_name.str.lower()
    return doors

def process_entryway(self):
    df = self.datasets['doors']
    entryways = ['front door', 'back door', 'main door']
    df = df.query('location_name in @entryways and transition == "opened>closed"').copy()
    df.location_name = df.location_name.astype('str')\
                           .str.replace('main door', 'front door')\
                           .astype('category')
                           
    return df.reset_index(drop=True)

def process_temperature(self):
    df = self.datasets['observation']
    columns = ['datetimeObserved','type','subject','location','valueQuantity','valueUnit']
    df = df.query('type == "60832-3"')[columns]
    df.columns = ['start_date', 'device_type', 'patient_id','location_name','value', 'unit']
    df = df.assign(source='raw_ambient_temperature')
    df = localize_time(df,['start_date'])
    return df
    
def process_light(self):
    df = self.datasets['observation']
    columns = ['datetimeObserved','type','subject','location','valueQuantity','valueUnit']
    df = df.query('type == "56242006"')[columns]
    df.columns = ['start_date', 'device_type', 'patient_id','location_name','value', 'unit']
    df = df.assign(source='raw_light')
    df = localize_time(df,['start_date'])
    return df    
    
def process_flags(self):
    cfg = get_config()
    flags = []
    for legacy in ['tihmdri','tihm15']:
        zip_ = zipfile.ZipFile(f'{cfg["data_folder"]}{TIHM}{legacy}.zip')
        df = pd.read_csv(zip_.open('Flags.csv'))
        _type = pd.read_csv(zip_.open('Flag-type.csv'))
        _val = pd.read_csv(zip_.open('FlagValidations.csv'))
        df = pd.merge(df, _val, how='outer', on=None,
                      left_on="flagId", right_on="flag",
                      suffixes=('df', '_val'), copy=True)
        df.category = df.category.replace({1:'Clinical',2:'Null',3:'Technical'}).astype('category')
        map_type = _type.set_index('code').display.to_dict()
        map_type = map_type | {8:'Blood Oxygen Saturation',24:'Shingles'}
        df.type = df.type.replace(map_type).astype('category')
        df['project_id'] = legacy
        flags.append(df)
    df = pd.concat(flags)
    return df

def process_wellbeing(self):
    cfg = get_config()
    wellbeing = []
    for legacy in ['tihmdri','tihm15']:
        zip_ = zipfile.ZipFile(f'{cfg["data_folder"]}{TIHM}{legacy}.zip')
        df = pd.read_csv(zip_.open('QuestionnaireResponses.csv'))
        df = df.drop(columns=["questionnaire", "datetimeReceived"])
        df = df.astype({'datetimeAnswered':'datetime64'})
        df = (df.sort_values(by=['datetimeAnswered']).
                drop_duplicates().
                set_index(['subject','datetimeAnswered','question']).
                unstack())
        df = df.droplevel(0,axis=1)
        df.insert(0,'project',legacy)
        wellbeing.append(df)
    df = pd.concat(wellbeing)
    return df
    
    
def process_physiology(self):
    df = self.datasets['observation']
    device_type = self.datasets['device_type']
    devices = ['8310-5','150456','55284-4','29463-7','251837008','163636005','8462-4','8480-6','8867-4']
    df = df.query('type in @devices')
    mapping = device_type.set_index('code').display.to_dict()
    mapping = mapping | {'8867-4':'raw_heart_rate'}
    columns = ['datetimeObserved','type','subject','valueQuantity','valueUnit']
    df = df[columns].assign(source=df.type.replace(mapping))
    mapper = {'Body temperature':'raw_body_temperature',
          'Total body water':'raw_total_body_water',
          'Systolic blood pressure':'systolic_bp',
          'Diastolic blood pressure':'diastolic_bp',
          'MDC_PULS_OXIM_SAT_O2':'raw_oxygen_saturation',
          'O/E - muscle mass':'raw_body_muscle_mass',
          'Body weight':'raw_body_weight'}
    df.source = df.source.replace(mapper)
    df.columns = ['start_date','device_type','patient_id','value','unit','source']
    df = localize_time(df,['start_date'])
    return df


def create_legacy_datasets(TIHM = '/external/tihm_dri/'):
    domain = 'legacy'
    module = 'legacy'
    LocalDataset('observation',{},['process_observation'],domain,module)
    observation = dcarte.load('observation','legacy')
    LocalDataset('device_type',{},['process_device_type'],domain,module)
    device_type = dcarte.load('device_type','legacy')
    LocalDataset('flags',{},['process_flags'],domain,module)
    flags = dcarte.load('flags','legacy')
    LocalDataset('wellbeing',{},['process_wellbeing'],domain,module)
    wellbeing = dcarte.load('wellbeing','legacy')
    
    for dataset in ['motion',
                    'doors',
                    'physiology',
                    'temperature',
                    'light']:
        
        LocalDataset(dataset_name= dataset,
                     datasets = {'observation': observation,
                                 'device_type': device_type},
                     pipeline = [f'process_{dataset}'],
                     domain = domain,
                     module = module,
                     dependencies = [['observation','legacy'],['device_type','legacy']])
    
    
    LocalDataset(dataset_name = 'entryway',
                 datasets = {'doors': dcarte.load('doors','legacy')},
                 pipeline = ['process_entryway'],
                 domain = domain,
                 module = module,
                 dependencies = [['doors','legacy']])

    
if __name__ == "__main__":
    create_legacy_datasets()  

    