import pandas as pd
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import dcarte
from dcarte.utils import process_transition, localize_time,load_csv_from_zip,timer
from dcarte.local import LocalDataset
from dcarte.config import get_config
import zipfile
import dcarte


# On RCS use the following location to point to the legacy zip files 
TIHM = '/../projects/dri_healthyhome/live/tihm/'
# On your personal computer I tend to use TIHM = '/external/tihm/' and manualy copy these zips from rcs



@timer('process_observation')
def process_observation(obj):
    cfg = get_config()
    tihmdri_zip = zipfile.ZipFile(f'{cfg["home"]}{TIHM}tihmdri.zip')
    tihm15_zip = zipfile.ZipFile(f'{cfg["home"]}{TIHM}tihm15.zip')
    tihm15_df = load_csv_from_zip(tihm15_zip,'Observations.csv')
    tihmdri_df = load_csv_from_zip(tihmdri_zip,'Observations.csv')
    df = pd.concat([tihm15_df,tihmdri_df])
    return df

@timer('process_device_type')                                 
def process_device_type(obj):
    cfg = get_config()
    tihmdri_zip = zipfile.ZipFile(f'{cfg["home"]}{TIHM}tihmdri.zip')
    Observation_type = pd.read_csv(tihmdri_zip.open('Observation-type.csv'))
    return Observation_type


@timer('process_motion')    
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

@timer('process_doors')  
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

@timer('process_entryway') 
def process_entryway(self):
    df = self.datasets['doors']
    entryways = ['front door', 'back door', 'main door']
    df = df.query('location_name in @entryways and transition == "opened>closed"').copy()
    df.location_name = df.location_name.astype('str')\
                           .str.replace('main door', 'front door')\
                           .astype('category')
                           
    return df.reset_index(drop=True)

@timer('process_temperature')                                   
def process_temperature(self):
    df = self.datasets['observation']
    columns = ['datetimeObserved','type','subject','location','valueQuantity','valueUnit']
    df = df.query('type == "60832-3"')[columns]
    df.columns = ['start_date', 'device_type', 'patient_id','location_name','value', 'unit']
    df = df.assign(source='raw_ambient_temperature')
    df = localize_time(df,['start_date'])
    return df
    
@timer('process_light')                                     
def process_light(self):
    df = self.datasets['observation']
    columns = ['datetimeObserved','type','subject','location','valueQuantity','valueUnit']
    df = df.query('type == "56242006"')[columns]
    df.columns = ['start_date', 'device_type', 'patient_id','location_name','value', 'unit']
    df = df.assign(source='raw_light')
    df = localize_time(df,['start_date'])
    return df    

@timer('process_flags')                                     
def process_flags(self):
    cfg = get_config()
    flags = []
    for legacy in ['tihmdri','tihm15']:
        zip_ = zipfile.ZipFile(f'{cfg["home"]}{TIHM}{legacy}.zip')
        df = pd.read_csv(zip_.open('Flags.csv'))
        _type = pd.read_csv(zip_.open('Flag-type.csv'))
        _val = pd.read_csv(zip_.open('FlagValidations.csv'))
        df = pd.merge(df, _val, how='outer', on=None,
                      left_on="flagId", right_on="flag",
                      suffixes=('df', '_val'), copy=True)
        df.category = df.category.replace({1:'Clinical',2:'Null',3:'Technical'}).astype('category')
        map_type = _type.set_index('code').display.to_dict()
        map_type = {**map_type, **{8:'Blood Oxygen Saturation',24:'Shingles'}}
        df.type = df.type.replace(map_type).astype('category')
        df['project_id'] = legacy
        flags.append(df)
    df = pd.concat(flags)
    return df

@timer('process_wellbeing')                                
def process_wellbeing(self):
    cfg = get_config()
    wellbeing = []
    for legacy in ['tihmdri','tihm15']:
        zip_ = zipfile.ZipFile(f'{cfg["home"]}{TIHM}{legacy}.zip')
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
    
@timer('process_physiology')     
def process_physiology(self):
    df = self.datasets['observation']
    device_type = self.datasets['device_type']
    devices = ['8310-5','150456','55284-4','29463-7','251837008','163636005','8462-4','8480-6','8867-4']
    df = df.query('type in @devices')
    mapping = device_type.set_index('code').display.to_dict()
    mapping = {**mapping, **{'8867':'raw_heart_rate'}}
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


@timer('create_legacy_datasets')                                
def create_legacy_datasets():
    domain = 'legacy'
    module = 'legacy'
    module_path = __file__

    LocalDataset(dataset_name='observation',
                 datasets={},
                 pipeline=['process_observation'],
                 domain=domain,
                 module=module,
                 module_path=module_path)
    observation = dcarte.load('observation', 'legacy')

    LocalDataset(dataset_name='device_type',
                 datasets={},
                 pipeline=['process_device_type'],
                 domain=domain,
                 module=module,
                 module_path=module_path)
    device_type = dcarte.load('device_type', 'legacy')

    LocalDataset(dataset_name='flags',
                 datasets={},
                 pipeline=['process_flags'],
                 domain=domain,
                 module=module,
                 module_path=module_path)
    flags = dcarte.load('flags', 'legacy')

    LocalDataset(dataset_name='wellbeing',
                 datasets={},
                 pipeline=['process_wellbeing'],
                 domain=domain,
                 module=module,
                 module_path=module_path)
    wellbeing = dcarte.load('wellbeing', 'legacy')

    for dataset in ['motion',
                    'doors',
                    'physiology',
                    'temperature',
                    'light']:
        LocalDataset(dataset_name=dataset,
                     datasets={'observation': observation,
                               'device_type': device_type},
                     pipeline=[f'process_{dataset}'],
                     domain=domain,
                     module=module,
                     module_path=module_path,
                     dependencies=[['observation', 'legacy'], ['device_type', 'legacy']])

    LocalDataset(dataset_name='entryway',
                 datasets={'doors': dcarte.load('doors', 'legacy')},
                 pipeline=['process_entryway'],
                 domain=domain,
                 module=module,
                 module_path=module_path,
                 dependencies=[['doors', 'legacy']])
    
if __name__ == "__main__":
    create_legacy_datasets()  