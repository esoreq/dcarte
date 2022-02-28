import pandas as pd
import numpy as np
import dcarte
from dcarte.local import LocalDataset

def process_occupancy(df):
    """process_occupancy convert sleep_mat raw minute activity to daily occupancy

    process_occupancy takes the raw sleep_mat observations and aggragates those to
    Daily summeries sampled between '18:00'-'12:00' to ignore naps 
    It recieves a dcarte dataset object and returns a pandas DataFrame

    Args:
        obj ([dcarte dataset object]): [description]

    Returns:
        pandas DataFrame: A tabular time series of daily frequencies containing
        total minutes spent in bed  
    """
    factors = ['patient_id','start_date']
    df = df[factors].assign(activity=True)
    df.patient_id = df.patient_id.astype('category')
    df = (df.set_index('start_date').
            between_time('18:00', '12:00').
            groupby(['patient_id']).
            resample('24h', offset="12h").
            activity.sum()/60).reset_index()
    df = df.assign(unit='hour')
    df = df.rename(columns={'activity': 'Time_in_bed'})
    return df

def iqr_outcome(x, labels=['Low TIB', 'Normal TIB', 'High TIB']):
    """iqr_outcome Calculates the interquartile range of the cohort 
                   To develop a naive and meaningful baseline label

    Args:
        x (pd.Series): hourly time spent in bed
        labels (list, optional): [description]. Defaults to ['Low TIB','Normal TIB', 'High TIB'].

    Returns:
        pd.Series: categorical outcome labels
    """
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3-q1
    bins = [q1-iqr*1.5, q1, q3, q3+iqr*1.5]
    outcome = (pd.cut(x, bins, labels= labels).
               cat.add_categories(['OOB',
                                   'Extremely low',
                                   'Extremely high']))
    
    outcome[x>q3+iqr*1.5]='Extremely high'
    outcome[x<q1-iqr*1.5]='Extremely low'
    outcome[x==0]='OOB'
    outcome = outcome.cat.reorder_categories(['OOB',
                                              'Extremely low',
                                              'Low TIB',
                                              'Normal TIB', 
                                              'High TIB',
                                              'Extremely high'])
    return outcome

def process_outcome(self):
    """Process_outcome approximates alerts calculated based on normative time
    (constructed using all legacy data)
    
    Args:
        self ([LocalDataset]): [description]

    Returns:
        pd.DataFrame: a pandas data with patient_id start_date and time_in_bed
    """
    df = process_occupancy(self.datasets['sleep'])
    df['global_outcome'] = (
        df[['Time_in_bed']].
        apply(iqr_outcome)
    )
    df.global_outcome = pd.Categorical(df.global_outcome,['OOB','Low TIB','Normal TIB', 'High TIB'],ordered=True)
    return df

def create_bed_occupancy():
    dataset_name = 'bed_occupancy'
    datasets = {'sleep': dcarte.load('sleep','base')}
    pipeline = ['process_outcome']
    domain = 'bed_habits'
    module = 'bed_occupancy'
    module_path = __file__
    df = LocalDataset(dataset_name = dataset_name,
                      datasets = datasets,
                      pipeline = pipeline,
                      domain = domain,
                      module = module,
                      module_path = module_path,
                      update = True,
                      dependencies = [['sleep','base']])

if __name__ == "__main__":
    create_bed_occupancy()
