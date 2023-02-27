import pandas as pd 
import numpy as np
import os 
import json
from .local import LocalDataset
from .minder import MinderDataset
from .utils import (load_yaml, 
                    timer,
                    date2iso,
                    merge_dicts,
                    path_exists,
                    read_table,
                    read_metadata)
from pathlib import Path
from .config import get_config
import datetime as dt

NOW = date2iso(str(dt.datetime.now()))
sep = os.sep

@timer('Loading')
def load(dataset:str,domain:str,**kwargs):
    """load [summary]

    [extended_summary]

    Args:
        dataset (str): [description]
        domain (str): [description]

    Raises:
        Exception: [description]

    Returns:
        [type]: [description]
    """
    dataset,domain = dataset.lower(),domain.lower()
    cfg = get_config()
    dflt = get_defaults(cfg,**kwargs) 
    home = cfg['home']
    data_folder = cfg['data_folder']
    datasets = pd.DataFrame(cfg['domains'])
    last_update = None
        
    if not (datasets == np.array([dataset,domain])).all(axis=1).any():
        raise Exception(f"Sorry, {dataset} is not a registered dataset in {domain} domain in dcarte")
    

        
    local_file = f'{data_folder}{sep}{domain}{sep}{dataset}.parquet'
    if path_exists(local_file):
        local_dataset = read_table(local_file)
        if 'start_date' in local_dataset.columns:
            last_update = local_dataset.start_date.max()
    

    if (dflt['reapply'] or dflt['reload']) and path_exists(local_file):
        dflt['reapply'] = False
        Path(local_file).unlink()
        return load(dataset,domain,**dflt)

    if path_exists(local_file):
        if not (dflt['reload'] or dflt['reapply'] or dflt['update']):
            return local_dataset
    
    if path_exists(local_file):
        hdr = read_metadata(local_file)
        metadata = json.loads(hdr.metadata[b'minder'].decode())
        update = pd.to_datetime(metadata['until'])+pd.Timedelta(hours=dflt['delay']) > pd.to_datetime(dflt['until'])
        if not update:
            if pd.to_datetime(metadata['until'])+pd.Timedelta(hours=dflt['delay']) > pd.to_datetime(dflt['until']):
                return read_table(local_file)

         
    info = load_yaml(f'{home}{sep}dcarte{sep}config{sep}{domain}.yaml')
    if domain in ['raw','lookup','care']:
        input = {'dataset_name':dataset,
                 'datasets':info[dataset]['datasets'],
                 'columns':info[dataset]['columns'],
                 'dtypes':info[dataset]['dtype'],
                 'domain':domain} 
        input = merge_dicts(input,dflt)
        output = MinderDataset(**input)
    else:
        dependencies = pd.DataFrame(info[dataset]['domains'])
        parent_datasets = {}
        if dflt['reapply']:
            dflt['reapply'] = False
        for _,row in dependencies.iterrows():
            parent_datasets[row.dataset] = load(row.dataset,row.domain, **dflt)
        input = {'dataset_name':dataset,
                 'datasets':parent_datasets,
                 'pipeline':info[dataset]['pipeline'],
                 'last_update':last_update,
                 'module':info[dataset]['module'],
                 'dependencies':info[dataset]['domains'],
                 'domain':domain} 
        input = merge_dicts(input,dflt)
        output = LocalDataset(**input)

    return output.data

def get_defaults(cfg,**kwargs):
    """get_defaults [summary]

    [extended_summary]

    Returns:
        [type]: [description]
    """
    # cfg = get_config()
    defaults = {'since': '2019-04-01',
                'until': NOW,
                'delay': 1,
                'home': cfg['home'],
                'compression': cfg['compression'],
                'data_folder':  cfg['data_folder'],
                'update': False,
                'reload':  False,
                'reapply': False}

    defaults.update(kwargs)
    return defaults

