import pandas as pd 
import numpy as np
from .local import LocalDataset
from .minder import MinderDataset
from .utils import (load_yaml, 
                    timer,
                    date2iso,
                    path_exists,
                    read_table)
from .config import get_config
import datetime as dt

NOW = date2iso(str(dt.datetime.now()))


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
    dflt = get_defaults(**kwargs) | kwargs
    home = cfg['home']
    data_folder = cfg['data_folder']
    datasets = pd.DataFrame(cfg['domains'])
    if not (datasets == np.array([dataset,domain])).all(axis=1).any():
        raise Exception(f"Sorry, {dataset} is not a registered dataset in {domain} domain in dcarte")
    
    local_file = f'{data_folder}/{domain}/{dataset}.parquet'
    if path_exists(local_file) and not (dflt['update'] or dflt['reapply'] or dflt['reload']):
        return read_table(local_file)
    else:     
        info = load_yaml(f'{home}/dcarte/config/{domain}.yaml')
        if domain in ['raw','lookup']:
            input = {'dataset_name':dataset,
                     'datasets':info[dataset]['datasets'],
                     'columns':info[dataset]['columns'],
                     'dtypes':info[dataset]['dtype'],
                     'domain':domain} | dflt

            output = MinderDataset(**input)
        else:
            dependencies = pd.DataFrame(info[dataset]['domains'])
            parent_datasets = {}
            for _,row in dependencies.iterrows():
                parent_datasets[row.dataset] = load(row.dataset,row.domain, **dflt)
            input = {'dataset_name':dataset,
                     'datasets':parent_datasets,
                     'pipeline':info[dataset]['pipeline'],
                     'module':info[dataset]['module'],
                     'domain':domain} | dflt
            
            output = LocalDataset(**input)

        return output.data

def get_defaults(**kwargs):
    """get_defaults [summary]

    [extended_summary]

    Returns:
        [type]: [description]
    """
    cfg = get_config()
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

