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
    if not (datasets == np.array([dataset,domain])).all(axis=1).any():
        raise Exception(f"Sorry, {dataset} is not a registered dataset in {domain} domain in dcarte")
    
    local_file = f'{data_folder}{sep}{domain}{sep}{dataset}.parquet'
    if path_exists(local_file) and not (dflt['update'] or dflt['reload'] or dflt['reapply']):
        return read_table(local_file)
        # hdr = read_metadata(local_file)
        # metadata = json.loads(hdr.metadata[b'minder'].decode())
        # if metadata['since'] == dflt['since'] and metadata['until'] == dflt['until']:
        #     return read_table(local_file)
        # else: 
        #     kwargs['reapply'] = True
        #     return load(dataset,domain,**kwargs)
    else:     
        info = load_yaml(f'{home}{sep}dcarte{sep}config{sep}{domain}.yaml')
        if domain in ['raw','lookup']:
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
            for _,row in dependencies.iterrows():
                parent_datasets[row.dataset] = load(row.dataset,row.domain, **dflt)
            input = {'dataset_name':dataset,
                     'datasets':parent_datasets,
                     'pipeline':info[dataset]['pipeline'],
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

