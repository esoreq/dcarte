import pandas as pd 
import os 

import datetime as dt
from .config import get_config
from .load import load
from pathlib import Path
from .utils import date2iso
sep = os.sep
cfg = get_config()
NOW = date2iso(str(dt.datetime.now()))

def update_raw() -> None:
    """Updates data by loading parquet files from specified directories and running `load`.

    The function iterates over all files in the directories 'raw', 'lookup', and 'care' specified in the config file.
    For each parquet file found, it calls the `load` function to update the data.
    Note that the `load` function is not defined in this code snippet and must be imported from elsewhere.

    Args:
        None

    Returns:
        None
    """
    print("THIS FUNCTION MAY TAKE LONG")
    datasets = {}
    data_dir = Path(cfg['data'])
    files = list(data_dir.rglob('*.parquet'))
    for i, file in enumerate(files):
        datasets[i] = dict(dataset=file.stem, domain=file.parent.stem)
    datasets = pd.DataFrame(datasets).T
    for _, row in datasets.query('domain in ["raw","lookup","care"]').iterrows():
        _ = load(row.dataset, row.domain, update=True)
    
        
        
def update_domain(domain:str ="base")-> None:
    """
    Updates datasets in the specified domain.

    Args:
    - domain (str, optional): Domain name to update (default: "base")

    Returns:
    - None
    """
    print("THIS FUNCTION MAY TAKE LONG")
    datasets = {}   
    files = list(Path(cfg['data']).rglob('*.parquet'))
    for i,file in enumerate(files):
        datasets[i] = dict(dataset = file.stem,domain = file.parent.stem)
    datasets = pd.DataFrame(datasets).T
    for _,row in datasets.query('domain in @domain').iterrows():
        _ = load(row.dataset,row.domain,update=True)
