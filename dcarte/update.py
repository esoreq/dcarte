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

def update_raw(**kwargs):
    """update_raw updates the raw data
    stored locally.

    To re-download all of the raw data, and overwrite the current
    local datasets for that domain, you would use: 

    >>> dcarte.update_raw(reload=True)

    Args:
        kwargs : Keyword arguments are passed to the dcarte.load function. update=True is already passed.

    Returns:
        None
    """
    print("THIS FUNCTION MAY TAKE LONG")
    datasets = {}   
    files = list(Path(cfg['data']).rglob('*.parquet'))
    for i,file in enumerate(files):
        datasets[i] = dict(dataset = file.stem,domain = file.parent.stem)
    datasets = pd.DataFrame(datasets).T
    for _,row in datasets.query('domain in ["raw","lookup"]').iterrows():
        _ = load(row.dataset,row.domain,update=True, **kwargs)
        
        
        
def update_domain(domain ="base", **kwargs):
    """update_raw updates the raw data
    stored locally.

    To re-download all of the data for DOMAIN, and overwrite the current
    local datasets for that domain, you would use: 

    >>> dcarte.update_domain(domain=DOMAIN, reload=True)

    Args:
        domain (str): The name of the domain to update.
        kwargs : Keyword arguments are passed to the dcarte.load function. update=True is already passed.

    Returns:
        None
    """
    print("THIS FUNCTION MAY TAKE LONG")
    datasets = {}   
    files = list(Path(cfg['data']).rglob('*.parquet'))
    for i,file in enumerate(files):
        datasets[i] = dict(dataset = file.stem,domain = file.parent.stem)
    datasets = pd.DataFrame(datasets).T
    for _,row in datasets.query('domain in @domain').iterrows():
        _ = load(row.dataset,row.domain,update=True, **kwargs)