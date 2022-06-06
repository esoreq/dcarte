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

def update_raw():
    """update_raw updates the all of raw data by downloading any 
    new data from the server. This function only updates the
    raw data and does not update any of the domains, which will 
    need to be updated using the function 
    ```dcarte.update_domain(DOMAIN)```.


    """
    print("THIS FUNCTION MAY TAKE LONG")
    datasets = {}   
    files = list(Path(cfg['data']).rglob('*.parquet'))
    for i,file in enumerate(files):
        datasets[i] = dict(dataset = file.stem,domain = file.parent.stem)
    datasets = pd.DataFrame(datasets).T
    for _,row in datasets.query('domain in ["raw","lookup"]').iterrows():
        _ = load(row.dataset,row.domain,update=True)
        
        
        
def update_domain(domain ="base"):
    print("THIS FUNCTION MAY TAKE LONG")
    """update_domain updates the data in the domain given.
    The usual behaviour of this function is to update all of the 
    data that feeds into the datasets within the domain, and then
    re-run the script that produced the domain in the first instance.
    This may lead to raw datasets being more or less updated than
    each other, but is preferable behaviour to updating all raw 
    datasets when only a fraction of them are used for the domain.
    Note that new data is downloaded from the server, and may take a long
    time, depending on the size of the update.


    KwArgs:
        domain (str): The domain to update the data of. Defaults to ```"base"```.


    """
    datasets = {}   
    files = list(Path(cfg['data']).rglob('*.parquet'))
    for i,file in enumerate(files):
        datasets[i] = dict(dataset = file.stem,domain = file.parent.stem)
    datasets = pd.DataFrame(datasets).T
    for _,row in datasets.query('domain in @domain').iterrows():
        _ = load(row.dataset,row.domain,update=True)