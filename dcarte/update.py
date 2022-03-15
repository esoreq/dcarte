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
    datasets = {}   
    files = list(Path(cfg['data']).rglob('*.parquet'))
    for i,file in enumerate(files):
        datasets[i] = dict(dataset = file.stem,domain = file.parent.stem)
    datasets = pd.DataFrame(datasets).T
    for _,row in datasets.query('domain in @domain').iterrows():
        _ = load(row.dataset,row.domain,update=True)