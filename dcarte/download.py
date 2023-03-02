import pandas as pd 
import os 

import datetime as dt
from .config import get_config
from .load import load
from pathlib import Path
from .utils import date2iso
from .domains import domains
sep = os.sep
cfg = get_config()
NOW = date2iso(str(dt.datetime.now()))



def download_domain(domain:str) -> None:
    """

    Args:
        domain: the name of the domain you want to download 

    Returns:
        None
    """
    print("THIS FUNCTION MAY TAKE LONG")
    output = [] 
    domains_ = domains()
    datasets = domains[[domain.upper()]].replace('',np.nan).dropna()
    for dataset in datasets:
        output.append(load(dataset,domain))
    
    return output
    
    