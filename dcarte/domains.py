from .config import get_config
from .utils import isnotebook
import pandas as pd

def domains():
    """domains prints the current potential local domains as a table to stdout

    [extended_summary]
    """
    cfg = get_config()
    df = pd.DataFrame(cfg['domains'])
    df.dataset = df.dataset.str.title()
    df = df.pivot(columns='domain', values='dataset')
    temp = []
    for col in df:
        temp.append(df[col].drop_duplicates().sort_values().dropna().reset_index(drop=True))
    df = pd.concat(temp, axis=1)
    df = df[df.notnull().sum().sort_values(ascending=False).index]
    df = df.fillna('')
    df.columns = df.columns.str.upper()
    if isnotebook:
        return df
    else: 
        print(df.to_string())    
    
