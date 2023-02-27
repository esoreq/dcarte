import pandas as pd 
import numpy as np
import os 
from .utils import (load_yaml, 
                    write_yaml,
                    delete_folder,
                    path_exists)
from pathlib import Path


sep = os.sep


def delete_dataset(dataset:str,domain:str):
    """Deletes a dataset from a specified domain.
    Parameters
    ----------
    dataset : str
        The name of the dataset to be deleted.
    domain : str
        The name of the domain from which to delete the dataset.

    Raises
    ------
    Exception
        If the dataset to be deleted is not registered in the specified domain.

    Returns
    -------
    None
    """
    from .config import get_config
    cfg = get_config()
    dataset,domain = dataset.lower(),domain.lower()
    datasets = pd.DataFrame(cfg['domains'])
    if not (datasets == np.array([dataset,domain])).all(axis=1).any():
        raise Exception(f"Sorry, {dataset} is not a registered dataset in {domain} domain in dcarte")
    else:
        print(f'ARE YOU SURE YOU WANT TO DELETE DATASET {dataset}?')    
        answer = input("Enter yes or no: ") 
        while answer not in ("yes", "no"): 
            print("Please only enter yes or no.")
            answer = input("Enter yes or no: ")  

        if answer == "yes": 
            delete_dataset_(cfg,dataset,domain)
            print(f'DATASET {dataset} from DOMAIN {domain} is now deleted')   

def delete_domain(domain:str) -> None:
    """
    Delete a domain and all datasets associated with it.

    Parameters
    ----------
    domain : str
        The name of the domain to be deleted.

    Raises
    ------
    Exception
        If the specified domain is not a registered domain in dcarte.

    """
    from .config import get_config
    cfg = get_config()
    domain = domain.lower()
    domains = pd.DataFrame(cfg['domains']).drop_duplicates(subset='domain')[['domain']].values
    if not (domain in domains):
        raise Exception(f"Sorry, {domain} is not a registered domain in dcarte")
    else:
        print(f'ARE YOU SURE YOU WANT TO DELETE DOMAIN {domain}?')    
        answer = input("Enter yes or no: ") 
        while answer not in ("yes", "no"): 
            print("Please only enter yes or no.")
            answer = input("Enter yes or no: ") 
        if answer == "yes": 
            print(f'ALL DATASETS INCLUDED IN DOMAIN {domain.upper()} WILL BE NOW DELETED!') 
            delete_domain_(cfg,domain)
            print(f'DOMAIN {domain} is now deleted')    

def delete_dataset_(cfg:dict,dataset:str,domain:str):        
    """
    Delete the specified dataset file from the specified domain folder and it's existents in the config file of dcarte and the domain.

    Parameters
    ----------
    cfg : dict
        The dictionary containing the dcarte configuration data.
    dataset : str
        The name of the dataset to be deleted.
    domain : str
        The name of the domain the dataset belongs to.

    Returns
    -------
    None
    """
    home = cfg['home']
    data_folder = cfg['data_folder']
    domain_path = f'{home}{sep}dcarte{sep}config{sep}{domain}.yaml'
    domain_yaml = load_yaml(domain_path)
    try:
        domain_yaml.pop(dataset)
    except:
        print(f'dataset {dataset} is not in config file')    
    if len(domain_yaml)==0:
        print('The domain {domain} is now empty and will also be removed') 
        delete_domain_(cfg,domain)
    else:    
        write_yaml(domain_path,domain_yaml)
    local_dataset_file = f'{data_folder}{sep}{domain}{sep}{dataset}.parquet'
    if path_exists(local_dataset_file):
        Path(local_dataset_file).unlink()
    config_path = f"{home}{sep}dcarte{sep}config.yaml"
    config_yaml = load_yaml(config_path)
    domains = pd.DataFrame(config_yaml['domains'])
    domains = domains.drop(domains.query('dataset == @dataset and domain == @domain').index)
    config_yaml['domains'] = domains.to_dict('records')
    write_yaml(config_path, config_yaml)


def delete_domain_(cfg:dict,domain:str):        
    """
    Deletes the domain yaml file, removes all datasets associated with the domain
    from the config file, and deletes the domain folder.

    Parameters:
    -----------
    cfg : dict
        A dictionary containing the configuration information for dcarte.
    domain : str
        The name of the domain to be deleted.

    """
    home = cfg['home']
    data_folder = cfg['data_folder']
    # delete domain yaml
    domain_path = f'{home}{sep}dcarte{sep}config{sep}{domain}.yaml'
    if path_exists(domain_path):
        Path(domain_path).unlink()
    # delete domain data folder
    local_domain_folder = f'{data_folder}{sep}{domain}{sep}'
    if path_exists(local_domain_folder):
        delete_folder(Path(local_domain_folder))
    # delete domain info from config file
    config_path = f"{home}{sep}dcarte{sep}config.yaml"
    config_yaml = load_yaml(config_path)
    domains = pd.DataFrame(config_yaml['domains'])
    domains = domains.drop(domains.query('domain == @domain').index)
    config_yaml['domains'] = domains.to_dict('records')
    write_yaml(config_path, config_yaml)