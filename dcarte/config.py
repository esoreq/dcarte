from pathlib import Path
import shutil
import uuid
import os
import webbrowser
import dcarte
import getpass
from .utils import (load_yaml,
                    write_yaml,
                    update_yaml,
                    path_exists)


def get_config(config_file : str = '/dcarte/config.yaml',
               root: Path = Path('__file__').parent.absolute(),
               home: Path = Path('~').expanduser(),
               dcarte_home: Path =  Path(dcarte.__file__).parent.absolute()) -> dict:
    """get_config a function that returns or creates and returns a local config file


    Args:
        config_file (str, optional): [description]. Defaults to '/dcarte/config.yaml'.
        root (Path, optional): [description]. Defaults to Path('__file__').parent.absolute().
        home (Path, optional): [description]. Defaults to Path('~').expanduser().

    Returns:
        [dict]: containing all the configuration information neeeded for dcarte
    """
    if path_exists(str(home)+config_file):
        cfg = load_yaml(str(home)+config_file)
    else:
        cfg =  create_config(home, root, dcarte_home)
    os.environ['MINDER_TOKEN'] = cfg['token']
    cfg.pop('token', None)
    return cfg


def update_config(new_dict:dict, home:Path = Path('~').expanduser()):
    """update_config updates the central config file with data from new_dict

    Args:
        new_dict (dict): [description]
        home (Path, optional): [description]. Defaults to Path('~').expanduser().
    """
    update_yaml(f"{home}/dcarte/config.yaml", new_dict)


def create_config(home:Path,root:Path, dcarte_home:Path):
    """create_config creates a baseline config file

    Args:
        home (Path): [description]
        root (Path): [description]

    Returns:
        [type]: [description]
    """
    # Create dcarte folder at home to store config and data folders
    for p in ["/dcarte/config", "/dcarte/data"]:
        Path(f"{home}{p}").mkdir(parents=True, exist_ok=True)
    # copy yaml files from source_yaml to home/config
    for p in Path(dcarte_home).rglob('source_yaml'):
        if p.is_dir():
            source_yaml = p.resolve()
        else:
            raise Exception("Sorry, unable to copy base config yaml files")               
    try:        
        files = list(Path(source_yaml).glob('*.yaml'))
        [shutil.copy2(f'{file}', f'{home}/dcarte/config') for file in files]
    except:
        raise Exception("Sorry, unable to copy base config yaml files")           
    # create a baseline config dict
    cfg = baseline_config(home,root,files)
    # open webpage and request user to copy token
    cfg['token'] = get_token()
    cfg['mac'] = get_mac()
    write_yaml(f"{home}/dcarte/config.yaml", cfg)
 
    return cfg
    

def get_mac() -> str:
    """get_mac return mac address of the compute node or computer

    Returns:
        str: [description]
    """
    return hex(uuid.getnode())


def get_token() -> str:
    """get_token opens the access-tokens website to create a unique REST token 

    Returns:
        str: a token generated at https://research.minder.care/portal/access-tokens
    """
    webbrowser.open('https://research.minder.care/portal/access-tokens')
    print('Please go to https://research.minder.care/portal/access-tokens to generate a token and copy it into the input bar')
    token = getpass.getpass(prompt='Token: ')
    return token
    

def baseline_config(home: Path, root: Path, files:list) -> dict:
    """baseline_config create a baseline config dict 

    Args:
        home (Path): [description]
        root (Path): [description]

    Returns:
        dict: [description]
    """
        
    dataset_yamels = {file.stem:load_yaml(file) for file in files}
    datasets = [{'domain':domain, 'dataset': dataset} for domain, d in dataset_yamels.items() for dataset in d.keys()]
    headers = {'Accept': 'text/plain','Content-type': 'application/json'}
    cfg = {
        'compression': 'GZIP',
        'data_folder': f"{home}/dcarte/data",
        'domains': datasets,
        'headers': headers,
        'home': f'{home}',
        'root': f'{root}',
        'server': 'https://research.minder.care/api/export'
    }
    return cfg
    
