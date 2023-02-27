from pathlib import Path
import numpy as np
import pandas as pd
from typing import List,Union,Dict,Any

import logging
import shutil
import uuid
import filecmp
import os
import webbrowser
import dcarte
import getpass
from .utils import load_yaml, write_yaml, update_yaml, merge_dicts, path_exists
from ._delete import delete_dataset_

sep = os.sep


def get_config(
    config_file: str = f"{sep}dcarte{sep}config.yaml",
    root: Path = Path("__file__").parent.absolute(),
    home: Path = Path("~").expanduser(),
    dcarte_home: Path = Path(dcarte.__file__).parent.absolute(),
) -> dict:
    """get_config a function that returns or creates and returns a local config file

        The get_config function returns or creates and returns a local configuration file for the dcarte package. It takes four arguments:

        config_file: a string representing the path to the configuration file, with a default value of "/dcarte/config.yaml".
        root: a Path object representing the root directory of the package, with a default value of the parent directory of the __file__ attribute.
        home: a Path object representing the home directory of the user, with a default value of the user's home directory.
        dcarte_home: a Path object representing the directory of the dcarte package, with a default value of the parent directory of the dcarte module file.
        The function returns a dictionary containing all the configuration information needed for dcarte. If the configuration file does not exist, the function creates it.

        The function first checks if the configuration file exists in the home directory. If it does, it checks if the configuration is valid by calling the check_config function. If the configuration is not valid, it calls the create_config function to create a new configuration file. The function then checks if there are any updated YAML versions in the dcarte package directory and compares them to the existing ones in the home directory. The function loads the main YAML configuration file and checks if it reflects all the datasets in the home directory. If not, it reconstructs the domains. Finally, the function sets the MINDER_TOKEN environment variable to the value in the configuration file, removes the token key from the configuration dictionary, and returns the configuration dictionary.

    Args:
        config_file (str, optional): [description]. Defaults to '/dcarte/config.yaml'.
        root (Path, optional): [description]. Defaults to Path('__file__').parent.absolute().
        home (Path, optional): [description]. Defaults to Path('~').expanduser().

    Returns:
        [dict]: containing all the configuration information needed for dcarte
    """
    if path_exists(str(home) + config_file):
        if not check_config(home, dcarte_home):
            create_config(home, root, dcarte_home, False)
        # check if any updated yaml version exists in the toolbox folder
        source_yaml = get_source_yaml(dcarte_home)
        compare_source_yaml(home, source_yaml)
        # load the main config yaml file
        cfg = load_yaml(str(home) + config_file)
        # Check if cfg file reflects all the datasets in home
        check_data_files()
        reset_config()
        files = list(Path(f"{home}{sep}dcarte{sep}config{sep}").glob("*.yaml"))
        domains = pd.DataFrame(cfg["domains"]).domain.unique()
        if domains.shape[0] != len(files):
            reconstruct_domains(files, cfg)

    else:
        cfg = create_config(home, root, dcarte_home)
    os.environ["MINDER_TOKEN"] = cfg["token"]
    cfg.pop("token", None)
    return cfg


def reconstruct_domains(files, cfg):
    ''' 
    Create a new domain in the configuration file for each YAML file that is not already represented in the configuration file.

    The function creates a new domain in the configuration file for each YAML file that is not already represented in the configuration file. It loads the YAML file using the load_yaml function, creates a DataFrame from the keys in the YAML file, renames the column to "dataset", and adds a new column called "domain" with the value of the YAML file's stem attribute. It then concatenates this DataFrame with the domains DataFrame in the configuration file, drops any duplicates, and saves the resulting DataFrame back into the cfg dictionary. Finally, it saves the cfg dictionary to the configuration file using the write_yaml function.
    The purpose of this function is to ensure that the domains list in the configuration file contains all of the domains for which there are YAML files in the dcarte directory. This is necessary because each YAML file represents a dataset, and each dataset belongs to a domain. The domains list is used elsewhere
    
    Parameters
    ----------
    files : list
        A list of `Path` objects representing the YAML files.
    cfg : dict
        A dictionary containing the configuration information.

    Returns
    -------
    None
    '''
    domains = pd.DataFrame(cfg["domains"])
    for file in files:
        if not file.stem in domains.domain.unique():
            tmp = load_yaml(file)
            tmp = (
                pd.Series(tmp.keys())
                .rename("dataset")
                .to_frame()
                .assign(domain=file.stem)
            )
            domains = pd.concat([domains, tmp])
    cfg["domains"] = domains.drop_duplicates().to_dict("records")
    config_file = f"{cfg['home']}{sep}dcarte{sep}config.yaml"
    write_yaml(config_file, cfg)
    domains = pd.DataFrame(cfg["domains"])
    for file in files:
        if not file.stem in domains.domain.unique():
            tmp = load_yaml(file)
            tmp = (
                pd.Series(tmp.keys())
                .rename("dataset")
                .to_frame()
                .assign(domain=file.stem)
            )
            domains = pd.concat([domains, tmp])
    cfg["domains"] = domains.drop_duplicates().to_dict("records")
    config_file = f"{cfg['home']}{sep}dcarte{sep}config.yaml"
    write_yaml(config_file, cfg)


def check_config(
    home: Path = Path("~").expanduser(),
    dcarte_home: Path = Path(dcarte.__file__).parent.absolute(),
):
    """
    Check if the necessary directories and files for dcarte exist, and create them if necessary.

    The function itself checks if the necessary directories and files for dcarte exist, and creates them if necessary. It returns a boolean indicating whether all necessary directories and files exist or not.

    Parameters
    ----------
    home : Path object, optional
        The home directory for dcarte. Default is the user's home directory.
    dcarte_home : Path object, optional
        The root directory for dcarte. Default is the parent directory of the dcarte package.

    Returns
    -------
    bool
        Returns True if all necessary directories and files exist, False otherwise.
    """
    # go over the four directories and check that they exist
    checks = np.ones((5,))
    for i, p in enumerate(["config", "data", "log", "recipes"]):
        target = f"{home}{sep}dcarte{sep}{p}"
        if not Path(target).is_dir():
            Path(target).mkdir(parents=True, exist_ok=True)
            checks[i] = 0
    # if recipes is in data move it outside
    recipes = f"{home}{sep}dcarte{sep}data{sep}recipes"
    if Path(recipes).is_dir():
        target_dir = f"{home}{sep}dcarte{sep}recipes{sep}"
        shutil.copytree(recipes, target_dir, dirs_exist_ok=True)
        shutil.rmtree(recipes)
    if not path_exists(f"{home}{sep}dcarte{sep}log{sep}monitor.log"):
        checks[4] = 0

    return np.all(checks)


def update_config(new_dict: dict, home: Path = Path("~").expanduser()):
    """
    Update the central config file with data from new_dict. If the config file does not exist,
    it creates one with the name 'config.yaml' in the 'dcarte' directory inside the specified home directory.

    Parameters
    ----------
    new_dict : dict
        Dictionary containing the data to update the config file with.
    home : Path, optional
        Absolute path to the root directory of the project. Default is the current user's home directory.

    Returns
    -------
    None
    """
    update_yaml(f"{home}{sep}dcarte{sep}config.yaml", new_dict)


def compare_source_yaml(home: Path, source_yaml: str) -> List[Path]:
    """
    Copies updated yaml files from the toolbox folder to the local config folder.

    Parameters
    ----------
        home (Path): Absolute path to the user's home directory.
        source_yaml (str): Absolute path to the toolbox folder.

    Returns
    -------
        List[Path]: A list of updated yaml files.
    Raises
    -------
        Exception: If there is an error copying the base config yaml files.
    """
    try:
        files = list(Path(source_yaml).glob("*.yaml"))
        for source in files:
            target = f"{home}{sep}dcarte{sep}config{sep}{source.name}"
            if not path_exists(target):
                shutil.copyfile(source, target)
            elif not filecmp.cmp(source, target):
                shutil.copy2(source, target)
    except:
        raise Exception("Sorry, unable to copy base config yaml files")
    return files


def get_source_yaml(dcarte_home: Path) -> Union[Path, None]:
    """
    Find the path to the source_yaml directory in the given dcarte_home directory.

    Parameters
    ----------
    dcarte_home : Path
        The path to the dcarte_home directory.

    Returns
    -------
    Union[Path, None]
        The path to the source_yaml directory, or None if it cannot be found.

    Raises
    ------
    Exception
        If the source_yaml directory cannot be found in the given dcarte_home directory.
    """
    source_yaml = None
    for p in Path(dcarte_home).rglob("source_yaml"):
        if p.is_dir():
            source_yaml = p.resolve()
        else:
            raise Exception("Sorry, unable to find base config yaml folder")
    return source_yaml


def create_config(home: Path, root: Path, dcarte_home: Path, update_token: bool = True) -> dict:
    """Creates the initial configuration for dcarte, including the necessary
    directories, configuration files, and tokens. 
    It also writes the config.yaml file 

    Parameters
    ----------
    home : Path
        The home directory to create the dcarte folder.
    root : Path
        The root directory of the project.
    dcarte_home : Path
        The path to dcarte.
    update_token : bool, optional
        Whether to update the token or not, by default True.

    Returns
    -------
    dict
        The dcarte configuration dictionary.
    """
    # Create dcarte folder at home to store config and data folders

    tmp = {}
    for p in ["config", "data", "log", "recipes"]:
        target = f"{home}{sep}dcarte{sep}{p}"
        Path(target).mkdir(parents=True, exist_ok=True)
        tmp[p] = target
    # copy yaml files from source_yaml to home/config
    source_yaml = get_source_yaml(dcarte_home)
    files = compare_source_yaml(home, source_yaml)
    # create a baseline config dict
    cfg = baseline_config(home, root, files)
    # open webpage and request user to copy token
    config_file = f"{home}{sep}dcarte{sep}config.yaml"
    if update_token:
        cfg["token"] = get_token()
    else:
        cfg["token"] = load_yaml(config_file)["token"]
    cfg["mac"] = get_mac()
    cfg = merge_dicts(cfg, tmp)
    log_output = f"{cfg['log']}{sep}monitor.log"
    cfg["log_output"] = log_output
    write_yaml(config_file, cfg)

    return cfg


def update_token() -> bool:
    """
    Updates the MINDER_TOKEN in the central config file with the new token obtained from the user.
    This function updates the config.yaml file with the new token and sets the token in the environment variable MINDER_TOKEN.

    Returns
    -------
        bool: True if the token was updated successfully, False otherwise.
    """
    cfg = get_config()
    cfg["token"] = get_token()
    write_yaml(f"{cfg['home']}{sep}dcarte{sep}config.yaml", cfg)
    os.environ["MINDER_TOKEN"] = cfg["token"]
    cfg.pop("token", None)
    return True


def get_mac() -> str:
    """get_mac return mac address of the compute node or computer

    Returns
    -------
        str: [description]
    """
    return hex(uuid.getnode())


def get_token() -> str:
    """get_token opens the access-tokens website to create a unique REST token

    Returns
    -------
        str: a token generated at https://research.minder.care/portal/access-tokens
    """
    webbrowser.open("https://research.minder.care/portal/access-tokens")
    print(
        "Please go to https://research.minder.care/portal/access-tokens to generate a token and copy it into the input bar"
    )
    token = getpass.getpass(prompt="Token: ")
    return token


def baseline_config(home: Path, root: Path, files: List[Path]) -> Dict[str, Any]:
    """baseline_config create a baseline config dict

   Create a baseline configuration dictionary.

    Parameters
    ----------
    home : Path
        Home directory path.
    root : Path
        Root directory path.
    files : List[Path]
        List of YAML configuration file paths.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing the baseline configuration.
    """

    dataset_yamels = {file.stem: load_yaml(file) for file in files}
    datasets = [
        {"domain": domain, "dataset": dataset}
        for domain, d in dataset_yamels.items()
        for dataset in d.keys()
    ]
    headers = {
        "Accept": "text/plain",
        "Content-type": "application/json",
        "Connection": "keep-alive",
        "X-Azure-DebugInfo": "1",
    }
    cfg = {
        "compression": "GZIP",
        "data_folder": f"{home}{sep}dcarte{sep}data",
        "domains": datasets,
        "headers": headers,
        "home": f"{home}",
        "root": f"{root}",
        "server": "https://research.minder.care/api/export",
    }
    return cfg


def reset_config():
    root = Path("__file__").parent.absolute()
    home = Path("~").expanduser()
    config_file = f"{home}{sep}dcarte{sep}config.yaml"
    dcarte_home = Path(dcarte.__file__).parent.absolute()
    cfg = load_yaml(config_file)
    source_yaml = get_source_yaml(dcarte_home)
    compare_source_yaml(home, source_yaml)
    files = list(Path(f"{home}{sep}dcarte{sep}config{sep}").glob("*.yaml"))
    domains = [ ]
    for file in files: 
        tmp = load_yaml(file)
        domain = file.stem
        datasets = tmp.keys()
        domains.append(pd.Series(datasets,name='dataset').to_frame().assign(domain = domain))
    domains = pd.concat(domains)    
    cfg['domains'] = domains.to_dict("records")
    write_yaml(config_file,cfg)  
    


def check_data_files():
    """Check data files and update config file domains to reflect the ones in the domain yaml files.

    Returns
    -------
        None
    """
    home: Path = Path("~").expanduser()
    config_file = f"{home}{sep}dcarte{sep}config.yaml"
    if path_exists(config_file):
        cfg = load_yaml(config_file)
        home = cfg["home"]
        domains = pd.DataFrame(cfg["domains"])
        datasets_paths = list(
            Path(f"{home}{sep}dcarte{sep}data{sep}").glob("*/*.parquet")
        )

    for dataset_path in datasets_paths:
        dataset = dataset_path.stem
        domain = dataset_path.parent.stem
        if domains.query("domain == @domain and dataset == @dataset").empty:
            delete_dataset_(cfg, dataset, domain)
