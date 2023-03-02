from dataclasses import dataclass, field
import datetime as dt
import json
import os
import sys
import filecmp
import shutil
from pathlib import Path
from .config import get_config, update_config
import pandas as pd
import numpy as np
import importlib.util
import yaml
from .utils import (write_table,
                   read_table,
                   read_metadata,
                   write_yaml,
                   path_exists,
                   update_yaml,
                   date2iso,
                   set_path)
import importlib
sep = os.sep
cfg = get_config()
NOW = date2iso(str(dt.datetime.now()))
# BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class LocalDataset(object):
    """LocalDataset [summary]

    [extended_summary]

    Args:
        dataset_name ([str]): [description]
        datasets ([list]): [description]
        pipeline ([list]): [description]
        domain ([str]): [description]
        module ([list]): [description]
        dependencies ([list]): [description]
        since ([list]): [description]
        until ([list]): [description]
        delay ([list]): [description]
        reload ([list]): [description]
        reapply ([list]): [description]
        update ([list]): [description]
        home ([Path]): [description]
        compression ([str]): [description]
        data_folder ([str]): [description]
        data ([pd.DataFrame]): [description]

    Returns:
        [type]: [description]
    """
    dataset_name: str
    datasets: dict
    pipeline: list
    domain:  str
    module_path: str = ''
    module: str = 'base'
    dependencies: list = field(default_factory=lambda: [])
    since: str = '2019-04-01'
    last_update: str = None
    until: str = NOW
    delay: float = 1
    reapply: bool = False
    reload: bool = False
    update: bool = False
    home: Path = cfg['home']
    compression: str = cfg['compression']
    data_folder: str = cfg['data_folder']
    data: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    local_data: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())

    def __post_init__(self):
        """__post_init__ [summary]

        [extended_summary]
        """
        self.dataset_name = self.dataset_name.lower()
        self.domain = self.domain.lower()
        self._delay = dt.timedelta(hours=self.delay)
        self.since = date2iso(self.since) 
        self.check_recipe()        
        self.local_file = (f'{self.data_folder}{sep}'
                           f'{self.domain}{sep}'
                           f'{self.dataset_name}.parquet')
        self.metadata = {'since': self.since,
                         'until': self.until,
                         'Mac': cfg['mac']}
        self.register_dataset()
        self.load_dataset()
        self.data = read_table(self.local_file)

    def check_recipe(self):
        if len(self.module_path)>0:
            module_path = self.module_path
        else:
            module_path = f'{cfg["recipes"]}{sep}{self.domain}{sep}{self.module}.py'
        recipe_path = f'{cfg["recipes"]}{sep}{self.domain}{sep}{self.module}.py' 
        if path_exists(recipe_path) and path_exists(module_path):
            # compare both recipies if a local one exists and copy over if they are different 
            same = filecmp.cmp(recipe_path, module_path, shallow=False)
            if not same:
                shutil.copyfile(module_path, recipe_path)
            else: 
                module_path = recipe_path
        elif path_exists(module_path) and not path_exists(recipe_path):
            set_path(recipe_path)
            shutil.copyfile(module_path, recipe_path)
        else:
            raise ValueError('Module not found')
        module_dir = os.path.dirname(recipe_path)
        if module_dir not in sys.path:
            sys.path.append(module_dir)
        spec = importlib.util.spec_from_file_location(self.module, recipe_path)
        self._module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._module)
        
    
    def load_dataset(self):
        """load_dataset [summary]

        [extended_summary]
        """
        if not path_exists(self.local_file) or self.reload or self.reapply:
            set_path(self.local_file)
            self.process_dataset()
            self.save_dataset()
        elif self.update:
            self.update_dataset()
            self.save_dataset()
        else:
            self.data = read_table(self.local_file)    
              

    def reapply_dataset(self):
        """reapply_dataset [summary]

        [extended_summary]
        """
        for func in self.pipeline:
            self.data = getattr(self._module, func)(self)
    
    def process_dataset(self):
        """process_dataset [summary]

        [extended_summary]
        """
        for func in self.pipeline:
            self.data = getattr(self._module, func)(self)

  

    def update_dataset(self):
        """update_dataset [summary]

        [extended_summary]
        """
        updated_data = False
        hdr = read_metadata(self.local_file)
        metadata = json.loads(hdr.metadata[b'minder'].decode())
        until = pd.to_datetime(metadata['until']).tz_localize(None) + self._delay
        self.data = read_table(self.local_file)
        if 'start_date' in self.data.columns:
            self.local_data = self.data.copy()
            self.last_update = self.local_data.start_date.max()
            for name,dataset in self.datasets.items():
                if 'start_date' in dataset.columns:
                    tmp = dataset.query('start_date > @self.last_update').copy()
                    if not tmp.empty:
                        self.datasets[name] = tmp
                        updated_data = True

    
        if until < pd.to_datetime(dt.datetime.now()) and updated_data:
            self.update_metadata()
            self.process_dataset()
        if not self.local_data.empty and not self.data.empty:
            self.data = pd.concat([self.local_data, self.data]).drop_duplicates()
        else:
            self.data =  self.local_data
    
            
            
    def update_metadata(self):
        """update_metadata [summary]

        [extended_summary]
        """
        self.metadata = {'since': self.since,
                         'until': self.until,
                         'Mac': cfg['mac']}            
    
    def load_metadata(self):
        """load_metadata [summary]

        [extended_summary]

        Returns:
            [type]: [description]
        """
        hdr = read_metadata(self.local_file)
        metadata = json.loads(hdr.metadata[b'minder'].decode())
        return metadata
    
            
    def save_dataset(self) -> None:
        """save_dataset [summary]

        [extended_summary]
        """
        write_table(self.data,
                    self.local_file,
                    self.compression,
                    self.metadata)

    def register_dataset(self) -> None:
        """register_dataset [summary]

        [extended_summary]
        """
        domains = cfg['domains']
        domains.append({'domain':self.domain.lower(),'dataset':self.dataset_name.lower()})
        domains = pd.DataFrame(domains).drop_duplicates().to_dict('records')
        cfg['domains'] = domains
        update_config(cfg)
        home = cfg['home']
        dependencies = (pd.DataFrame(self.dependencies).
                          iloc[:,::-1].
                          rename(columns={1:'domain',0:'dataset'}).
                          to_dict(orient='records'))
        collection_file = f'{home}{sep}dcarte{sep}config{sep}{self.domain}.yaml'
        data = {self.dataset_name: {"domains": dependencies,
                                    "pipeline": self.pipeline,
                                    "module": self.module}}
        if path_exists(collection_file):
            update_yaml(collection_file, data)
        else:
            write_yaml(collection_file, data)

