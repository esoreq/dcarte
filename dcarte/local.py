from dataclasses import dataclass, field
import datetime as dt
import json
import os
import sys
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
    module: str = 'base'
    dependencies: list = field(default_factory=lambda: [])
    since: str = '2019-04-01'
    until: str = NOW
    delay: float = 1
    reapply: bool = False
    reload: bool = False
    update: bool = False
    home: Path = cfg['home']
    compression: str = cfg['compression']
    data_folder: str = cfg['data_folder']
    data: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())

    def __post_init__(self):
        """__post_init__ [summary]

        [extended_summary]
        """
        self.dataset_name = self.dataset_name.lower()
        self.domain = self.domain.lower()
        self._delay = dt.timedelta(hours=self.delay)
        self.since = date2iso(self.since)         
        module_path = list(Path().rglob(self.module+'.py'))
        module_dir = os.path.dirname(module_path[0])
        if module_dir not in sys.path:
            sys.path.append(module_dir)
        spec = importlib.util.spec_from_file_location(self.module, module_path[0])
        self._module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._module)
        self.local_file = (f'{self.data_folder}/'
                           f'{self.domain}/'
                           f'{self.dataset_name}.parquet')
        self.metadata = {'since': self.since,
                         'until': self.until,
                         'Mac': cfg['mac']}
        self.load_dataset()
        self.data = read_table(self.local_file)

    def load_dataset(self):
        """load_dataset [summary]

        [extended_summary]
        """
        if not path_exists(self.local_file) or self.reload:
            set_path(self.local_file)
            self.process_dataset()
        elif self.update:
            self.update_dataset()
        elif self.reapply:
            self.reapply_dataset()    
        else:
            self.data = read_table(self.local_file)    

    def reapply_dataset(self):
        """reapply_dataset [summary]

        [extended_summary]
        """
        for func in self.pipeline:
            self.data = getattr(self._module, func)(self)
        self.save_dataset()
    
    def process_dataset(self):
        """process_dataset [summary]

        [extended_summary]
        """
        for func in self.pipeline:
            self.data = getattr(self._module, func)(self)
        domains = pd.DataFrame(cfg['domains'])
        dataset = np.array([self.domain,self.dataset_name])
        dataset_exist = (domains == dataset).all(axis=1).any()
        if not dataset_exist:
            self.register_dataset()
        self.save_dataset()    

    def update_dataset(self):
        """update_dataset [summary]

        [extended_summary]
        """
        hdr = read_metadata(self.local_file)
        metadata = json.loads(hdr.metadata[b'minder'].decode())
        until = pd.to_datetime(metadata['until']).tz_localize(None) + self._delay
        if until < pd.to_datetime(dt.datetime.now()):
            self.update_metadata()
            self.process_dataset()
            
            
    def update_metadata(self):
        """update_metadata [summary]

        [extended_summary]
        """
        if path_exists(self.local_file):
            since = self.load_metadata()['since']
        else:
            since = self.since
        self.metadata = {'since': since,
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
        cfg['domains'].append({'domain':self.domain.lower(),'dataset':self.dataset_name.lower()})
        cfg['domains'] = pd.DataFrame(cfg['domains']).drop_duplicates().to_dict('records')
        update_config(cfg)
        home = cfg['home']
        dependencies = (pd.DataFrame(self.dependencies).
                          iloc[:,::-1].
                          rename(columns={1:'domain',0:'dataset'}).
                          to_dict(orient='records'))
        collection_file = f'{home}/dcarte/config/{self.domain}.yaml'
        data = {self.dataset_name: {"domains": dependencies,
                                    "pipeline": self.pipeline,
                                    "module": self.module}}
        if path_exists(collection_file):
            update_yaml(collection_file, data)
        else:
            write_yaml(collection_file, data)

