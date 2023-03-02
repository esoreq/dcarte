from dataclasses import dataclass, field
from typing import Optional,List
import datetime as dt
import json
import os
import logging
import numpy as np
from pathlib import Path
import pandas as pd
from time import sleep
from io import StringIO
import requests
from .config import get_config,update_token
from .utils import (write_table,
                   read_table,
                   read_metadata,
                   date2iso,
                   BearerAuth,
                   path_exists,
                   set_path)

# if isnotebook():
#     from tqdm.notebook import tqdm
# else:
from tqdm import tqdm
sep = os.sep
cfg = get_config()
NOW = date2iso(str(dt.datetime.now()))
class MinderException(Exception):
    pass

@dataclass
class MinderDataset(object):
    """
    MinderDataset class handles the downloading of datasets from the minder research platform.

    Args:
        dataset_name (str): Name of the dataset.
        datasets (list): List of dataset IDs to download.
        columns (list): List of columns to include in the downloaded data.
        domain (str): Domain ID of the dataset.
        dtypes (list): List of data types for each column specified in `columns`.
        since (str, optional): Start date for the dataset in ISO format (yyyy-mm-dd). Defaults to '2019-04-01'.
        until (str, optional): End date for the dataset in ISO format (yyyy-mm-dd). Defaults to NOW.
        log_level (str, optional): Level of logging. Defaults to 'DEBUG'.
        delay (float, optional): Delay between requests in hours. Defaults to 1.
        headers (dict, optional): HTTP headers to use for requests. Defaults to `cfg['headers']`.
        server (str, optional): URL of the minder research platform. Defaults to `cfg['server']`.
        home (Path, optional): Path to the directory for the downloaded data. Defaults to `cfg['home']`.
        compression (str, optional): Compression format of the data. Defaults to `cfg['compression']`.
        data_folder (str, optional): Name of the directory to store the downloaded data in `home`. Defaults to `cfg['data_folder']`.
        data (pd.DataFrame, optional): Dataframe containing the downloaded data. Defaults to an empty dataframe.
        request_id (str, optional): ID of the request for the dataset. Defaults to ''.
        reload (bool, optional): If True, re-downloads the dataset. Defaults to False.
        reapply (bool, optional): If True, reads the local file rather than downloading the dataset. Defaults to False.
        update (bool, optional): If True, updates the dataset with new data. Defaults to False.

    Raises:
        MinderException: Raised when there is an issue with the dataset request or response.

    Returns:
        MinderDataset: Instance of the MinderDataset class.

    Examples:
        To download a dataset from the minder research platform:

        >>> dataset = MinderDataset(
        ...     dataset_name='vitals',
        ...     datasets=['raw_body_weight', 'raw_heart_rate','raw_body_temperature],
        ...     columns=['home_id', 'patient_id', 'start_date', 'value', 'unit','device_type'],
        ...     domain='raw',
        ...     dtypes=['category', 'category', 'datetime64[ns]', 'float', 'category', 'category'],
        ...     since='2022-01-01',
        ...     until='2022-02-01'
        ... )

        This will download the 'vitals' dataset with columns ['home_id', 'patient_id', 'start_date', 'value', 'unit','device_type'], into the 'raw' domain,
        and containing data from January 1, 2022 to February 1, 2022.
    """
    dataset_name: str
    datasets: list
    columns: list
    domain: str
    dtypes: list
    since: str = '2019-04-01'
    until: str = NOW
    log_level: str = 'DEBUG'
    delay: float = 1
    organizations: Optional[List[str]] = None
    headers: dict = field(default_factory=lambda: cfg['headers'])
    server: str = cfg['server']
    home: Path = cfg['home']
    compression: str = cfg['compression']
    data_folder: str = cfg['data_folder']
    data: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    request_id: str = ''
    reload: bool = False
    reapply: bool = False
    update: bool = False

    def __post_init__(self):
        self._delay = dt.timedelta(hours=self.delay)
        self.since = date2iso(self.since)
        self.until = date2iso(self.until)
        self.auth = BearerAuth(os.getenv('MINDER_TOKEN'))
        self.data_request = {'since': self.since,
                             'until': self.until,
                             'datasets': {ds: {"columns": self.columns}
                                          for ds in self.datasets}}
        if self.organizations is not None:
            self.data_request.update({'organizations':self.organizations})
        self.local_file = (f'{self.data_folder}{sep}'
                           f'{self.domain}{sep}'
                           f'{self.dataset_name}.parquet')
        if self.log_level == 'DEBUG':
            logging.basicConfig(
                filename=cfg['log_output'], 
                level=logging.DEBUG,
                format='%(asctime)s %(levelname)-8s %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')
        if not path_exists(self.local_file) or self.reload:
            set_path(self.local_file)
            self.download_dataset()               
        elif self.reapply:
            self.data = read_table(self.local_file)
        elif self.update:
            self.update_dataset()
        else:
            self.data = read_table(self.local_file)
    
    def download_dataset(self):
        self.post_request()
        self.process_request()
        self.download_data()
        self.update_metadata()
        self.save_dataset()

    def post_request(self):
        try:
            request = requests.post(self.server,
                                    data=json.dumps(self.data_request),
                                    headers=self.headers,
                                    auth=self.auth)
            if request.status_code == 403:
                raise MinderException("You need an active VPN connection") 
            elif request.status_code == 401:    
                if update_token():
                    self.auth = BearerAuth(os.getenv('MINDER_TOKEN'))
                    self.post_request()
                else:
                    raise MinderException('There is a problem with your Token') 
            elif request.status_code != 403:
                self.request_id = (request.headers['Content-Location']
                                .split('/')[-1])
                logging.debug(f'{self.request_id} posted')
            else:
                raise MinderException("You need an active VPN connection")    
        except BaseException as err:
            logging.debug(f'Unexpected {self.request_id} {err=},{type(err)=}')
            raise       

        

    def process_request(self, sleep_time:int=10):
        print(f'Processing {self.dataset_name} ',end=':')
        request_output = self.get_output()
        while request_output.empty:
            sleep(sleep_time)
            request_output = self.get_output()
        self.csv_url = request_output
        print('')

    def get_output(self):
        try:
            with requests.get(f'{self.server}/{self.request_id}/', auth=self.auth) as request:
                request_elements = pd.DataFrame(request.json())
                output = pd.DataFrame()
                if request_elements.status.iat[0] == 202:
                        print('*',end='')
                elif request_elements.status.iat[0] == 200: 
                    if  'output' in request_elements.index: 
                        output = pd.DataFrame(request_elements.loc['output'].jobRecord)
                        if output.empty:
                            logging.debug(f'{self.request_id} has no info')
                            output = pd.DataFrame([False])
                        else:    
                            logging.debug(f'{self.request_id} recived with info')                        
                else:   
                    logging.debug(f'Unexpected {request_elements.status.iat[0]} status')        
                          
        except BaseException as err:
            logging.debug(f'Unexpected {self.request_id} {err=},{type(err)=}')
            raise
        
        return output
    
    def persistent_download(self,url,idx):
        df = pd.DataFrame()
        while df.empty:
            try:
                with requests.get(url, stream=True, auth=self.auth) as request:
                    decoded_data = StringIO(request.content.decode('utf-8-sig'))
                    df = pd.read_csv(decoded_data, sep=',', engine='python')
                    df['source'] = self.csv_url.type[idx]
            except:
                print(self.request_id) 
            sleep(2)    
        return df 
    
    def download_data(self):
        data = []
        for idx, url in enumerate(tqdm(self.csv_url.url,
                                       desc=f'Downloading {self.dataset_name}',
                                       dynamic_ncols=True)):
            df = self.persistent_download(url,idx)
            data.append(df)
        self.data = pd.concat(data).reset_index(drop=True)
        self.data = self.data[np.any(self.data.values==self.data.columns.values.reshape(1,-1),axis=1)==False]
        dtypes = dict(zip(self.columns, self.dtypes))
        self.data = self.data.replace({'false':0.0,'true':1.0}).astype(dtypes)
        logging.debug(f'{self.request_id} was downloaded')

    def update_metadata(self):
        # if 'start_date' in self.data.columns:
        #     self.metadata = {'since': date2iso(str(self.data['start_date'].min())),
        #                     'until': date2iso(str(self.data['start_date'].max())),
        #                     "request_id": self.request_id,
        #                     'Mac': cfg['mac']}
        # else:   
        self.metadata = {'since': self.since,
                         'until': self.until,
                         "request_id": self.request_id,
                         'Mac': cfg['mac']}

    def load_metadata(self):
        hdr = read_metadata(self.local_file)
        metadata = json.loads(hdr.metadata[b'minder'].decode())
        return metadata

    def append_dataset(self):
        # TODO: fix the append to file function
        dtypes = dict(zip(self.columns, self.dtypes))
        _data = read_table(self.local_file).replace({'false':0.0,'true':1.0}).astype(dtypes)
        write_table(pd.concat([_data, self.data]).reset_index(drop=True),
                    self.local_file,
                    self.compression,
                    self.metadata)

    def save_dataset(self):
        dtypes = dict(zip(self.columns, self.dtypes))
        try:
            write_table(self.data.astype(dtypes),
                    self.local_file,
                    self.compression,
                    self.metadata)
        except:
            print ('save failed')            

    def update_dataset(self):
        hdr = read_metadata(self.local_file)
        metadata = json.loads(hdr.metadata[b'minder'].decode())
        until = pd.to_datetime(metadata['until']).tz_localize(None) + self._delay
        if until < pd.to_datetime(dt.datetime.now()):
            self.data_request['since'] = metadata['until']
            self.data_request['until'] = self.until
            self.post_request()
            self.process_request()
            if 'url' in self.csv_url.columns:
                self.download_data()
                self.update_metadata()
                self.append_dataset()    
            self.load_dataset()
        else:
            self.load_dataset()

    def load_dataset(self):
        self.data = read_table(self.local_file)
