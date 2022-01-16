from dataclasses import dataclass, field
import datetime as dt
import json
import os
from pathlib import Path
import pandas as pd
from time import sleep
from io import StringIO
import requests
from .config import get_config
from .utils import (write_table,
                   read_table,
                   read_metadata,
                   isnotebook,
                   date2iso,
                   BearerAuth,
                   path_exists,
                   timer,
                   set_path)

# if isnotebook():
#     from tqdm.notebook import tqdm
# else:
from tqdm import tqdm

cfg = get_config()
NOW = date2iso(str(dt.datetime.now()))

@dataclass
class MinderDataset(object):
    """MinderDataset class handles the downloading of datasets from the minder reserch platform

    [extended_summary]

    Args:
        dataset_name ([str]): [description]
        datasets ([list]): [description]
        columns ([list]): [description]
        domain ([str]): [description]
        dtypes ([list]): [description]
        since ([list]): [description]
        until ([list]): [description]
        delay ([list]): [description]
        auth ([list]): [description]
        headers ([list]): [description]
        server ([list]): [description]
        token ([list]): [description]
        compression ([list]): [description]
        data_folder ([list]): [description]
        data ([list]): [description]
        request_id ([list]): [description]
        reload ([list]): [description]
        reapply ([list]): [description]
        update ([list]): [description]

    Raises:
        Exception: [description]

    Returns:
        [type]: [description]
    """
    dataset_name: str
    datasets: list
    columns: list
    domain: str
    dtypes: list
    since: str = '2019-04-01'
    until: str = NOW
    delay: float = 1
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
        self.local_file = (f'{self.data_folder}/'
                           f'{self.domain}/'
                           f'{self.dataset_name}.parquet')

        if not path_exists(self.local_file) or self.reload:
            set_path(self.local_file)
            self.download_dataset()
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
        request = requests.post(self.server,
                                data=json.dumps(self.data_request),
                                headers=self.headers,
                                auth=self.auth)
        if request.status_code != 403:
            self.request_id = (request.headers['Content-Location']
                               .split('/')[-1])
        else:
            raise Exception("You need an active VPN connection")

    def process_request(self, sleep_time:int=10):
        request_output = self.get_output()
        print(f'Processing {self.dataset_name} ',end=':')
        while request_output.empty:
            sleep(sleep_time)
            request_output = self.get_output()
        self.csv_url = request_output

    def get_output(self):
        request = requests.get(self.server, auth=self.auth)
        request_elements = pd.DataFrame(request.json())
        request_elements = request_elements[request_elements.id ==
                                            self.request_id]
        output = pd.DataFrame()
        if not request_elements.empty:
            if request_elements.jobRecord.notnull().iat[0]:
                output = pd.DataFrame(
                    request_elements.jobRecord.values[0]['output'])
            if request_elements.status.iat[0] == 202:
                print('*',end='')
        return output
    
    def download_data(self):
        data = []
        for idx, url in enumerate(tqdm(self.csv_url.url,
                                       desc=f'Downloading {self.dataset_name}',
                                       dynamic_ncols=True)):
            request = requests.get(url, stream=True, auth=self.auth)
            decoded_data = StringIO(request.content.decode('utf-8-sig'))
            df = pd.read_csv(decoded_data, sep=',', engine='python')
            df['source'] = self.csv_url.type[idx]
            data.append(df)
        self.data = pd.concat(data).reset_index(drop=True)
        if (self.data[self.columns[0]] == self.columns[0]).any():
            self.data = self.data[self.data[self.columns[0]]
                                  != self.columns[0]]
        dtypes = dict(zip(self.columns, self.dtypes))
        self.data = self.data.replace({'false':0.0,'true':1.0}).astype(dtypes)

    def update_metadata(self):
        if path_exists(self.local_file):
            since = self.load_metadata()['since']
        else:
            since = self.since
        self.metadata = {'since': since,
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
        write_table(self.data.astype(dtypes),
                    self.local_file,
                    self.compression,
                    self.metadata)

    def update_dataset(self):
        hdr = read_metadata(self.local_file)
        metadata = json.loads(hdr.metadata[b'minder'].decode())
        until = pd.to_datetime(metadata['until']).tz_localize(None) + self._delay
        if until < pd.to_datetime(dt.datetime.now()):
            self.data_request['since'] = metadata['until']
            self.data_request['until'] = self.until
            self.post_request()
            self.process_request()
            self.download_data()
            self.update_metadata()
            self.append_dataset()
            self.load_dataset()
        else:
            self.load_dataset()

    def load_dataset(self):
        self.data = read_table(self.local_file)
