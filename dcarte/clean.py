from dataclasses import dataclass
import numpy as np
from numpy.lib.shape_base import column_stack
import pandas as pd
import time
import zipfile
import os
from pathlib import Path
import _pickle as cPickle
import datetime
from datetime import timedelta
import itertools
from functools import wraps


def timer(text=None, pre_text=None, post_text=None):
    def wrapper(f):
        @wraps(f)
        def wrapped(other, *f_args, **f_kwargs):
            if other.verbose:
                _start = time.perf_counter()
            out = f(other, *f_args, **f_kwargs)
            if other.verbose:
                elapsed = time.perf_counter() - _start
                if pre_text != None:
                    print(pre_text)
                st = f"Finished {text} in:"
                ed = f"{np.round(elapsed, 1)}"
                print(f"{st:<60}{ed:>10} {'seconds':<10}")
                if post_text != None:
                    print(post_text)
            return out
        return wrapped
    return wrapper

class Base:

    def __init__(self):
        self.pickle_file = ""

    def set_attributes(self):
        self.domains = ["flags",
                        "sleep",
                        "physiology",
                        "wellbeing",
                        "location",
                        "demographics",
                        "transitions",
                        "movement",
                        "activity",
                        "clinical",
                        "doors",
                        "appliances",
                        "device_type",
                        "temperature",
                        "light"]
        for domain in self.domains:
            setattr(self, domain, pd.DataFrame())

    @staticmethod
    def file_date(path_to_file):
        '''
        returns a files time stamp as a datetime object
        '''
        stat = os.stat(path_to_file)
        date = datetime.datetime.fromtimestamp(stat.st_mtime)
        return date.strftime("%Y%m%d")

    @staticmethod
    def file_parts(file):
        '''
        splits a string file to path, file_name, file_type
        '''
        head_tail = os.path.split(file)
        path = head_tail[0]
        file_name, file_type = head_tail[1].split('.')
        return path, file_name, file_type

    @staticmethod
    def remap_cat(_cat, _mapping, df):
        df.loc[:,_cat] = pd.Categorical(df[_cat])
        df.loc[:, _cat] = df[_cat].cat.rename_categories(_mapping)
        return df

    @staticmethod
    def load_csv_from_zip(_zip, csv_file):
        df = pd.read_csv(_zip.open(csv_file),
                         encoding='unicode_escape', low_memory=False)
        return df

    @staticmethod
    def pickle_exist(pickle_file):
        return os.path.exists(pickle_file)

    @staticmethod
    def map(value, key):
        return pd.Series(value, index=key).to_dict()

    def find_duplicate(self, L):
        '''
        identifies duplicates in a list and returns their index
        '''
        seen, duplicate = set(), set()
        index = np.zeros(len(L), dtype=bool)
        seen_add, duplicate_add = seen.add, duplicate.add
        for idx, item in enumerate(L):
            if item in seen:
                duplicate_add(item)
                index[idx] = True
            else:
                seen_add(item)

        return self.ismember(L, list(duplicate)) != None, duplicate

    @staticmethod
    def ismember(a, b):
        '''
        mimic's Matlabs ismemeber function (should be removed in later versions)
        '''
        bind = {}
        for i, elt in enumerate(b):
            if elt not in bind:
                bind[elt] = i
        return np.array([bind.get(itm, None) for itm in a])

    def save_pickle(self):
        if not os.path.exists(self.pickle_file):
            filepath, _, _ = self. file_parts(self.pickle_file)
            Path(filepath).mkdir(parents=True, exist_ok=True)
        output_pickle = open(self.pickle_file, "wb")
        cPickle.dump(self, output_pickle)
        output_pickle.close()

    def load_pickle(self):
        input_pickle = open(self.pickle_file, 'rb')
        data = cPickle.load(input_pickle)
        for domain in self.domains:
            setattr(self, domain, getattr(data, domain))
        input_pickle.close()


@dataclass
class LoadData(Base):
    input_path: str
    output_path: str
    datasets: list
    reload_data: bool = True
    reload_subset: bool = True
    verbose: bool = True

    def __post_init__(self):
        self.set_attributes()
        self.pickle_file = f"{self.output_path}/pkl/merged_tihm.pkl"
        if self.pickle_exist(self.pickle_file) & self.reload_data:
            self.__load_past_file()
        else:
            data_tmp = self.__parse_dataset()
            self.merge_data(data_tmp)
            self.save_merged_file()

    @timer('creating merged file')
    def merge_data(self, data_tmp):
        for domain in self.domains:
            if len(self.datasets) > 1:
                _tmp = pd.concat([getattr(data_tmp[dataset], domain)
                                  for dataset in self.datasets])
                _tmp = _tmp.drop_duplicates().reset_index(drop=True)
            else:
                _tmp = getattr(data_tmp[self.datasets[0]], domain)
            setattr(self, domain, _tmp)

            

    @timer('saving merged file')
    def save_merged_file(self):
        self.save_pickle()

    @timer('loading previously parsed file')
    def __load_past_file(self):
        self.load_pickle()

    @timer('parsing all datasets', "="*80)
    def __parse_dataset(self):
        _tmp = {}
        for dataset in self.datasets:
            if self.verbose:
                print(f'{"="*80}\nParsing DATASET {dataset}\n{"="*80}')
            _tmp[dataset] = PreProcess(name=dataset, input_path=self.input_path,
                                       output_path=self.output_path,
                                       reload_data=self.reload_subset,
                                       verbose=self.verbose)
        return _tmp


@dataclass
class PreProcess(Base):
    name: str
    input_path: str
    output_path: str
    verbose: bool = True
    reload_data: bool = True
    
    # TODO: add a report that asks for the tihm zip files if they are not present  
    def __post_init__(self):
        self.set_attributes()
        self.source_file = str(
            list(Path(self.input_path).rglob(f'tihm{self.name}.zip'))[0])
        self.date = self.file_date(self.source_file)
        self.pickle_file = f"{self.output_path}/pkl/tihm_{self.name}_{self.date}.pkl"

        if self.pickle_exist(self.pickle_file) & self.reload_data:
            self.__load_past_file()
        else:
            self.__parse_domains()

    @timer('loading previously parsed file')
    def __load_past_file(self):
        self.load_pickle()

    def __parse_domains(self):
        _zip = zipfile.ZipFile(self.source_file)
        _pid = pd.read_csv(_zip.open('Patients.csv'))
        _pid.insert(0, 'project', self.name)
        # _mapping = self.map(_pid.sabpId.values, _pid.subjectId.values) 
        self.subjectId = _pid.subjectId.values
        self.demographics = _pid
        self.__parse_observations(_zip)
        self.__parse_flags(_zip)
        self.__parse_wellbeing(_zip)

    @timer('parsing observations')
    def __parse_observations(self, _zip):
        df = self.__load_observation(_zip)
        df = self.__clean_observations(df, _zip)
        self.__parse_location(df)
        self.__parse_activity(df)
        self.__parse_sleep(df)
        self.__parse_physiology(df)
        self.__parse_transitions()
        self.save_pickle()

    @timer('loading observation file')
    def __load_observation(self, _zip):
        df = self.load_csv_from_zip(_zip, 'Observations.csv')
        df.type = pd.Categorical(df.type)
        return df

    @timer('cleaning observations')
    def __clean_observations(self, df, _zip):
        _type = self.load_csv_from_zip(_zip, 'Observation-type.csv')
        self.device_type = _type
        # subset any event that is logged but contains no actual values to a Dataframe under null key
        idx_null = df[["valueBoolean", "valueState", "valueQuantity",
                       "valueDatetimeStart", "valueDatetimeEnd"]].isnull().values.all(axis=1)
        df = df[idx_null == False]
        df['datetimeObserved'] = pd.to_datetime(df['datetimeObserved'])
        # filter out dates before 2014
        df = df[df['datetimeObserved'].dt.year > 2014]
        df = df[df.type != "724061007"]  # filter out device status
        df = df.sort_values(by=['datetimeObserved']).reset_index(drop=True)
        df['activity'] = 1
        df['project_id'] = df.subject
        df['display'] = df.type
        df['device_name'] = df.type.astype('category')
        df = self.remap_cat('device_name', self.map(
            _type.display.values, _type.code.values), df)
        df = self.remap_cat('display', self.map(
            _type.display.values, _type.code.values), df)
        # df = self.remap_cat('subject', _mapping, df)
        df['project'] = self.name
        df['project'] = pd.Categorical(df['project'])
        df['subject'] = pd.Categorical(df['subject'])
        return df

    @timer('seperating observations to different domains')
    def __parse_location(self,  df):
        df = df[df.location.notnull()].drop(columns=["datetimeReceived",
                                                     "provider", "valueUnit", "valueDatetimeStart", "valueDatetimeEnd"])
        self.location = df
        self.appliances = df[df.display == 'Does turn on domestic appliance'][[
            "project", "subject", "datetimeObserved", "location", "activity"]].copy()
        self.movement = df[df.display == 'Movement'][[
            "project", "subject", "datetimeObserved", "location", "activity"]].copy()
        self.temperature = df[(df.valueQuantity.notnull()) & (df.display == "Room temperature")][[
            "project", "subject", "datetimeObserved", "location", "valueQuantity"]].copy()
        self.temperature = self.temperature[~self.temperature.location.isin(
            ["Living Room", "Study"])].copy()
        self.light = df[(df.valueQuantity.notnull()) & (df.display == "Light")][[
            "project", "subject", "datetimeObserved", "location", "valueQuantity"]].copy()
        self.light = self.light[~self.light.location.isin(
            ["Living Room"])].copy()

    @timer('parsing activity')
    def __parse_activity(self,  df):
        # convert movement, doors activity and appliances activity to a cleaned dataframe in day, hour and raw frequencies
        doors = []
        for k, subset in df[df.display == 'Door'].groupby(['subject', 'location', 'project']):
            subset = subset[["datetimeObserved", "valueState"]].pivot(
                columns="valueState", values="datetimeObserved").reset_index(drop=True)
            if 'False' in subset.columns and 'True' in subset.columns:
                subset = subset.rename(
                    columns={'False': 'Close', 'True': "Open"})
            if ('False' in subset.columns) ^ ('True' in subset.columns):
                subset = subset.head(1)
            if subset.shape[0] > 1:
                idx = np.where(subset.Open.isnull())
                open = subset.iloc[idx[0]-1].Open.values
                close = subset.iloc[idx[0]].Close.values
                delta = (close - open).astype('timedelta64[s]')
                m = open.shape[0]
                doors.append(pd.DataFrame({'project': [k[2]]*m, 'subject': [k[0]]*m, 'location': [
                             k[1]]*m, 'datetimeObserved': open, "Close": close, "delta": delta, "activity": [1]*m}))

        doors = pd.concat(doors)
        doors = doors[~doors.location.isin(
            ['B', 'Bathroom', 'C', 'Dining Room'])]
        doors = doors[doors.delta < timedelta(
            minutes=15)].reset_index(drop=True)
        self.doors = doors
        self.appliances = self.appliances[~self.appliances.location.isin([
                                                                         'A', 'B'])].copy()
        self.appliances.loc[self.appliances.location.isin(
            ['Microwave', 'Toaster']), "location"] = 'Oven'
        self.movement = self.movement[~self.movement.location.isin(
            ['D', 'Study', 'Living Room', "Front Door", 'Dining Room'])]
        self.activity = pd.concat([self.doors, self.movement, self.appliances])[
            ['project', 'subject', 'datetimeObserved', 'location']].copy()
        self.activity = pd.get_dummies(self.activity, columns=[
                                       'location'], prefix='', prefix_sep='')

    @timer('parsing sleep')
    def __parse_sleep(self,  df):
        # TODO: load the new sleep data
        # self.sleep_disturbance = df[df.type.isin(['67233009'])].drop(columns=["datetimeReceived", "provider","type","device",
        #                                                                         "location", "valueBoolean", "valueState","provider",
        #                                                                         "device","valueDatetimeStart",
        #                                                                         "valueDatetimeEnd","activity"]).copy()
        df = df[df.type.isin(['258158006', '29373008', '248218005',
                              '60984000', '89129007', '421355008', '307155000'])].copy()
        df['valueDatetimeStart'] = pd.to_datetime(df['valueDatetimeStart'])
        df['valueDatetimeEnd'] = pd.to_datetime(df['valueDatetimeEnd'])
        df['Start_End_logged'] = df.valueDatetimeStart.isnull()
        df = df.reset_index()
        idx = -df.Start_End_logged
        df.loc[idx, "valueQuantity"] = (
            df.loc[idx, "valueDatetimeEnd"] - df.loc[idx, "valueDatetimeStart"]).dt.seconds
        self.sleep = df.drop(columns=["datetimeReceived", "device", "provider",
                                      "location", "valueBoolean", "valueState", "valueUnit", "activity"])

    @timer('parsing physiology')
    def __parse_physiology(self,  df):
        df = df[df.type.isin(
            ['8310-5', '55284-4', '8867-4', '29463-7', '251837008', '163636005', '248362003', '8462-4', '8480-6', '150456'])].copy()       
        df['device_name'] = df.type.astype('category')
        df = self.remap_cat('device_name', self.map(
            self.device_type.display.values, self.device_type.code.values), df)
        self.physiology = df.drop(columns=["datetimeReceived", "provider", "location", "valueBoolean",
                                           "valueState", "valueDatetimeStart", "valueDatetimeEnd"])    

    @timer('parsing flags')
    def __parse_flags(self,_zip):
        df = pd.read_csv(_zip.open('Flags.csv'))
        _type = pd.read_csv(_zip.open('Flag-type.csv'))
        _cat = pd.read_csv(_zip.open('Flag-category.csv'))
        _val = pd.read_csv(_zip.open('FlagValidations.csv'))
        df = pd.merge(df, _val, how='outer', on=None,
                      left_on="flagId", right_on="flag",
                      suffixes=('df', '_val'), copy=True)
        df.category = pd.Categorical(df.category)
        df.rename(columns={'subjectdf': 'subject'}, inplace=True)
        df['project_id'] = df.subject
        # df = self.remap_cat('subject', _mapping, df)
        df = self.remap_cat('category', self.map(
            _cat.display.values, _cat.code.values), df)
        idx = self.find_duplicate(_type.display.values)[0]
        if any(idx):
            values = list(_type.code.values[idx])
            key = list(np.where(idx)[0])
            for key, val in dict(zip(key[0:-1], values[0:-1])).items():
                df.type[df.type == val] = values[-1]
                _type = _type.drop(key)
        df = self.remap_cat('type', self.map(
            _type.display.values, _type.code.values), df)
        df['project'] = self.name
        df['project'] = pd.Categorical(df.project)
        self.flags = df
        self.save_pickle()

    @timer('parsing wellbeing')
    def __parse_wellbeing(self, _zip):
        df = pd.read_csv(_zip.open('QuestionnaireResponses.csv'))
        df['datetimeAnswered'] = pd.to_datetime(df['datetimeAnswered'])
        df = df.sort_values(by=['datetimeAnswered'])
        df = df.drop(columns=["questionnaire", "datetimeReceived"])
        df.question, questions = pd.factorize(df.question)
        df['project_id'] = df.subject
        # df = self.remap_cat('subject', _mapping, df)
        df = df.dropna().reset_index(drop=True)
        df = df.drop_duplicates()
        index = pd.MultiIndex.from_tuples(zip(df.subject, df.datetimeAnswered,
                                              df.question), names=['subject', 'datetimeAnswered', 'question'])
        df = pd.DataFrame(df.answer.values, index=index,
                          columns=['answer']).unstack()
        df.columns = df.columns.droplevel()
        df = df.rename(columns=dict(zip(df.columns, questions)))
        df = df.reset_index()
        df['project'] = self.name
        dtype = dict(zip(df.columns, ['category']*df.shape[1]))
        dtype['datetimeAnswered'] = "datetime64[ns]"
        df = df.astype(dtype)
        df = df.sort_values(by=['datetimeAnswered'])
        self.wellbeing = df.copy()
        self.save_pickle()

    @timer('parsing transitions')
    def __parse_transitions(self):
        df = pd.concat([self.doors[self.doors.location != 'Fridge Door'], self.movement])[
            ['project', 'subject', 'datetimeObserved', 'location']].copy()
        df = pd.get_dummies(df, columns=['location'], prefix='', prefix_sep='')
        nodes = [n.replace(" ", "") for n in df.columns[3::]]
        cols = [f'{a}>{b}' for a, b in itertools.product(nodes, repeat=2)]
        tr = []
        for subj, subset in df.groupby('subject'):
            if subset.shape[0] > 1:
                ix = [i for _, i in np.argwhere(subset.set_index('datetimeObserved')
                                                .sort_index().iloc[:, 2::].values)]
                df = pd.get_dummies(np.ravel_multi_index(
                    [ix[0:-1], ix[1::]], (len(nodes), len(nodes))))
                df = df.rename(columns=dict(zip(df.columns, cols)))
                df = subset.reset_index(
                )[["datetimeObserved"]].diff().shift(-1).join(df)
                df = df.rename(columns={"datetimeObserved": "delta"})
                df.insert(0, 'subject', subj)
                df.insert(1, 'datetimeObserved',
                          subset.datetimeObserved.values)
                tr.append(df)
        df = pd.concat(tr)
        self.transitions = df.dropna(subset=['delta'])


def main():
    input_path = '/Users/eyalsoreq/GoogleDrive/Projects/OnGoing/DRI/Data/'
    output_path = '/Users/eyalsoreq/github/data/'
    datasets = ['15', 'dri']
    tihm = LoadData(input_path, output_path, datasets, False, False, True)
    

if __name__ == "__main__":
    main()
