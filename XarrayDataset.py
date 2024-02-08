# -*- coding: utf-8 -*-
#
# @Author: Jebb Q. Stewart
# @Date:   2023-12-16
# @Email: jebb.q.stewart@noaa.gov 
#
# @Last modified by:   Jebb Q. Stewart
# @Last Modified time: 2024-02-08 11:19:35

import datasets
import numpy as np
import pandas as pd
import dateutil
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import os
import itertools
from datasets import Array2D, Array3D, Features, Sequence, Value, load_dataset

from torch.utils.data import DataLoader, IterableDataset
import torch
from torchvision import transforms as T


class XarrayDataset(IterableDataset):
    def __init__(self, data_start, data_end, name, batch_size, image_size):
        super().__init__()

        self.name = name
        self.batch_size = batch_size

        self.image_size=image_size
        tfms = [T.Resize((self.image_size, self.image_size),antialias=None)] if self.image_size is not None else []
        # tfms += [T.RandomCrop(img_size)] if not valid else [T.CenterCrop(img_size)]
        self.tfms = T.Compose(tfms)

        LATITUDE = 512
        LONGITUDE = 512

        self.NUM_FEATURES = 1 
        self.NUM_PREVIOUS_FRAMES = 3 # T-6, T-3, T
        self.NUM_FUTURE_FRAMES = 2 # T+3, T+6
        self.variables = ['t2m']

        # TODO: extract out to file or something else
        self.means = {}
        self.means['u10']   = -0.07404756
        self.means['v10']   = 0.19169427
        self.means['t2m']   = 278.6412
        self.means['prmsl'] = 100966.72
        self.means['pwat']   = 18.408087

        self.stds = {}
        self.stds['u10']   = 5.567108
        self.stds['v10']   = 4.7881403
        self.stds['t2m']   = 21.236853
        self.stds['prmsl'] = 1330.6351
        self.stds['pwat']   = 16.482214

        if self.name == "hrrr_v4_more_analysis":
             self.variables = ['t2m', 'u10', 'v10', 'prmsl', 'pwat']
             self.NUM_FEATURES = len(self.variables)

        print (f"using {self.name}")
        print ("  with variables:")
        for v in self.variables:
            print (f"    {v}")

        features = {
            "past": datasets.Array3D((self.NUM_PREVIOUS_FRAMES*self.NUM_FEATURES,512,512), dtype="float32"),
            "predict": datasets.Array3D((self.NUM_FUTURE_FRAMES*self.NUM_FEATURES,512,512), dtype="float32"),
            "timestamp": datasets.Sequence(datasets.Value("timestamp[ns]")),
            # "latitude": datasets.Sequence(datasets.Value("float32")),
            # "longitude": datasets.Sequence(datasets.Value("float32"))
        }

        features = datasets.Features(features)

        self.data_start =  dateutil.parser.parse(data_start)
        self.data_end = dateutil.parser.parse(data_end)

        # Build a list of timestamps we will use for this dataset
        timestamp = self.data_start
        data = []
        while timestamp < self.data_end:
            data.append([timestamp])
            timestamp = timestamp + timedelta(hours=1)
        data = np.array(data)

        self.dataset = data

    def read_file(self, timestamp):

        file =  timestamp.strftime("../data/%Y/%Y-%m-%d/HRRR_PRS/%Y%m%d_%H00.zarr")
        ds = xr.open_zarr(file)
        ds = ds[self.variables].isel(valid_time=0).isel(x=slice(351,863),y=slice(263,775))

        data = []

        for v in self.variables:
            data.append((ds[v].values - self.means[v])/self.stds[v])

        data = np.array(data)

        return data

    def __len__(self):
        return len(self.dataset)

    def read(self, timestamp):

        # Look back in time, taking previous frames at 3 hour stride to T0
        past_data = []
        for i in range(self.NUM_PREVIOUS_FRAMES-1,-1,-1):
            filetime = timestamp[0] - timedelta(hours=i*3)
            data = self.read_file(filetime)
            # print (data.shape)
            past_data.append(data)

        past_data = np.vstack(past_data)
        
        # our predict frames going fowward from T0
        predict_data = []
        for i in range(1,self.NUM_FUTURE_FRAMES+1):
            filetime = timestamp[0] + timedelta(hours=i*3)
            data = self.read_file(filetime)
            predict_data.append(data)

        predict_data = np.vstack(predict_data)

        # Add transform
        past_data = self.tfms(torch.from_numpy(past_data))
        predict_data = self.tfms(torch.from_numpy(predict_data))

        value = {
            "past": past_data, 
            "predict": predict_data,
            "timestamp": timestamp
        }

        return value

    def shuffle(self, seed):
        """Shuffles the dataset, useful for getting 
        interesting samples on the validation dataset"""
        idxs = torch.randperm(len(self.dataset),generator=torch.Generator().manual_seed(seed))
        self.dataset = self.dataset[idxs]
        return self

    def __getitem__(self, idx):
        yield self.read[idx]

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        # Split our dataset based on the number of workers
        if worker_info:
            worker_total_num = worker_info.num_workers
            worker_id = worker_info.id
            partition = itertools.islice(self.dataset, worker_id, None, self.batch_size)
        else:
            partition = self.dataset

       
        for timestamp in iter(partition):
            try:
                data = self.read(timestamp)
                yield data["past"],data["predict"]
            except Exception as e:
                print (e)
                continue

       
           

