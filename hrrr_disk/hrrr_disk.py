import dateutil
from datetime import datetime, timedelta
import xarray as xr
import numpy as np
import os

import datasets

_CITATION = """\
@InProceedings{noaa:hrrr,
title = {HRRR ZARR},
author={jebb.q.stewart@noaa.gov.
},
year={2023}
}
"""

# You can copy an official description
_DESCRIPTION = """\
This dataset provides access to the hrrr zarr archive.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    # "hrrr_v4": "hrrr_v4.json",
    # "hrrr_v3": "hrrr_v3.json",
}

class HrrrOnline(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="hrrr_v4_analysis", version=VERSION, description="HRRR v4 3km Analysis files"),
  ]

    DEFAULT_CONFIG_NAME = "hrrr_v4_analysis"  # It's not mandatory to have a default configuration. Just use one if it make sense.


    def _info(self):

        self.NUM_FEATURES = 1 
        self.NUM_PREVIOUS_FRAMES = 3 # T-6, T-3, T
        self.NUM_FUTURE_FRAMES = 2 # T+3, T+6

        # features = {}

        features = {
            "past": datasets.Array3D((self.NUM_PREVIOUS_FRAMES*self.NUM_FEATURES,512,512), dtype="float32"),
            "predict": datasets.Array3D((self.NUM_FUTURE_FRAMES*self.NUM_FEATURES,512,512), dtype="float32"),

            "timestamp": datasets.Sequence(datasets.Value("timestamp[ns]")),
            # "latitude": datasets.Sequence(datasets.Value("float32")),
            # "longitude": datasets.Sequence(datasets.Value("float32"))
        }
        # if "forecast" in self.config.name:
        # change feature set based on config name


        features = datasets.Features(features)

        self.train_start =  dateutil.parser.parse("2023-01-04T00:00:00")
        self.train_end = dateutil.parser.parse("2023-01-12T23:01:00")

        self.test_start =  dateutil.parser.parse("2023-01-12T00:00:00")
        self.test_end = dateutil.parser.parse("2023-01-12T23:01:00")

        self.val_start =  dateutil.parser.parse("2023-01-12T00:00:00")
        self.val_end = dateutil.parser.parse("2023-01-12T23:01:00")


        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,

            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):

        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        # urls = _URLS[self.config.name]
        # streaming = dl_manager.is_streaming
        # if streaming:
        #     urls = dl_manager.download_and_extract(urls)
        # else:
        #     with open(filepath, "r") as f:
        #         filepaths = json.load(f)
        #         data_dir = dl_manager.download_and_extract(filepaths)

        streaming = dl_manager.is_streaming

        timestamp = self.train_start
        train_data = []
        while timestamp < self.train_end:
            train_data.append([timestamp])
            timestamp = timestamp + timedelta(hours=1)
        train_data = np.array(train_data)
        print (train_data.shape)


        # timestamp = self.val_start
        # val_data = []
        # while timestamp < self.val_end:
        #     val_data.append([timestamp])
        #     timestamp = timestamp + timedelta(hours=1)
        # val_data = np.array(val_data)

        # timestamp = self.test_start
        # test_data = []
        # while timestamp < self.test_end:
        #     test_data.append([timestamp])
        #     timestamp = timestamp + timedelta(hours=1)
        # test_data = np.array(test_data)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": train_data,
                    "split": "train",
                    "streaming": streaming,
                    "future_frames": self.NUM_FUTURE_FRAMES,
                    "past_frames": self.NUM_PREVIOUS_FRAMES,
                },
            ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.TEST,
            #     # These kwargs will be passed to _generate_examples
            #     gen_kwargs={
            #         "filepath": test_data,
            #         "split": "test",
            #         "streaming": streaming,
            #         "future_frames": self.NUM_FUTURE_FRAMES,
            #         "past_frames": self.NUM_PREVIOUS_FRAMES,
            #     },
            # ),
            # datasets.SplitGenerator(
            #     name=datasets.Split.VALIDATION,
            #     # These kwargs will be passed to _generate_examples
            #     gen_kwargs={
            #         "filepath": val_data,
            #         "split": "valid",
            #         "streaming": streaming,
            #         "future_frames": self.NUM_FUTURE_FRAMES,
            #         "past_frames": self.NUM_PREVIOUS_FRAMES,
            #     },
            # ),
        ]

    def read_file(self, timestamp):

        file =  timestamp.strftime("../data/%Y/%Y-%m-%d/HRRR_PRS/%Y%m%d_%H00.zarr")
        # print (f"reading {file}")
        ds = xr.open_zarr(file)
        #ds = ds[['t2m','u10','v10']].isel(valid_time=0).isel(x=slice(351,863),y=slice(263,775))
        ds = ds[['t2m']].isel(valid_time=0).isel(x=slice(351,863),y=slice(263,775))
        # print ("have values")
        return ds['t2m'].values


    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split, streaming, future_frames, past_frames):

        t2m_mean = 278.6412
        t2m_std = 21.236853
        u10_mean = -0.07404756
        u10_std = 5.567108
        v10_mean = 0.19169427
        v10_std = 4.7881403

        idx = 0

        for timestamps in filepath:
            try:

                past_data = []
                for i in range(past_frames-1,-1,-1):
                    filetime = timestamps[0] - timedelta(hours=i*3)
                    data = self.read_file(filetime)
                    past_data.append(data)

                past_data = np.array(past_data)
                
                predict_data = []
                for i in range(1,future_frames+1):
                    filetime = timestamps[0] + timedelta(hours=i*3)
                    data = self.read_file(filetime)
                    predict_data.append(data)

                predict_data = np.array(predict_data)

                value = {
                    "past": past_data, #np.stack(data.values, axis=2), 
                    "predict": predict_data,
                    "timestamp": timestamps
                }

                idx += 1 
                
                yield idx, value
            except Exception as e:
                print (e)
                # Some of the zarrs potentially have corrupted data at the end, and might fail, so this avoids that
                continue


