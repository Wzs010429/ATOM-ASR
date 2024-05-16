# NOTE: RUN THIS FILE IN THE CORRESPONDING FILE DIRECTORY, OR CHANGE THE FILE PATH VARIBLES IN THE CODE

# %%
# load packages
import numpy as np
import pandas as pd

import librosa
from datasets import Dataset, Audio

import evaluate
from typing import Dict, Any
import requests

# %%
# define constants
seed = 42
raw_file_path = "data/raw/100测试语音.xlsx"
audio_format = "m4a"
audio_sr = 16000
audio_read_folder = "data/raw/100测试语音"
audio_save_folder = "data/processed/100测试语音"

# %%
# load the raw data
data = pd.read_excel(raw_file_path)

# rename the columns
data.rename(
    columns={
        "voice_url": "audio_url",
        "语音转文本结果": "prediction",
        "正确语音": "sentence",
        "应出标签": "tags",
    },
    inplace=True,
)

data["audio_id"] = data["audio_url"].str.extract(f"(\d+).{audio_format}")
data["audio"] = data["audio_id"].apply(
    lambda x: f"{audio_read_folder}/{x}.{audio_format}"
)


# %% data validation
# (data_asr["sentence"].str.strip() == data_asr["sentence"]).all()
# data.isnull().sum()


# %%
# compare the wer between the result and the sentence
# https://zhuanlan.zhihu.com/p/449264305

# drop the audio from default folder as they cannot be recognized properly (for now)
# _cnt = 0
# for _i in range(1, 11):
#     try:
#         _i = str(_i).zfill(2)
#         dataset = Dataset.from_pandas(data[data.audio_id == _i].drop(columns=["audio_url", "prediction", "tags"]))
#         dataset = dataset.cast_column("audio", Audio(sampling_rate=audio_sr))
#         print(dataset[0])
#     except:
#         _cnt += 1
#         print(f"audio_id {_i} failed")

# cer_metric = evaluate.load("cer")
# cer = cer_metric.compute(predictions=data["prediction"], references=data["sentence"])
# cer: 0.263
data = data.loc[data["audio_id"].str.len() != 2].reset_index(drop=True)
# cer = cer_metric.compute(predictions=data_no_default["prediction"], references=data_no_default["sentence"])
# cer: 0.258

data.to_csv(f"{audio_save_folder}/data.csv", index=False)


# %%
# convert to HuggingFace dataset
dataset = Dataset.from_pandas(data.drop(columns=["audio_url", "prediction", "tags"]), preserve_index=False)
dataset = dataset.cast_column("audio", Audio(sampling_rate=audio_sr))
dataset = dataset.map(
    lambda x: {
        "duration": librosa.get_duration(
            y=x["audio"]["array"], sr=x["audio"]["sampling_rate"]
        )
    }
)

dataset.save_to_disk(f"{audio_save_folder}/sr_{audio_sr}.hf")

# %%
# archive code
# load the audio data
# http://jar.bds-analytics.com:9090/group1/voice/726157.m4a
# def load_audio(audio_url) -> Dict[str, Any]:
#     """Load the audio from the url."""
#     response = requests.get(audio_url, headers={
#         "Accept-Encoding": "gzip, deflate",
#         "Accept-Language": "en",
#         "Host": "jar.bds-analytics.com:9090",
#         "Upgrade-Insecure-Requests": "1",
#         "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
#     })
#     # raise an exception if the request failed
#     response.raise_for_status()
#     # audio, sr = librosa.load(audio_url, sr=None) for local audio files
#     audio, sr = sf.read(io.BytesIO(response.content))
#     return {"array": audio, "path": audio_url, "sampling_rate": sr}


# def audio_length(audio_array: np.ndarray, sr: int) -> float:
#     """Get the length of the audio in seconds."""
#     return len(audio_array) / sr

# np.array(dataset['duration']).max() # 25.7045, no audio longer than 30 seconds


# def prepare_audio(folder_path: str, audio_id: str) -> dict:
#     """Prepare the audio data."""
#     audio_path = f"{folder_path}{audio_id}.{audio_format}"
#     audio, sr = librosa.load(audio_path, sr=None)
#     return {"array": audio, "sampling_rate": sr}


# # read the audio data and
# # add duration (in seconds) to the dataset
# dataset = dataset.map(
#     lambda x: {"audio": prepare_audio(audio_read_folder, x["audio_id"])}
# )
# # cast sampling rate to 16000 for whisper to consume
# dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
# dataset = dataset.map(
#     lambda x: {"duration": librosa.get_duration(y=x["audio"]["array"], sr=x["audio"]["sampling_rate"])}
# )
# dataset.save_to_disk(f"{audio_save_folder}sr_16000.hf")
