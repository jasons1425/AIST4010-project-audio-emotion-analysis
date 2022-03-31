import os
import glob
import re
import imageio
import cv2
import pandas as pd
import numpy as np
from data.preprocess import spectrum_transform
from data.dataset import ImageDataset
from torch.utils.data import DataLoader
from PIL import Image


DATA_DIR = r"D:\Documents\datasets\AIST4010\muse"
PNG_DIR = os.path.join(DATA_DIR, "spectrograms")
SPEC_DIR = os.path.join(DATA_DIR, "spectrograms_jpg")
SONGS_DATA = os.path.join(DATA_DIR, "extracted_data.csv")


def load_imgs(fp=SPEC_DIR):
    fp = glob.glob(os.path.join(fp, '*'))
    rematch_pattern = r"^.*\\([^\.]*).jpg"
    fp.sort(key=lambda fp: re.match(rematch_pattern, fp).group(1))
    imgs = [None] * len(fp)
    img_ids = [None] * len(fp)
    for idx, img_fp in enumerate(fp):
        img_id = re.match(rematch_pattern, img_fp).group(1)
        with Image.open(img_fp) as f:
            imgs[idx] = np.asarray(f.convert("RGB"))
        img_ids[idx] = img_id
    return np.array(imgs), np.array(img_ids)


def load_imgs_png(fp=PNG_DIR):
    fp = glob.glob(os.path.join(fp, '*'))
    rematch_pattern = r"^.*\\([^\.]*).png"
    fp.sort(key=lambda fp: re.match(rematch_pattern, fp).group(1))
    imgs = [None] * len(fp)
    img_ids = [None] * len(fp)
    for idx, img_fp in enumerate(fp):
        img_ids[idx] = re.match(rematch_pattern, img_fp).group(1)
        imgs[idx] = cv2.cvtColor(cv2.imread(img_fp), cv2.COLOR_BGR2RGB)
    return np.array(imgs), np.array(img_ids)


def get_labels(ids, fp=SONGS_DATA):
    songs_data = pd.read_csv(fp)
    songs_data.set_index("spotify_id", inplace=True)
    labels = songs_data.loc[ids, ["valence_tags", "arousal_tags", "dominance_tags"]].values
    return labels


def get_loader(data, labels, batch_size, shuffle=True,
               transform=spectrum_transform(), ds_class=ImageDataset, **kwargs):
    ds = ds_class(data, labels, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, **kwargs)
    return loader
