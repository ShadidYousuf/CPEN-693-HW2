from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

DATA = Path(__file__).resolve().parent / "data"
EEG_SETS = {"A": "healthy", "D": "interictal", "E": "seizure"}
EEG_PAIRS = [("A", "D"), ("A", "E"), ("D", "E")]


def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), y_train, y_test


def get_breast_cancer():
    df = pd.read_csv(DATA / "BreastCancerWisconsin.csv")
    X = df.iloc[:, 2:32].values
    y = (df["diagnosis"] == "M").astype(int).values
    return split_and_scale(X, y)


def get_eeg():
    eeg = {}
    for s in EEG_SETS:
        files = sorted((DATA / "eeg_raw" / s).glob("*.txt"))
        eeg[s] = np.array([np.loadtxt(f) for f in files])
    return eeg


def get_eeg_pair(eeg, a, b):
    X = np.concatenate([eeg[a].var(axis=1), eeg[b].var(axis=1)]).reshape(-1, 1)
    y = np.repeat([0, 1], len(eeg[a]))
    return split_and_scale(X, y)


def get_loaders(name, batch_size=128):
    dataset = getattr(datasets, name)
    train = dataset(DATA, train=True, download=True)
    test = dataset(DATA, train=False, download=True)
    train_dl = DataLoader(TensorDataset(train.data.float() / 255, train.targets), batch_size, shuffle=True)
    test_dl = DataLoader(TensorDataset(test.data.float() / 255, test.targets), batch_size)
    return train_dl, test_dl, train.classes
