"""Dataset loading utilities for the cleaned TBME pipeline.

All loaders return ``Dataset`` with signals shaped as
``(n_trials, n_channels, n_timepoints)``. Dataset paths are resolved beneath
``root / "data"``, where ``root`` defaults to this project directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence
import pickle
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d


ROOT = Path(__file__).resolve().parent


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    fs: float
    stim_rate: float
    y_clean: Optional[np.ndarray] = None
    stim_channels: Optional[Sequence[int]] = None
    metadata: Optional[dict] = None
    baseline: Optional[np.ndarray] = None


def _scalar(x) -> float:
    return float(np.asarray(x).squeeze())


def _load_dictionary_mat(path: Path, scale: float, fs_key: str, crop: Optional[tuple[int, int]], stim_rate: float, name: str, clean_stop: Optional[int] = None) -> Dataset:
    mat = loadmat(path)
    X = np.transpose(scale * mat["dataInt"], (2, 1, 0))
    fs = _scalar(mat[fs_key])
    epoch = np.asarray(mat.get("t_epoch", mat.get("tEpoch", []))).ravel()
    baseline = None
    metadata = {}
    if epoch.size == X.shape[-1] and np.any(epoch < 0):
        prestim = epoch < 0
        baseline = X[:, :, prestim].copy()
        metadata["baseline_source"] = f"{path.name}: epoch time < 0 (prestimulation)"
        metadata["baseline_interval_s"] = [float(epoch[prestim][0]), float(epoch[prestim][-1])]
    y_clean = X[:, :, :clean_stop] if clean_stop is not None else None
    if crop is not None:
        X = X[:, :, crop[0]:crop[1]]
    stim_channels = None
    if "stimChans" in mat:
        stim_channels = (mat["stimChans"].flatten() - 1).astype(int).tolist()
    return Dataset(name=name, X=X, fs=fs, stim_rate=stim_rate, stim_channels=stim_channels,
                   y_clean=y_clean,
                   baseline=baseline, metadata=metadata)


def load_eraasr(root: Path = ROOT) -> Dataset:
    path = root / "data/eraasr-1.0.0/exampleDataTensor.mat"
    mat = loadmat(path)
    X = np.transpose(mat["data_trials_by_time_by_channels"], (0, 2, 1))
    return Dataset(
        name="ERAASR",
        X=X[:, :, 1600:3200],
        y_clean=X[:, :, 0:1200],
        baseline=X[:, :, 0:1200],
        metadata={"baseline_source": "ERAASR prestimulation samples [0:1200]",
                  "baseline_interval_s": [0.0, 1199 / 30000.0]},
        fs=30000.0,
        stim_rate=1000.0 / 3.0,
        stim_channels=[0],
    )


def load_dictionary_2stim(root: Path = ROOT, clean_stop: Optional[int] = None) -> Dataset:
    """Load stimulation data, optionally including the first clean_stop samples."""
    return _load_dictionary_mat(
        root / "data/dictionary-learning/+data/50ad9_paramSweep4.mat",
        scale=4e3,
        fs_key="fs_data",
        crop=(50000, 70000),
        stim_rate=185.0,
        name="dictionary_2stim",
        clean_stop=clean_stop,
    )


def load_dictionary_4stim(root: Path = ROOT) -> Dataset:
    return _load_dictionary_mat(
        root / "data/dictionary-learning/+data/693ffd_exampData_800ms.mat",
        scale=1e3,
        fs_key="fs_data",
        crop=(12220, 17000),
        stim_rate=200.0,
        name="dictionary_4stim",
    )


def load_dictionary_5stim(root: Path = ROOT) -> Dataset:
    return _load_dictionary_mat(
        root / "data/dictionary-learning/+data/2fd831_exampData_400ms.mat",
        scale=1e3,
        fs_key="fs_data",
        crop=(12220, 17000),
        stim_rate=200.0,
        name="dictionary_5stim",
    )


def load_dictionary_14stim(root: Path = ROOT, clean_stop: Optional[int] = None) -> Dataset:
    """Load stimulation data, optionally including the first clean_stop samples."""
    return _load_dictionary_mat(
        root / "data/dictionary-learning/+data/693ffd_exampData_400ms.mat",
        scale=1e3,
        fs_key="fsData",
        crop=(12220, 17000),
        stim_rate=200.0,
        name="dictionary_14stim",
        clean_stop=clean_stop,
    )


def load_swec_synthetic(
    root: Path = ROOT,
    stim_channels: Sequence[int] = (0, 1, 2, 8),
    artifact_scale: float = 1.0,
    crop_to_stim: bool = True,
) -> Dataset:
    path = (
        root / "data/swec_synthetic/"
        "StimChannels0_1_2_8_9_10_16_17_18_24_25_26_84_85_86_87_"
        "StimRate129Hz_fpulse2500_SampleRate10240.npz"
    )
    data = np.load(path)
    fs = 10240.0
    y_clean = data["seizure_data"].copy()
    X = y_clean.copy()
    params = data["seizure_stim_params"]
    artifacts = data["seizure_artifact"]

    for stim_ch in stim_channels:
        matched = False
        for idx in range(artifacts.shape[0]):
            if int(params[idx][0][2]) == int(stim_ch):
                X += artifact_scale * artifacts[idx]
                matched = True
                break
        if not matched:
            raise ValueError(f"Stim channel {stim_ch} was not found in seizure_stim_params.")

    stim_rate = _scalar(params[0][0][0])
    if crop_to_stim:
        t0 = int(_scalar(params[0][0][4]) * fs)
        t1 = max(t0 + 1, int(_scalar(params[0][0][5]) * fs) - 1000)
        X = X[:, :, t0:t1]
        y_clean = y_clean[:, :, t0:t1]

    return Dataset(
        name=f"swec_synthetic_{'_'.join(map(str, stim_channels))}",
        X=X,
        y_clean=y_clean,
        fs=fs,
        stim_rate=stim_rate,
        stim_channels=list(stim_channels),
        metadata={"artifact_scale": artifact_scale},
    )


def _load_xlsx_voltage_columns(path: Path) -> np.ndarray:
    zf = zipfile.ZipFile(path)
    ns = {"s": "http://purl.oclc.org/ooxml/spreadsheetml/main"}
    strings = [t.text for t in ET.fromstring(zf.read("xl/sharedStrings.xml")).findall(".//s:t", ns)]
    rows = ET.fromstring(zf.read("xl/worksheets/sheet1.xml")).findall(".//s:row", ns)
    header = [strings[int(c.find("s:v", ns).text)] for c in rows[0].findall("s:c", ns)]
    data = [[float(c.find("s:v", ns).text) for c in r.findall("s:c", ns)] for r in rows[1:]]
    cols = [i for i, name in enumerate(header) if name.startswith("Volts")]
    return np.asarray(data, dtype=float)[:, cols]


def load_swec_pbs(
    root: Path = ROOT,
    configuration_number: int = 0,
    artifact_scale: float = 10000.0,
    separated_recording_channels: bool = False,
) -> Dataset:
    fs = 10240.0
    pbs_fs = 16000.0
    pbs_path = root / f"data/PBS_data/Raw-Data/Data{configuration_number}.xlsx"
    pbs = _load_xlsx_voltage_columns(pbs_path)
    pbs = pbs[int(0.1 * pbs_fs):int(1.0 * pbs_fs), :]

    n_target = int(pbs.shape[0] * fs / pbs_fs)
    pbs_rs = np.zeros((n_target, pbs.shape[1]), dtype=float)
    source_t = np.arange(pbs.shape[0]) / pbs_fs
    target_t = np.arange(n_target) / fs
    for ch in range(pbs.shape[1]):
        pbs_rs[:, ch] = interp1d(source_t, pbs[:, ch], kind="linear")(target_t)
    pbs_artifact = pbs_rs.T[np.newaxis, :, :]

    base = load_swec_synthetic(root=root, stim_channels=(0,), artifact_scale=0.0, crop_to_stim=False)
    chans = [0, 11, 22, 33, 44, 55, 66, 77] if separated_recording_channels else list(range(8))
    y_clean = base.X[0:1, chans, int(0.1 * fs):int(1.0 * fs)]
    X = y_clean + artifact_scale * pbs_artifact
    return Dataset(
        name=f"swec_pbs_config{configuration_number}",
        X=X,
        y_clean=y_clean,
        fs=fs,
        stim_rate=250.0,
        stim_channels=[0, 1, 2, 3],
        metadata={"pbs_path": str(pbs_path), "artifact_scale": artifact_scale},
    )


def load_cornell_jb(root: Path = ROOT) -> Dataset:
    import h5py

    path = root / "data/DBS_sessions_JB_Cornell/H061720_Block1_BGMR_ALL_RIGHT.mat"
    with h5py.File(path, "r") as f:
        data = f["BGMR/BB"][:].T
        fs = float(f["BGMR/SamplingRate"][()].item())
    t0 = int(20 * 10**6 - fs)
    t1 = int(20 * 10**6)
    return Dataset(name="cornell_jb", X=data[np.newaxis, :, t0:t1], fs=fs, stim_rate=200.0)


def load_selected_clips(file_path: str | Path) -> dict:
    with open(file_path, "rb") as f:
        return pickle.load(f)


def split_data_per_patient(selected_data: dict, patient_id: str) -> tuple[np.ndarray, np.ndarray]:
    seizure = selected_data[patient_id]["seizure_clips"]
    non_seizure = selected_data[patient_id]["non_seizure_clips"]
    channels = seizure[0].shape[0]
    seizure_data = np.stack([clip.reshape(channels, -1) for clip in seizure], axis=0)
    non_seizure_data = np.stack([clip.reshape(channels, -1) for clip in non_seizure], axis=0)
    return seizure_data, non_seizure_data


DATASET_LOADERS: Dict[str, Callable[..., Dataset]] = {
    "ERAASR": load_eraasr,
    "dictionary_2stim": load_dictionary_2stim,
    "dictionary_4stim": load_dictionary_4stim,
    "dictionary_5stim": load_dictionary_5stim,
    "dictionary_14stim": load_dictionary_14stim,
    "swec_synthetic": load_swec_synthetic,
    "swec_pbs": load_swec_pbs,
    "cornell_jb": load_cornell_jb,
}


def available_datasets() -> Iterable[str]:
    return DATASET_LOADERS.keys()


def load_dataset(name: str, root: Path = ROOT, **kwargs) -> Dataset:
    try:
        loader = DATASET_LOADERS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(DATASET_LOADERS))
        raise ValueError(f"Unknown dataset '{name}'. Available datasets: {choices}") from exc
    return loader(root=root, **kwargs)


# write the main file to load the datasets and print their shapes
def main():
    for name in available_datasets():
        dataset = load_dataset(name)
        print(f"Dataset: {name}")
        print(f"  X shape: {dataset.X.shape}")
        if dataset.y_clean is not None:
            print(f"  y_clean shape: {dataset.y_clean.shape}")
        if dataset.baseline is not None:
            print(f"  baseline shape: {dataset.baseline.shape}")
        print(f"  fs: {dataset.fs}")
        print(f"  stim_rate: {dataset.stim_rate}")
        if dataset.stim_channels is not None:
            print(f"  stim_channels: {dataset.stim_channels}")
        if dataset.metadata is not None:
            print(f"  metadata: {dataset.metadata}")
        print()

if __name__ == "__main__":
    main()

