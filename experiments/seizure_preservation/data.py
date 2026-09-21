"""Lazy HDF5 access, recording/seizure grouping, and split-first window extraction."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import h5py
import numpy as np
from scipy.signal import resample_poly
from .common import derived_seed, write_json, write_csv


@dataclass
class PatientData:
    patient: str
    path: Path
    fs: int
    channels: int
    samples: int
    seizures: np.ndarray  # half-open native-sample intervals
    file_order: list[str]


def load_patient_data(patient, config):
    pid = f'ID{int(str(patient).removeprefix("ID")):02d}'
    path = Path(config['data_root']) / f'{pid}.h5'
    with h5py.File(path, 'r') as f:
        g = f[pid]
        channels, samples = g['EEG'].shape
        fs = float(g.attrs['fs_target'])
        if fs != int(fs) or fs <= 0:
            raise ValueError('Expected positive integer stored sampling rate')
        intervals = np.stack([g['seizure_begin'][:], g['seizure_end'][:]], axis=1)
        intervals = np.rint(intervals * fs).astype(np.int64)
        intervals = intervals[np.argsort(intervals[:, 0])]
        order = g.attrs['file_order']
        if isinstance(order, bytes):
            order = order.decode()
        files = order.splitlines()
        if len(files) != int(g.attrs['n_files']) or channels != int(g.attrs['n_channels']):
            raise ValueError(f'{pid}: inconsistent recording metadata')
    if not len(intervals) or np.any(intervals[:, 0] < 0) or np.any(intervals[:, 1] > samples) or np.any(intervals[:, 1] <= intervals[:, 0]):
        raise ValueError(f'{pid}: invalid seizure boundaries')
    # This export records file order but not per-file lengths. Explicit hourly
    # assumption is validated against names/count/total length, and saved below.
    expected = [f'{pid}_{i}h.mat' for i in range(1, len(files) + 1)]
    block = round(config['recording_seconds'] * fs)
    if files != expected or not (len(files)-1)*block < samples <= len(files)*block:
        raise ValueError(f'{pid}: cannot infer hourly boundaries safely; supply a different loader with exact offsets')
    return PatientData(pid, path, int(fs), channels, samples, intervals, files)


def _allocation(n, ratios, minima):
    counts = np.asarray(minima, dtype=int)
    if counts.sum() > n:
        raise ValueError('Not enough independent recording groups')
    while counts.sum() < n:
        counts[np.argmax(n * np.asarray(ratios) - counts)] += 1
    return counts


def split_patient_data(patient, config, output_dir=None):
    """Assign whole hourly recordings before making any windows.

    Union recordings touched by the same seizure (plus contextual margin).
    Stratify independent seizure-bearing and seizure-free groups. Never split
    one seizure across train/validation/test, even at a recording boundary.
    """
    n = len(patient.file_order)
    block = round(config['recording_seconds'] * patient.fs)
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    margin = round(config['seizure_group_margin_seconds'] * patient.fs)
    for start, stop in patient.seizures:
        first = max(0, int(start-margin)) // block
        last = min(patient.samples-1, int(stop+margin-1)) // block
        for i in range(first+1, last+1):
            parent[find(i)] = find(first)
    members = {}
    for i in range(n):
        members.setdefault(find(i), []).append(i)
    groups = []
    for indices in members.values():
        start, stop = indices[0]*block, min((indices[-1]+1)*block, patient.samples)
        seizures = np.flatnonzero((patient.seizures[:, 0] < stop) & (patient.seizures[:, 1] > start)).tolist()
        groups.append(dict(group=f'{patient.patient}_recordings_{indices[0]+1}_{indices[-1]+1}',
                           start=start, stop=stop, recording_indices=indices, seizures=seizures))
    positive = [g for g in groups if g['seizures']]
    negative = [g for g in groups if not g['seizures']]
    if len(positive) < 2:
        raise ValueError(f'{patient.patient}: fewer than two independent seizure groups; cannot train/test without leakage')
    note = None
    if len(positive) == 2:
        if config['two_seizure_policy'] != 'negative_only_validation':
            raise ValueError('Two seizure groups require explicit negative-only validation policy')
        minima = [1, 0, 1]
        note = 'Only two independent seizure groups: one train, one test; validation is seizure-free. Validation loss cannot assess seizure sensitivity.'
    else:
        minima = [1, 1, 1]
    rng = np.random.default_rng(derived_seed(config['seed'], patient.patient, 'split'))
    splits = {key: [] for key in ('train', 'validation', 'test')}
    for pool, minimum in [(positive, minima), (negative, [1, 1, 1])]:
        rng.shuffle(pool)
        counts = _allocation(len(pool), config['split_ratios'], minimum)
        start = 0
        for name, count in zip(splits, counts):
            splits[name].extend(pool[start:start+count])
            start += count
    for name in splits:
        splits[name].sort(key=lambda g: g['start'])
    if output_dir is not None:
        write_json(Path(output_dir)/'splits.json', dict(patient=patient.patient, seed=config['seed'],
            native_fs=patient.fs, target_fs=config['target_fs'],
            boundary_provenance='Inferred hourly offsets from sequential file_order; last recording may be partial; per-file lengths are absent in the export.',
            file_order=patient.file_order, seizure_intervals_samples=patient.seizures.tolist(),
            note=note, groups=splits,
            durations_seconds={k: sum(g['stop']-g['start'] for g in v)/patient.fs for k,v in splits.items()}))
    return splits


def extract_windows(patient, groups, split, config, output_dir):
    """Enumerate only within assigned groups, then read selected windows to disk.

    Caps sample within a split after grouping; no random-window splitting.
    Resampling is window-local and cannot read across split boundaries.
    """
    output_dir = Path(output_dir)
    width = round(config['window_seconds'] * patient.fs)
    stride = round(width * (1-config['overlap']))
    guard = round(config['split_boundary_guard_seconds'] * patient.fs)
    candidates = []
    for group in groups:
        start = group['start'] + (guard if group['start'] else 0)
        stop = group['stop'] - (guard if group['stop'] < patient.samples else 0)
        starts = np.arange(start, stop-width+1, stride, dtype=np.int64)
        labels = np.zeros(len(starts), dtype=np.int8)
        for onset, offset in patient.seizures:
            labels[(starts < offset) & (starts+width > onset)] = 1
        candidates.extend(dict(start=int(s), stop=int(s+width), label=int(y), group=group['group'])
                          for s,y in zip(starts,labels))
    rng = np.random.default_rng(derived_seed(config['seed'], patient.patient, split, 'window_selection'))
    selected = []
    counts = {}
    pools = [[i for i,row in enumerate(candidates) if row['label'] == label] for label in (0,1)]
    counts = {str(label): len(pool) for label,pool in enumerate(pools)}
    budget = None
    caps = config['max_windows_per_class'][split]
    if config.get('samples_per_patient') is not None:
        budgets = _allocation(config['samples_per_patient'], config['split_ratios'], [2,2,2])
        budget = int(budgets[('train','validation','test').index(split)])
        # Prefer equal classes; fill a scarce/absent class's quota from the
        # other class without replacement, never borrowing from another split.
        caps = [min(len(pools[0]), budget//2), min(len(pools[1]), budget-budget//2)]
        remaining = budget-sum(caps)
        for label in (0,1):
            extra = min(remaining, len(pools[label])-caps[label])
            caps[label] += extra
            remaining -= extra
    for label, cap in enumerate(caps):
        indices = pools[label]
        if cap is not None and len(indices) > cap:
            indices = rng.choice(indices, cap, replace=False).tolist()
        selected.extend(candidates[i] for i in indices)
    selected.sort(key=lambda row: row['start'])
    if not selected:
        raise ValueError(f'{patient.patient}/{split}: no windows')
    target_width = round(width * config['target_fs'] / patient.fs)
    divisor = np.gcd(patient.fs, config['target_fs'])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir/f'{split}_clean.npy'
    x = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                  shape=(len(selected), patient.channels, target_width))
    with h5py.File(patient.path, 'r') as f:
        eeg = f[f'{patient.patient}/EEG']
        for i, row in enumerate(selected):
            raw = eeg[:, row['start']:row['stop']]
            if not np.isfinite(raw).all():
                raise ValueError(f'Nonfinite input in {patient.patient}: {row}')
            x[i] = resample_poly(raw, config['target_fs']//divisor, patient.fs//divisor, axis=-1)
    x.flush()
    labels = np.array([r['label'] for r in selected], dtype=np.int64)
    np.save(output_dir/f'{split}_labels.npy', labels)
    write_csv(output_dir/f'{split}_windows.csv', [dict(window=i, **r) for i,r in enumerate(selected)])
    write_json(output_dir/f'{split}_counts.json', dict(candidate_counts=counts,
        selected_counts={str(y): int((labels==y).sum()) for y in [0,1]},
        requested_split_budget=budget,
        selection=('Seeded class-balanced sampling where feasible; unused class quota filled from the other class within this split. Overrides per-class caps.'
                   if budget is not None else 'All positives unless explicitly capped; seeded negative subsampling if capped.')
                  + ' Metrics describe this selected prevalence.'))
    return np.load(path, mmap_mode='r'), labels
