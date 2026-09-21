"""Offline artifact-template dictionaries adapted from data/dictionary-learning.

Ports the default template_dictionary path: baseline removal, zero padding,
HDBSCAN cluster means, peak-bracket matching, peak-to-peak scaling, subtraction.
See LICENSE-DICTIONARY and README for provenance and deliberate edge-case fixes.
"""
from dataclasses import dataclass, replace
import numpy as np
from scipy.signal import savgol_filter

from .methods import BenchmarkConfig, BenchmarkResult, validate_signal, pulse_times, window_samples, windows


@dataclass
class ArtifactDictionary:
    """Per-channel arrays (time, atoms), with frozen fitting settings and diagnostics."""
    templates: tuple
    peak_indices: tuple
    fs: float
    config: BenchmarkConfig
    diagnostics: dict


def _validate_config(config, fs):
    if fs is None:
        raise ValueError('fs must be supplied in Hz')
    window_samples(fs, config)
    for name, minimum in [('dictionary_min_points', 2), ('dictionary_min_cluster_size', 2),
                          ('dictionary_baseline_samples', 1), ('dictionary_bracket', 0)]:
        value = getattr(config, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if config.dictionary_normalize not in ('preAverage', 'firstSamp', 'mean', 'none'):
        raise ValueError('dictionary_normalize must be preAverage, firstSamp, mean or none')
    if config.dictionary_match not in ('corr', 'cosine', 'eucl'):
        raise ValueError('dictionary_match must be corr, cosine or eucl')
    scalars = [config.dictionary_outlier_threshold, config.dictionary_onset_threshold,
               config.dictionary_detection_ms, config.dictionary_min_duration_ms,
               config.dictionary_end_percentile]
    if (not np.isfinite(scalars).all() or not 0 <= scalars[0] <= 1 or scalars[1] <= 0
            or scalars[2] <= 0 or scalars[3] < 0 or not 0 <= scalars[4] <= 100):
        raise ValueError('Invalid dictionary outlier/detection parameters')


def _normalize(epoch, config):
    if config.dictionary_normalize == 'preAverage':
        baseline = epoch[:config.dictionary_baseline_samples].mean()
    elif config.dictionary_normalize == 'firstSamp':
        baseline = epoch[0]
    elif config.dictionary_normalize == 'mean':
        baseline = epoch.mean()
    else:
        baseline = 0.
    return epoch - baseline


def _abs_zscore(x):
    if x.size < 2:
        return np.zeros_like(x, dtype=float)
    std = np.std(x, ddof=1)
    return np.abs((x-x.mean())/std) if std > 0 else np.zeros_like(x, dtype=float)


def dictionary_windows(x, stim_times, fs, config):
    """Return trial/channel half-open intervals; supplied markers bypass detection.

    Automatic detection adapts get_artifact_indices.m: Savitzky-Golay(3, 7),
    derivative z-score on a dominant channel, and per-channel percentile ends.
    Short/constant recordings yield no windows rather than indexing failures.
    """
    pre, post = window_samples(fs, config)
    trials, channels, samples = x.shape
    if stim_times is not None:
        times = pulse_times(stim_times, trials, samples)
        return [[windows(row, samples, pre, post) for _ in range(channels)] for row in times]
    if samples < 7:
        return [[[] for _ in range(channels)] for _ in range(trials)]
    smooth = savgol_filter(x, 7, 3, axis=-1)
    diff = np.diff(smooth, axis=-1, prepend=smooth[..., :1])
    search = max(1, int(round(config.dictionary_detection_ms*fs/1000)))
    minimum = int(round(config.dictionary_min_duration_ms*fs/1000))
    if search < pre+minimum:
        raise ValueError('dictionary_detection_ms must cover pre_ms + dictionary_min_duration_ms')
    output = []
    for tr in range(trials):
        dominant = np.argmax(np.max(np.abs(diff[tr]), axis=1))
        hits = np.flatnonzero(_abs_zscore(diff[tr, dominant]) > config.dictionary_onset_threshold)
        if not hits.size:
            output.append([[] for _ in range(channels)])
            continue
        gaps = np.diff(hits)
        splits = (gaps > 1) & (_abs_zscore(gaps) > config.dictionary_onset_threshold)
        if gaps.size and np.std(gaps) == 0:
            splits = gaps > 1
        onsets = hits[np.r_[0, np.flatnonzero(splits)+1]]
        channel_windows = []
        for ch in range(channels):
            bounds = []
            for onset in onsets:
                start = max(0, int(onset)-pre)
                stop = min(samples, start+search+1)
                voltage = _abs_zscore(smooth[tr, ch, start:stop])
                derivative = _abs_zscore(diff[tr, ch, start:stop])
                first = max(0, int(onset)-start+minimum)
                candidates = []
                for signal in (voltage, derivative):
                    # MATLAB percentile locations are (i-.5)/N (Hazen).
                    threshold = np.percentile(signal, config.dictionary_end_percentile, method='hazen')
                    inds = np.flatnonzero(signal[first:] > threshold)
                    if inds.size:
                        candidates.append(first+int(inds[-1]))
                last = max(candidates, default=min(stop-start-1, first))
                end = min(samples, start+last+post+1)
                if bounds and start <= bounds[-1][1]:
                    bounds[-1] = (bounds[-1][0], max(end, bounds[-1][1]))
                else:
                    bounds.append((start, end))
            channel_windows.append(bounds)
        output.append(channel_windows)
    return output


def _cluster(features, config):
    """Use the Python HDBSCAN implementation, with explicit GLOSH rejection."""
    try:
        from hdbscan import HDBSCAN
    except ImportError as exc:
        raise ImportError('Dictionary learning requires: pip install -r benchmarks/requirements-dictionary.txt') from exc
    if len(features) < max(config.dictionary_min_cluster_size, config.dictionary_min_points):
        return np.full(len(features), -1, dtype=int), np.zeros(len(features))
    # MATLAB minPts includes the zero self-distance; hdbscan 0.8.40's
    # KD-tree query uses min_samples+1 neighbors, so subtract one here.
    clusterer = HDBSCAN(min_cluster_size=config.dictionary_min_cluster_size,
                        min_samples=config.dictionary_min_points-1, metric='euclidean',
                        approx_min_span_tree=False, core_dist_n_jobs=1).fit(features)
    labels = clusterer.labels_.copy()
    scores = np.nan_to_num(clusterer.outlier_scores_, nan=1., posinf=1., neginf=0.)
    labels[scores > config.dictionary_outlier_threshold] = -1
    return labels, scores


def fit_dictionary(training_data, stim_times=None, fs=None, config=None):
    """Learn per-channel cluster-average artifact templates from explicit recordings.

    Signals use (trials, channels, time). No clean labels are consumed. If
    stim_times is None, detect pulse windows from these recordings offline.
    """
    config = replace(config or BenchmarkConfig(method='dictionary_learning'))
    _validate_config(config, fs)
    x = validate_signal(training_data)
    bounds = dictionary_windows(x, stim_times, fs, config)
    templates, peaks, labels_all, scores_all, fallbacks = [], [], [], [], []
    for ch in range(x.shape[1]):
        epochs = [_normalize(x[tr, ch, start:end], config)
                  for tr in range(x.shape[0]) for start, end in bounds[tr][ch]]
        if not epochs:
            templates.append(np.empty((0, 0)))
            peaks.append(0)
            labels_all.append(np.empty(0, dtype=int))
            scores_all.append(np.empty(0))
            fallbacks.append('no_windows')
            continue
        length = max(map(len, epochs))
        padded = np.zeros((len(epochs), length))
        for i, epoch in enumerate(epochs):
            padded[i, :len(epoch)] = epoch
        # Per-channel rather than a single global peak; supports opposite
        # polarity channels while retaining the original signed-positive peak.
        peak = int(np.floor(np.median([np.argmax(e) for e in epochs])+.5))
        bracket = slice(max(0, peak-config.dictionary_bracket), min(length, peak+config.dictionary_bracket+1))
        labels, scores = _cluster(padded[:, bracket], config)
        groups = np.unique(labels[labels >= 0])
        if groups.size:
            atoms = np.column_stack([padded[labels == group].mean(axis=0) for group in groups])
            fallbacks.append('')
        else:
            # Source mean(templateArray)' averages the wrong axis. Average
            # pulses here so the fallback is one full-length waveform.
            atoms = padded.mean(axis=0)[:, None]
            fallbacks.append('mean_all_pulses')
        atoms.flags.writeable = False
        templates.append(atoms)
        peaks.append(peak)
        labels_all.append(labels)
        scores_all.append(scores)
    return ArtifactDictionary(tuple(templates), tuple(peaks), float(fs), config, {
        'training_windows': bounds, 'cluster_labels': labels_all, 'outlier_scores': scores_all,
        'fallbacks': fallbacks, 'clustering': 'hdbscan',
        'window_source': 'markers' if stim_times is not None else 'detected'})


def _best_template(epoch, atoms, peak, config):
    lo, hi = max(0, peak-config.dictionary_bracket), min(len(epoch), peak+config.dictionary_bracket+1)
    if lo >= hi:
        lo, hi = 0, len(epoch)
    y, candidates = epoch[lo:hi], atoms[lo:hi]
    distances = np.sum((candidates-y[:, None])**2, axis=0)
    if config.dictionary_match == 'eucl':
        return int(np.argmin(distances))
    if config.dictionary_match == 'corr':
        y = y-y.mean()
        candidates = candidates-candidates.mean(axis=0)
    denominator = np.linalg.norm(y)*np.linalg.norm(candidates, axis=0)
    similarities = np.full(atoms.shape[1], -np.inf)
    valid = denominator > 0
    similarities[valid] = y @ candidates[:, valid]/denominator[valid]
    if config.dictionary_match == 'cosine':
        similarities[valid] = np.abs(similarities[valid])
    # Undefined correlation/cosine on constant/one-sample brackets: use
    # deterministic Euclidean matching rather than propagate NaNs.
    return int(np.argmax(similarities)) if valid.any() else int(np.argmin(distances))


def dictionary_learning(x, stim_times=None, fs=None, config=None, *, model=None,
                        training_data=None, training_stim_times=None):
    """Offline windowed cancellation, optionally with a separately fitted dictionary.

    Without a model/training_data, learn from x as in the MATLAB source. Never
    claim sample-level online causality: matching/scaling need the full epoch.
    """
    x = validate_signal(x)
    if model is not None and (training_data is not None or training_stim_times is not None):
        raise ValueError('Provide either a fitted model or training recordings, not both')
    source = 'provided_model'
    if model is None:
        if training_data is None and training_stim_times is not None:
            raise ValueError('training_stim_times requires training_data')
        source = 'same_recording' if training_data is None else 'provided_training'
        model = fit_dictionary(x if training_data is None else training_data,
                               stim_times if training_data is None else training_stim_times, fs, config)
    if not isinstance(model, ArtifactDictionary) or len(model.templates) != x.shape[1]:
        raise ValueError('Model must be an ArtifactDictionary with matching channel count')
    if fs != model.fs:
        raise ValueError('Training and inference sampling rates must match')
    if config is not None:
        keys = ['pre_ms', 'post_ms'] + [k for k in vars(config) if k.startswith('dictionary_')]
        if any(getattr(config, k) != getattr(model.config, k) for k in keys):
            raise ValueError('Dictionary settings must match the fitted model')
    config = model.config
    bounds = dictionary_windows(x, stim_times, fs, config)
    cleaned = x.copy()
    selections = []
    skipped = 0
    for tr in range(x.shape[0]):
        for ch, templates in enumerate(model.templates):
            for start, end in bounds[tr][ch]:
                if not templates.size:
                    skipped += 1
                    continue
                epoch = _normalize(x[tr, ch, start:end], config)
                atoms = np.zeros((len(epoch), templates.shape[1]))
                n = min(len(epoch), len(templates))
                atoms[:n] = templates[:n]
                index = _best_template(epoch, atoms, model.peak_indices[ch], config)
                atom = atoms[:, index]
                amplitude = np.ptp(atom)
                if amplitude == 0:
                    skipped += 1
                    continue
                scale = np.ptp(epoch)/amplitude
                cleaned[tr, ch, start:end] -= scale*atom
                selections.append({'trial': tr, 'channel': ch, 'window': (start, end),
                                   'template': index, 'scale': float(scale)})
    return BenchmarkResult(cleaned, x-cleaned, {
        'model': model, 'windows': bounds, 'selections': selections,
        'skipped_windows': skipped, 'training_source': source,
        'processing': 'offline_full_pulse_windows',
        'window_source': 'markers' if stim_times is not None else 'detected'})
