"""Shared synthetic input and adapters to the existing cancellation methods."""
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly
from data_loading import Dataset
from pipeline import PipelineConfig, run_pipeline
from benchmarks.methods import BenchmarkConfig, run_benchmark, fit_lrr_trials
from .common import import_source, derived_seed, write_json, array_sha256


@lru_cache(maxsize=None)
def _artifact_class(source):
    return import_source(source, 'reused_ieeg_artifact_generator').IEEGArtifactGenerator


def generate_synthetic_artifact(clean, train_rms, config, seed):
    """Generate ONCE per test window; return mixed signal and full provenance.

    Reuse the adjacent project's generator on zeros to obtain artifact alone.
    Generate at 10240 Hz then anti-alias to 1024 Hz. Scale to a random multiple
    of clean TRAINING RMS, never to test-window energy or seizure labels.
    The clean input has already been upsampled from its native 256 Hz.
    """
    cfg = config['artifact']
    fs = config['target_fs']
    rng = np.random.default_rng(seed)
    period = int(rng.choice(cfg['period_samples']))
    rate = fs/period
    channels = sorted(rng.choice(clean.shape[0], size=cfg['n_stim_channels'], replace=False).tolist())
    current = float(rng.uniform(*cfg['current_range']))
    pulse_frequency = int(rng.integers(cfg['pulse_frequency_range'][0], cfg['pulse_frequency_range'][1]+1))
    ratio = float(rng.uniform(*cfg['rms_ratio_range']))
    first = int(round(rng.uniform(*cfg['first_pulse_seconds'])*fs))
    end = clean.shape[-1]/fs - cfg['end_margin_seconds']
    trigger_times = np.arange(first/fs, end, 1/rate)
    triggers = np.rint(trigger_times*fs).astype(int)
    generator = _artifact_class(config['artifact_source'])(native_fs=fs, high_fs=cfg['high_fs'],
                                                k1_mean=cfg['k1_mean'], k2_mean=cfg['k2_mean'])
    # The reused generator uses legacy global NumPy RNG. Isolate its state so
    # artifacts cannot change model training or depend on cancellation order.
    state = np.random.get_state()
    try:
        np.random.seed(seed)
        high = generator.generate_artifact_clip(np.zeros_like(clean), channels, stim_rates=rate,
            stim_current_low=[current], stim_current_high=[current],
            f_pulse_low=[pulse_frequency], f_pulse_high=[pulse_frequency],
            stim_start=first/fs, stim_end=end, strip_distance_mean=cfg['strip_distance'],
            normal_strip_distance_mean=cfg['same_strip_distance'], delta_t_channels=0)
    finally:
        np.random.set_state(state)
    artifact = resample_poly(high, 1, cfg['high_fs']//fs, axis=-1)[:, :clean.shape[-1]]
    source_rms = float(np.sqrt(np.mean(artifact.astype(np.float64)**2)))
    if source_rms <= 0 or not np.isfinite(source_rms) or train_rms <= 0:
        raise ValueError('Cannot scale a zero/nonfinite artifact or training signal')
    multiplier = ratio*train_rms/source_rms
    artifact = (artifact*multiplier).astype(np.float32)
    mixed = (clean+artifact).astype(np.float32)
    parameters = dict(seed=int(seed), stim_channels=channels, stim_rate=rate, period_samples=period,
        first_pulse_sample=first, trigger_samples=triggers.tolist(), stim_end_seconds=end,
        current=current, pulse_frequency=pulse_frequency, training_rms=train_rms,
        artifact_rms_ratio=ratio, amplitude_multiplier=multiplier, source_artifact_rms=source_rms,
        high_fs=cfg['high_fs'], target_fs=fs)
    return mixed, artifact, parameters


def calibrate_lrr(channels, samples, train_rms, config, patient, output_dir):
    """Fit once on independent synthetic-only calibration, never neural test data."""
    artifacts, markers, records = [], [], []
    for i in range(config['cancellation']['lrr']['calibration_windows']):
        seed = derived_seed(config['seed'], patient, 'lrr_calibration', i)
        _, artifact, parameters = generate_synthetic_artifact(
            np.zeros((channels, samples), dtype=np.float32), train_rms, config, seed)
        artifacts.append(artifact)
        markers.append(parameters['trigger_samples'])
        records.append(dict(window=i, **parameters, artifact_sha256=array_sha256(artifact)))
    weights = fit_lrr_trials(np.stack(artifacts), markers, config['target_fs'],
                            BenchmarkConfig(**config['cancellation']['baseline']))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir/'lrr_weights.npy', weights)
    write_json(output_dir/'lrr_calibration.json', dict(
        source='synthetic_artifacts_only; zero neural signal; independent calibration seeds',
        weights_sha256=array_sha256(weights), parameters=records,
        pulse_window=config['cancellation']['baseline']))
    weights.flags.writeable = False
    return weights


def _pca_regression(matrix, rank, omit, pca_only_omitted=False):
    """ERAASR PCR: center, exclude target/neighbor columns, regress on PCs.

    Supports both upstream pcaOnlyOmitted modes. Computes temporal PC scores
    via the smaller Gram matrix instead of a full tall-matrix SVD.
    """
    matrix = np.asarray(matrix, dtype=float)
    centered = matrix-matrix.mean(axis=0, keepdims=True)
    if rank == 0 or matrix.shape[1] < 2:
        return centered
    gram = centered.T @ centered
    cleaned = centered.copy()
    if not pca_only_omitted:
        values, vectors = np.linalg.eigh(gram)
        valid = np.flatnonzero(values > max(0.0, values[-1])*1e-12)
        coefficients = vectors[:, valid[-rank:]]
        full_scores = centered @ coefficients
    for target in range(matrix.shape[1]):
        left, right = target-(omit-1)//2, target+int(np.ceil((omit-1)/2))
        keep = np.array([i for i in range(matrix.shape[1]) if not left <= i <= right])
        if not len(keep):
            continue
        if pca_only_omitted:
            values, vectors = np.linalg.eigh(gram[np.ix_(keep, keep)])
            valid = np.flatnonzero(values > max(0.0, values[-1])*1e-12)
            if not len(valid):
                continue
            pcs = centered[:, keep] @ vectors[:, valid[-rank:]]
        else:
            excluded = np.setdiff1d(np.arange(matrix.shape[1]), keep)
            pcs = full_scores-centered[:, excluded] @ coefficients[excluded]
        cleaned[:, target] -= pcs @ np.linalg.lstsq(pcs, centered[:, target], rcond=None)[0]
    return cleaned


def eraasr_single_trial(x, parameters, options):
    """Single-trial ERAASR adaptation: sequential channel and pulse PCR.

    Based on djoshea/eraasr cleanArtifactTensor/cleanMatrixViaPCARegression,
    commit b5af88ae6388ccd57a4c7f28fa2c6af7b944c439. Known synthetic integer
    pulse markers replace onset detection/alignment. With one independent
    test window, the across-trial stage is inapplicable and is explicitly
    disabled (nPC_trials=0); no unrelated test trials are pooled. No HP filter
    or post-stim PCR; full pulse periods are cleaned with continuity restored.
    This is a documented Python adaptation, not the original MATLAB pipeline.
    """
    x = np.asarray(x, dtype=float)
    start, width = parameters['first_pulse_sample'], parameters['period_samples']
    pulses = min(len(parameters['trigger_samples']), (x.shape[-1]-start)//width)
    if pulses < 2:
        raise ValueError('ERAASR requires at least two complete pulse periods')
    stop = start+pulses*width
    raw = x[:, start:stop].reshape(x.shape[0], pulses, width)
    tensor = raw-raw.mean(axis=2, keepdims=True)
    if options['channel_rank']:
        matrix = tensor.transpose(1,2,0).reshape(-1,x.shape[0])
        tensor = _pca_regression(matrix, options['channel_rank'], options['omit_channels'], options['pca_only_omitted']).reshape(pulses,width,-1).transpose(2,0,1)
        tensor = tensor-tensor.mean(axis=2, keepdims=True)
    if options['pulse_rank']:
        matrix = tensor.transpose(0,2,1).reshape(-1,pulses)
        tensor = _pca_regression(matrix, options['pulse_rank'], options['omit_pulses'], options['pca_only_omitted']).reshape(x.shape[0],width,pulses).transpose(0,2,1)
        tensor = tensor-tensor.mean(axis=2, keepdims=True)
    # Upstream continuity convention: anchor the first period and join later
    # periods, then shift post-stim samples by the final correction.
    tensor[:, 0] += (raw[:, 0, 0]-tensor[:, 0, 0])[:, None]
    for pulse in range(1,pulses):
        tensor[:, pulse] += (tensor[:, pulse-1,-1]-tensor[:, pulse,0])[:, None]
    cleaned = x.copy()
    cleaned[:, start:stop] = tensor.reshape(x.shape[0], -1)
    if stop < x.shape[-1]:
        cleaned[:, stop:] += (cleaned[:, stop-1]-x[:, stop-1])[:, None]
    return cleaned


def apply_artifact_cancellation(contaminated, method, parameters, config, lrr_weights=None, pulse_model=None):
    """No access to clean test signals or seizure labels; fixed parameters.

    Every method returns the common decoder grid, cropped by the DAST finite
    difference offset. Input is copied so an adapter cannot mutate shared data.
    """
    x = np.array(contaminated, dtype=np.float32, copy=True)
    offset = config['evaluation_offset_samples']
    dataset = Dataset(name='heldout_swec', X=x[None], fs=config['target_fs'],
                      stim_rate=parameters['stim_rate'])
    if method == 'DAST':
        result = run_pipeline(dataset, PipelineConfig(n_test=0, **config['cancellation']['dast']))
        output = result.stages['cleaned'][0]
    elif method == 'pulse':
        from .pulse_adapter import apply_pulse_checkpoint
        output, _ = apply_pulse_checkpoint(x,parameters,config,pulse_model)
        output = output[:, offset:]
    elif method == 'ERAASR':
        output = eraasr_single_trial(x, parameters, config['cancellation']['eraasr'])[:, offset:]
    else:
        mapped = {'SVD':'window_svd', 'LRR':'lrr'}.get(method, method)
        settings = dict(config['cancellation']['baseline'])
        settings.update(config['cancellation'].get('method_overrides', {}).get(method, {}))
        baseline = BenchmarkConfig(method=mapped, **settings)
        kwargs = {'W': lrr_weights} if mapped == 'lrr' else {}
        output = run_benchmark(dataset, baseline, parameters['trigger_samples'], **kwargs).cleaned[0, :, offset:]
    if output.shape != (x.shape[0], x.shape[1]-offset) or not np.isfinite(output).all():
        raise ValueError(f'{method} produced invalid/misaligned output')
    return output.astype(np.float32)
