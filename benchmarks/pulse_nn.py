"""PULSE residual U-Net with the local source's unsupervised physics losses.

Training is separate from inference. Network/loss implementations are vendored
unchanged from paper/pulse/sparc-nn; this module adapts the TBME array interface.
PyTorch is imported only when this neural benchmark is requested.
"""
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import numpy as np
import torch

from ._pulse_unet import UNet1D
from ._pulse_physics import PhysicsLoss
from ._pulse_uncertainty import UncertaintyWeightedLoss
from .methods import BenchmarkConfig, BenchmarkResult, validate_signal, pulse_times


LOSS_KEYS = ('cosine', 'rank_a', 'rank_s_penalty', 'spectral', 'spectral_slope_s')


def _device(name):
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else torch.device(name)


def _validate_config(config, fs):
    for key in ('pulse_epochs', 'pulse_batch_size'):
        value = getattr(config, key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{key} must be a positive integer')
    if (not isinstance(config.pulse_blur_samples, int) or isinstance(config.pulse_blur_samples, bool)
            or config.pulse_blur_samples < 0):
        raise ValueError('pulse_blur_samples must be a nonnegative integer')
    positive = [fs, config.pulse_learning_rate, config.pulse_artifact_duration_ms, config.pulse_f_cutoff]
    if not np.isfinite(positive).all() or min(positive) <= 0 or config.pulse_f_cutoff >= fs/2:
        raise ValueError('fs, learning rate, artifact duration and cutoff must be positive; cutoff must be below Nyquist')
    weights = _loss_weights(config)
    if not np.isfinite(list(weights.values())).all() or min(weights.values()) < 0 or sum(weights.values()) <= 0:
        raise ValueError('PULSE loss weights must be finite, nonnegative and not all zero')


def _loss_weights(config):
    return dict(zip(LOSS_KEYS, (config.pulse_w_cosine, config.pulse_w_rank_a,
                               config.pulse_w_rank_s, config.pulse_w_spectral,
                               config.pulse_w_spectral_slope)))


def _trace(stim_trace, shape):
    if stim_trace is None:
        raise ValueError('PULSE requires a stimulation trace shaped (trials, 1, samples)')
    trace = np.asarray(stim_trace, dtype=np.float32)
    if trace.shape != (shape[0], 1, shape[2]) or not np.isfinite(trace).all():
        raise ValueError('stim_trace must be finite with shape (trials, 1, samples) matching X')
    return trace


def pulse_trace_from_markers(stim_times, n_trials, n_samples):
    """Explicit binary-impulse surrogate when no measured stimulation trace exists."""
    trace = np.zeros((n_trials, 1, n_samples), dtype=np.float32)
    for tr, times in enumerate(pulse_times(stim_times, n_trials, n_samples)):
        trace[tr, 0, times] = 1.
    return trace


def pulse_artifact_mask(stim_times, n_trials, n_samples, fs, duration_ms=40., blur_samples=5):
    """Source soft mask: onset-to-duration interval plus linear edge ramps."""
    if not np.isfinite([fs, duration_ms]).all() or fs <= 0 or duration_ms <= 0:
        raise ValueError('fs and duration_ms must be finite and positive')
    if not isinstance(blur_samples, int) or isinstance(blur_samples, bool) or blur_samples < 0:
        raise ValueError('blur_samples must be a nonnegative integer')
    duration = int(duration_ms*fs/1000)
    if duration < 1:
        raise ValueError('PULSE artifact duration must cover at least one sample')
    mask = np.zeros((n_trials, 1, n_samples), dtype=np.float32)
    for tr, times in enumerate(pulse_times(stim_times, n_trials, n_samples)):
        for start in times:
            end = min(int(start)+duration, n_samples)
            mask[tr, 0, start:end] = 1.
            for j in range(1, blur_samples+1):
                value = 1-j/blur_samples
                if start-j >= 0:
                    mask[tr, 0, start-j] = max(mask[tr, 0, start-j], value)
                if end+j-1 < n_samples:
                    mask[tr, 0, end+j-1] = max(mask[tr, 0, end+j-1], value)
    return mask


def _signal(x):
    x = validate_signal(x)
    if x.shape[-1] < 16:
        raise ValueError('PULSE requires at least 16 samples per trial (three pools and instance normalization)')
    result = x.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('PULSE inputs must be representable in float32')
    return result


@dataclass
class PULSEModel:
    network: UNet1D
    mean: torch.Tensor
    std: torch.Tensor
    fs: float
    config: BenchmarkConfig
    history: list
    uncertainty_state: dict | None = None


def fit_pulse(X_train, stim_trace, fs, config=None, progress=None):
    """Train only on mixed recordings and stimulus traces, without clean labels.

    Whole trials are minibatch items, as in the upstream default train.py path.
    Median/IQR normalization is fitted on training data only. Optional progress
    receives (completed_epoch, total_epochs, mean_loss).
    """
    config = replace(config or BenchmarkConfig(method='pulse'))
    _validate_config(config, fs)
    x = _signal(X_train)
    trace = _trace(stim_trace, x.shape)
    data = torch.from_numpy(x)
    flat = data.transpose(0, 1).reshape(x.shape[1], -1)
    mean = torch.quantile(flat, .5, dim=1)[None, :, None]
    std = ((torch.quantile(flat, .75, dim=1)-torch.quantile(flat, .25, dim=1))/1.3489).clamp(min=1e-6)[None, :, None]
    normalized = (data-mean)/(std+1e-8)
    traces = torch.from_numpy(trace)
    device = _device(config.pulse_device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    # Restore caller RNG state after model initialization and training.
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.random_state)
        network = UNet1D(x.shape[1]+1, x.shape[1]).to(device)
        criterion = PhysicsLoss(sampling_rate=fs, f_cutoff=config.pulse_f_cutoff).to(device)
        uncertainty = UncertaintyWeightedLoss(5).to(device) if config.pulse_uncertainty else None
        parameters = list(network.parameters()) + ([] if uncertainty is None else list(uncertainty.parameters()))
        optimizer = torch.optim.Adam(parameters, lr=config.pulse_learning_rate)
        generator = torch.Generator().manual_seed(config.random_state)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(normalized, traces),
                    batch_size=config.pulse_batch_size, shuffle=True, generator=generator)
        history = []
        for epoch in range(config.pulse_epochs):
            network.train()
            total = 0.
            components = dict.fromkeys(LOSS_KEYS, 0.)
            for batch, stimulation in loader:
                batch, stimulation = batch.to(device), stimulation.to(device)
                prediction = network(batch, stimulation)
                neural = prediction if config.pulse_predict_neural else batch-prediction
                artifact = batch-prediction if config.pulse_predict_neural else prediction
                raw = criterion(neural, artifact)
                weights = _loss_weights(config)
                loss = (sum(weights[k]*raw[k] for k in LOSS_KEYS) if uncertainty is None
                        else uncertainty(raw, priority_weights=weights))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite PULSE training loss at epoch {epoch+1}')
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1., error_if_nonfinite=True)
                optimizer.step()
                total += loss.item()
                for key in components:
                    components[key] += raw[key].item()
            history.append({'total': total/len(loader), **{k: v/len(loader) for k, v in components.items()}})
            if progress is not None:
                progress(epoch+1, config.pulse_epochs, history[-1]['total'])
        network.eval()
        network.requires_grad_(False)
        state = None if uncertainty is None else {k: v.detach().cpu() for k, v in uncertainty.state_dict().items()}
    return PULSEModel(network, mean, std, float(fs), config, history, state)


def pulse(x, stim_times=None, fs=None, config=None, *, model=None, stim_trace=None, artifact_mask=None):
    """Frozen U-Net inference over whole trials, with source-style mask blending."""
    raw = validate_signal(x)
    data = _signal(raw)
    if not isinstance(model, PULSEModel):
        raise ValueError('PULSE requires a trained PULSEModel; use fit_pulse or load_pulse')
    if data.shape[1] != model.network.out_channels or fs != model.fs:
        raise ValueError('PULSE channel count and sampling rate must match training')
    trace = _trace(stim_trace, data.shape)
    if artifact_mask is None:
        artifact_mask = pulse_artifact_mask(stim_times, data.shape[0], data.shape[-1], fs,
                    model.config.pulse_artifact_duration_ms, model.config.pulse_blur_samples)
    mask = np.asarray(artifact_mask, dtype=np.float32)
    if mask.shape != trace.shape or not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
        raise ValueError('artifact_mask must have shape (trials, 1, samples) with values in [0, 1]')
    batch_size = model.config.pulse_batch_size if config is None else config.pulse_batch_size
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('pulse_batch_size must be positive')
    device = next(model.network.parameters()).device
    mean, scale = model.mean.to(device), (model.std+1e-8).to(device)
    cleaned = raw.copy()
    model.network.eval()
    with torch.inference_mode():
        for start in range(0, len(data), batch_size):
            stop = start+batch_size
            batch = torch.from_numpy(data[start:stop]).to(device)
            normalized = (batch-mean)/scale
            prediction = model.network(normalized, torch.from_numpy(trace[start:stop]).to(device))
            neural = prediction if model.config.pulse_predict_neural else normalized-prediction
            neural_raw = (neural*scale+mean).cpu().numpy()
            # Preserve exact original samples where mask==0; report the removed
            # component as input-cleaned (do not add the mean to both outputs).
            cleaned[start:stop] = raw[start:stop] + mask[start:stop]*(neural_raw-raw[start:stop])
    if not np.isfinite(cleaned).all():
        raise FloatingPointError('PULSE inference produced nonfinite samples')
    return BenchmarkResult(cleaned, raw-cleaned, {
        'model': model, 'stim_trace': trace.copy(), 'artifact_mask': mask.copy(),
        'implementation': 'PULSE residual UNet1D + PhysicsLoss',
        'processing': 'offline_full_trial', 'predict_neural': model.config.pulse_predict_neural,
        'training_history': model.history})


def save_pulse(model, path):
    """Save an upstream-shaped checkpoint including training-only normalization."""
    payload = {
        'model_state_dict': {k: v.detach().cpu() for k, v in model.network.state_dict().items()},
        'model_config': {'in_channels': model.network.in_channels, 'out_channels': model.network.out_channels},
        'training_config': {'sampling_rate': model.fs, 'predict_neural': model.config.pulse_predict_neural,
                            'artifact_duration_ms': model.config.pulse_artifact_duration_ms,
                            'learning_rate': model.config.pulse_learning_rate,
                            'batch_size': model.config.pulse_batch_size, 'num_epochs': model.config.pulse_epochs},
        'benchmark_config': asdict(model.config), 'data_mean': model.mean.cpu(), 'data_std': model.std.cpu(),
        'training_history': model.history, 'format_version': 1,
        'loss_config': {'f_cutoff': model.config.pulse_f_cutoff,
                        'use_uncertainty_loss': model.config.pulse_uncertainty, 'use_expert': False,
                        'loss_weights': {k.removeprefix('pulse_'): v for k, v in asdict(model.config).items()
                                         if k.startswith('pulse_w_')}},
    }
    if model.uncertainty_state is not None:
        payload['uncertainty_loss_state_dict'] = model.uncertainty_state
    torch.save(payload, Path(path))


def load_pulse(path, device='auto'):
    """Load a benchmark or source train.py checkpoint, strictly and without refitting."""
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    for key in ('model_state_dict', 'model_config', 'training_config', 'data_mean', 'data_std'):
        if key not in payload:
            raise ValueError(f'PULSE checkpoint missing {key}')
    architecture, training = payload['model_config'], payload['training_config']
    channels = architecture['out_channels']
    if architecture['in_channels'] != channels+1:
        raise ValueError('PULSE checkpoint must use C recording channels plus one stimulation trace')
    config = BenchmarkConfig(**payload.get('benchmark_config', {}))
    config.pulse_predict_neural = training.get('predict_neural', False)
    config.pulse_artifact_duration_ms = training.get('artifact_duration_ms', 40.)
    config.pulse_f_cutoff = payload.get('loss_config', {}).get('f_cutoff', 10.)
    loss_config = payload.get('loss_config', {})
    if loss_config.get('use_expert', False):
        raise ValueError('Expert-guided checkpoints are outside this unsupervised PULSE baseline')
    config.pulse_uncertainty = loss_config.get('use_uncertainty_loss', False)
    for source, target in [('learning_rate', 'pulse_learning_rate'), ('batch_size', 'pulse_batch_size'),
                           ('num_epochs', 'pulse_epochs')]:
        if source in training:
            setattr(config, target, training[source])
    for key, value in loss_config.get('loss_weights', {}).items():
        if hasattr(config, f'pulse_{key}'):
            setattr(config, f'pulse_{key}', value)
    config.pulse_device = device
    fs = float(training['sampling_rate'])
    _validate_config(config, fs)
    mean, std = torch.as_tensor(payload['data_mean']).float(), torch.as_tensor(payload['data_std']).float()
    if (mean.shape != (1, channels, 1) or std.shape != mean.shape
            or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or torch.any(std <= 0)):
        raise ValueError('Checkpoint normalization must be finite (1, C, 1) with positive std')
    with torch.random.fork_rng(devices=[]):
        network = UNet1D(channels+1, channels)
    network.load_state_dict(payload['model_state_dict'], strict=True)
    if not all(torch.isfinite(t).all() for t in network.state_dict().values()):
        raise ValueError('Checkpoint contains nonfinite weights')
    network.to(_device(device)).eval().requires_grad_(False)
    history = payload.get('training_history', [{'total': float(v)} for v in payload.get('loss_history', [])])
    return PULSEModel(network, mean, std, fs, config, history,
                      payload.get('uncertainty_loss_state_dict'))
