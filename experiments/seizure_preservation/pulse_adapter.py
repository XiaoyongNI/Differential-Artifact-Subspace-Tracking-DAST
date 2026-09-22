"""Frozen PULSE checkpoint inference on the common contaminated samples.

The external checkpoint has a fixed channel count and sampling rate. No fit,
normalization refit, or decoder update is performed here.
"""
from fractions import Fraction
from pathlib import Path
import hashlib
import numpy as np
import torch
from scipy.signal import resample_poly
from benchmarks.pulse_nn import load_pulse, pulse
from benchmarks.train_pulse_swec import stimulation_trace, pad_channels
from benchmarks.methods import BenchmarkConfig
from .common import write_json


def checkpoint_path(config, patient):
    options = config['cancellation']['pulse']
    return Path(options.get('checkpoints', {}).get(patient, options['checkpoint'])).resolve()


def inspect_checkpoint(config, patient, channels):
    path = checkpoint_path(config, patient)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    expected = int(payload['model_config']['out_channels'])
    fs = float(payload['training_config']['sampling_rate'])
    if not 1 <= channels <= expected:
        raise ValueError(f'{patient}: PULSE checkpoint supports 1–{expected} channels; '
                         f'patient has {channels}. Real channels cannot be dropped.')
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError('Invalid PULSE checkpoint sampling rate')
    return dict(checkpoint=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                checkpoint_fs=fs, channels=expected, input_channels=channels,
                padded_channels=expected-channels)


def prepare_pulse(config, patient, channels, output_dir):
    description = inspect_checkpoint(config, patient, channels)
    options = config['cancellation']['pulse']
    device = config['training']['device'] if options['device']=='same_as_decoder' else options['device']
    model = load_pulse(description['checkpoint'], device=device)
    description.update(input_fs=config['target_fs'], frozen=True, trained_in_this_run=False,
        channel_policy='append zero channels using training pad_channels; retain original output channels',
        trace_source='synthetic pulse metadata for the SAME artifact realization; shared stimulation_trace helper',
        sampling_adapter='polyphase resampling to checkpoint rate; subtract resampled predicted correction on original grid')
    write_json(Path(output_dir)/'pulse_checkpoint.json',description)
    return model


def matched_trace(parameters, length, fs):
    rows = [[[parameters['stim_rate'],parameters['current'],channel,parameters['pulse_frequency'],
              parameters['first_pulse_sample']/parameters['target_fs'],parameters['stim_end_seconds']]]
            for channel in parameters['stim_channels']]
    return stimulation_trace(np.asarray(rows),0,parameters['stim_channels'],0,length,fs)


def apply_pulse_checkpoint(contaminated, parameters, config, model):
    if model is None:
        raise ValueError('PULSE requires the configured frozen checkpoint; training is never automatic')
    if model.network.training or any(p.requires_grad for p in model.network.parameters()):
        raise ValueError('PULSE checkpoint must be frozen in eval mode')
    ratio = Fraction(str(model.fs))/Fraction(str(config['target_fs']))
    x = np.asarray(contaminated,dtype=np.float32)
    padded = pad_channels(x, model.network.out_channels)
    up = resample_poly(padded,ratio.numerator,ratio.denominator,axis=-1)
    trace = matched_trace(parameters,up.shape[-1],model.fs)
    onsets = np.arange(parameters['first_pulse_sample']/parameters['target_fs'],
                       parameters['stim_end_seconds'],1/parameters['stim_rate'])
    markers = np.ceil(onsets*model.fs-1e-9).astype(int)
    markers = np.unique(markers[(markers>=0)&(markers<up.shape[-1])])
    options = BenchmarkConfig(pulse_batch_size=config['cancellation']['pulse']['batch_size'])
    result = pulse(up[None],markers,model.fs,options,model=model,stim_trace=trace[None])
    # Resample the correction, not the whole output: an identity checkpoint
    # retains exactly x rather than adding an up/downsampling filter effect.
    correction = resample_poly(result.artifact[0, :x.shape[0]],ratio.denominator,ratio.numerator,axis=-1)[:, :x.shape[-1]]
    return (x-correction).astype(np.float32), trace
