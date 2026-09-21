"""Load frozen patient decoders with their original split and normalization."""
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from .common import write_json, derived_seed
from .models import build_model
from .training import selected_device

# Preserve sample identity and split membership when reusing trained decoders.
DATA_SETTINGS = ('data_root', 'patients', 'seed', 'target_fs', 'recording_seconds',
    'split_ratios', 'seizure_group_margin_seconds', 'split_boundary_guard_seconds',
    'two_seizure_policy', 'window_seconds', 'overlap', 'max_windows_per_class',
    'samples_per_patient')


def load_patient_decoders(patient, config, directory):
    source = Path(config['load_decoders']).resolve()/patient.patient
    directory = Path(directory)
    previous = json.loads((source/'splits.json').read_text())
    current = json.loads((directory/'splits.json').read_text())
    for key in ('patient','native_fs','target_fs','groups','seizure_intervals_samples'):
        if previous[key] != current[key]:
            raise ValueError(f'{patient.patient}: checkpoint split mismatch: {key}')
    for key in ('bytes','mtime_ns','channels','samples','fs','annotations_sha256'):
        old = json.loads((source/'input_fingerprint.json').read_text())
        new = json.loads((directory/'input_fingerprint.json').read_text())
        if old[key] != new[key]:
            raise ValueError(f'{patient.patient}: checkpoint input data mismatch: {key}')
    models, normalization, provenance = {}, None, {}
    for name in config['decoders']:
        path = source/name/'best.pt'
        saved = torch.load(path, map_location='cpu', weights_only=True)
        original_config = saved['config']
        expected_seeds = {derived_seed(original_config['seed'],patient.patient,name,'training',i)
                          for i in range(len(original_config['training']['learning_rates']))}
        if saved.get('patient',patient.patient) != patient.patient or saved['seed'] not in expected_seeds:
            raise ValueError(f'{path}: checkpoint belongs to a different patient')
        expected_samples = round(config['window_seconds']*config['target_fs'])-config['evaluation_offset_samples']
        if (saved['decoder'],saved['channels'],saved['samples']) != (name,patient.channels,expected_samples):
            raise ValueError(f'{path}: incompatible decoder/input dimensions')
        for key in DATA_SETTINGS:
            if key == 'patients':
                continue
            if saved['config'].get(key) != config.get(key):
                raise ValueError(f'{path}: incompatible data setting {key}')
        if saved['config']['evaluation_offset_samples'] != config['evaluation_offset_samples']:
            raise ValueError(f'{path}: incompatible sample offset')
        mean, scale = saved['mean'].numpy(), saved['scale'].numpy()
        if mean.shape != (patient.channels,) or scale.shape != mean.shape or not np.isfinite([mean,scale]).all() or np.any(scale<=0):
            raise ValueError(f'{path}: invalid saved normalization')
        if normalization is not None and not all(np.array_equal(a,b) for a,b in zip(normalization,(mean,scale))):
            raise ValueError('Decoder checkpoints disagree on training normalization')
        normalization = mean, scale
        model_config = dict(saved['config'], eegnet_source=config['eegnet_source'])
        model = build_model(name,patient.channels,expected_samples,model_config)
        model.load_state_dict(saved['model'])
        models[name] = model.to(selected_device(config)).eval().requires_grad_(False)
        target = directory/name/'best.pt'
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
        provenance[name] = dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                epoch=saved['epoch'],validation_loss=saved['validation_loss'],frozen=True)
    write_json(directory/'loaded_decoders.json',provenance)
    validation_labels = np.load(source/'signals/validation_labels.npy',allow_pickle=False)
    return models, normalization, validation_labels
