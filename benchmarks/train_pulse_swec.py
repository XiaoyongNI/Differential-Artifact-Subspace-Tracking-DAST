"""Train/reuse PULSE across SWEC synthetic stimulation configurations.

The source archive is streamed into a cropped float32 cache once. Original
recording trial IDs, not their artifact variants, define the train/test split.
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
import zipfile

import numpy as np
import torch

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_loading import ROOT
from benchmarks.methods import BenchmarkConfig
from benchmarks.pulse_nn import fit_pulse, load_pulse, pulse, save_pulse


FS = 10240.
MAX_CHANNELS = 128


def pad_channels(recordings, channels=MAX_CHANNELS):
    """Append zero recording channels, preserving the original channel order."""
    recordings = np.asarray(recordings)
    if recordings.ndim not in (2, 3):
        raise ValueError('Expected (channels, samples) or (trials, channels, samples)')
    count = recordings.shape[-2]
    if not 1 <= count <= channels:
        raise ValueError(f'Expected 1–{channels} recording channels; got {count}')
    padding = [(0, 0)] * recordings.ndim
    padding[-2] = (0, channels-count)
    return np.pad(recordings, padding)


def prepare_cache(root, directory):
    """Use the same recording, component sum, and crop as load_swec_synthetic."""
    source = next((root/'data/swec_synthetic').glob('StimChannels*.npz'))
    signature = {'path': str(source.resolve()), 'size': source.stat().st_size,
                 'mtime_ns': source.stat().st_mtime_ns}
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory/'source.json'
    if manifest.exists() and json.loads(manifest.read_text())['source'] == signature:
        return (np.load(directory/'clean.npy', mmap_mode='r'),
                np.load(directory/'artifacts.npy', mmap_mode='r'),
                np.load(directory/'params.npy'), json.loads(manifest.read_text()))
    with np.load(source) as data:
        params = data['seizure_stim_params']
    first = int(params[0, 0, 4]*FS)
    stop = max(first+1, int(params[0, 0, 5]*FS)-1000)
    with zipfile.ZipFile(source) as archive:
        for member, filename in [('seizure_data.npy', 'clean.npy'), ('seizure_artifact.npy', 'artifacts.npy')]:
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
                if fortran or dtype.hasobject:
                    raise ValueError('Expected numeric C-order SWEC tensors')
                out_shape = (*shape[:-1], stop-first)
                target = np.lib.format.open_memmap(directory/filename, mode='w+', dtype=np.float32, shape=out_shape)
                n = int(np.prod(shape[-2:]))
                flat = target.reshape(-1, *out_shape[-2:])
                for idx in range(flat.shape[0]):
                    chunk = stream.read(n*dtype.itemsize)
                    if len(chunk) != n*dtype.itemsize:
                        raise ValueError('Truncated SWEC archive')
                    flat[idx] = np.frombuffer(chunk, dtype=dtype).reshape(shape[-2:])[:, first:stop]
                target.flush()
            print(f'Cached {filename}: {out_shape}', flush=True)
    np.save(directory/'params.npy', params)
    details = {'source': signature, 'fs': FS, 'crop': [first, stop],
               'stim_channels': params[:, 0, 2].astype(int).tolist(), 'dtype': 'float32'}
    manifest.write_text(json.dumps(details, indent=2)+'\n')
    return np.load(directory/'clean.npy', mmap_mode='r'), np.load(directory/'artifacts.npy', mmap_mode='r'), params, details


def stimulation_trace(params, trial, channels, first, length, fs=FS):
    """Reconstruct the normalized biphasic trace from synthetic pulse metadata.

    Shared by checkpoint training and downstream matched-artifact inference.
    """
    ids = params[:, trial, 2].astype(int).tolist()
    trace = np.zeros(length, dtype=np.float32)
    time = (np.arange(length)+first)/fs
    for channel in channels:
        if channel not in ids:
            raise ValueError(f'Stimulation channel {channel} is unavailable; choose from {ids}')
        rate, current, _, frequency, onset, end = params[ids.index(channel), trial]
        for t in np.arange(onset, end, 1/rate):
            trace[(time >= t) & (time < t+.5/frequency)] -= current
            trace[(time >= t+.5/frequency) & (time < t+1/frequency)] += current
    trace /= max(float(np.max(np.abs(trace))), 1.)
    return trace[None]


def make_example(clean, artifacts, params, trial, channels, first):
    """Component sum with a metadata-derived, normalized aggregate biphasic trace."""
    ids = params[:, trial, 2].astype(int).tolist()
    mixed = clean[trial].copy()
    trace = stimulation_trace(params, trial, channels, first, clean.shape[-1])
    for channel in channels:
        mixed += artifacts[ids.index(channel), trial]
    return mixed, trace


def configuration_markers(params, trial, channels, first, length):
    indices = []
    for channel in channels:
        row = params[np.flatnonzero(params[:, trial, 2] == channel)[0], trial]
        # First acquisition sample at or after the pulse onset.
        times = np.ceil(np.arange(row[4], row[5], 1/row[0])*FS-1e-9).astype(int)-first
        indices.extend(times[(times >= 0) & (times < length)].tolist())
    return np.unique(indices)


def evaluation(model, clean, artifacts, params, first, trials, configurations, directory):
    rows = []
    for channels in configurations:
        examples = [make_example(clean, artifacts, params, tr, channels, first) for tr in trials]
        mixed = np.stack([a for a, _ in examples])
        trace = np.stack([b for _, b in examples])
        markers = [configuration_markers(params, tr, channels, first, mixed.shape[-1]) for tr in trials]
        begin = perf_counter()
        result = pulse(pad_channels(mixed, model.network.out_channels), markers, FS,
                       model=model, stim_trace=trace)
        cleaned = result.cleaned[:, :mixed.shape[1]]
        artifact = result.artifact[:, :mixed.shape[1]]
        elapsed = perf_counter()-begin
        for i, tr in enumerate(trials):
            gt = np.asarray(clean[tr], dtype=np.float64)
            before = np.mean((mixed[i]-gt)**2)
            after = np.mean((cleaned[i]-gt)**2)
            power = np.mean(gt**2)
            rows.append({'trial': int(tr), 'stim_channels': '_'.join(map(str, channels)),
                         'rmse_before': float(np.sqrt(before)), 'rmse_after': float(np.sqrt(after)),
                         'relative_rmse_after': float(np.sqrt(after/power)),
                         'snr_improvement_db': float(10*np.log10(before/after)),
                         'inference_seconds_per_trial': elapsed/len(trials)})
        print(f'Test sites {channels}: mean SNR improvement {np.mean([r["snr_improvement_db"] for r in rows[-len(trials):]]):.3f} dB', flush=True)
        if len(configurations) == 1 or channels == [0, 1, 2, 8]:
            np.savez_compressed(directory/'test_outputs.npz', cleaned=cleaned, raw=mixed,
                                y_clean=clean[trials], artifact=artifact, stim_trace=trace,
                                artifact_mask=result.diagnostics['artifact_mask'], trial_indices=trials,
                                stim_channels=channels, fs=FS)
    with (directory/'test_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--cache-dir', type=Path, default=ROOT/'results/pulse_swec_cache')
    parser.add_argument('--device', default='cuda:1' if torch.cuda.device_count() > 1 else 'auto')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stim-channels', nargs='+', type=int, default=[0, 1, 2, 8])
    parser.add_argument('--trials', nargs='+', type=int, default=[16, 17, 18, 19])
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = args.root/'results'/('pulse_swec_synthetic' if args.mode == 'train' else 'pulse_swec_test')
    torch.set_num_threads(4)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    clean, artifacts, params, source = prepare_cache(args.root, args.cache_dir)
    if clean.ndim != 3 or clean.shape[0] != 20 or not 1 <= clean.shape[1] <= MAX_CHANNELS:
        raise ValueError('This experiment expects a 20-trial SWEC synthetic archive with 1–128 recording channels')
    checkpoint = args.checkpoint or ((args.root/'results/pulse_swec_synthetic' if args.mode == 'test' else args.out_dir)
                                     /'pulse_swec_synthetic.pt')
    first = source['crop'][0]
    if args.mode == 'test':
        if len(set(args.stim_channels)) != len(args.stim_channels) or not args.stim_channels:
            parser.error('Provide distinct stimulation channels')
        if any(tr < 0 or tr >= len(clean) for tr in args.trials):
            parser.error('Trial indices must lie in [0, 19]')
        model = load_pulse(checkpoint, args.device)
        evaluation(model, clean, artifacts, params, first, args.trials, [args.stim_channels], args.out_dir)
        (args.out_dir/'test_manifest.json').write_text(json.dumps({
            'checkpoint': str(checkpoint.resolve()),
            'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'mode': 'inference_only', 'trials': args.trials, 'stim_channels': args.stim_channels,
            'source': source, 'config': asdict(model.config),
        }, indent=2)+'\n')
        return
    rng = np.random.default_rng(args.seed)
    available = source['stim_channels']
    train_ids, test_ids = list(range(16)), list(range(16, 20))
    manifest = []
    examples = []
    # Each training recording sees one of all 16 single sites, a reproducible
    # 2/4/8-site combination, and the loader's default four-site configuration.
    for tr in train_ids:
        combinations = [[available[tr]], sorted(rng.choice(available, [2, 4, 8][tr % 3], replace=False).tolist()), [0, 1, 2, 8]]
        for channels in combinations:
            examples.append(make_example(clean, artifacts, params, tr, channels, first))
            manifest.append({'trial': tr, 'stim_channels': channels})
    mixed = np.stack([pad_channels(a) for a, _ in examples])
    traces = np.stack([b for _, b in examples])
    del examples
    config = BenchmarkConfig(method='pulse', random_state=args.seed, pulse_epochs=args.epochs,
                             pulse_batch_size=args.batch_size, pulse_device=args.device,
                             pulse_f_cutoff=200., pulse_artifact_duration_ms=4.)
    metadata = {'source': source, 'training_examples': manifest, 'train_trials': train_ids,
                'test_trials': test_ids, 'config': asdict(config),
                'trace_source': 'normalized aggregate biphasic waveform reconstructed from saved stimulation parameters',
                'artifact_scale': 1., 'ground_truth_usage': 'test metrics only',
                'recording_channels': int(clean.shape[1]), 'padded_channels': MAX_CHANNELS,
                'training_shape': list(mixed.shape), 'status': 'training'}
    metadata_path = args.out_dir/'training_manifest.json'
    metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Training {mixed.shape}, device={args.device}, epochs={args.epochs}', flush=True)
    begin = perf_counter()
    with (args.out_dir/'training_log.csv').open('w', buffering=1) as log:
        log.write('epoch,loss,elapsed_seconds\n')
        def progress(epoch, total, loss):
            seconds = perf_counter()-begin
            log.write(f'{epoch},{loss},{seconds}\n')
            if epoch == 1 or epoch % 10 == 0 or epoch == total:
                print(f'Epoch {epoch}/{total}, loss={loss:.6g}, elapsed={seconds:.1f}s', flush=True)
        model = fit_pulse(mixed, traces, FS, config, progress=progress, checkpoint_path=checkpoint)
    save_pulse(model, checkpoint)
    metadata['training_seconds'] = perf_counter()-begin
    metadata['status'] = 'trained'
    metadata['checkpoint'] = str(checkpoint.resolve())
    metadata['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
    # Load from disk for every reported test metric: this verifies the saved artifact.
    restored = load_pulse(checkpoint, args.device)
    configurations = [[ch] for ch in available] + [[0, 1, 2, 8], [0, 2, 9, 16, 24, 26, 84, 87]]
    rows = evaluation(restored, clean, artifacts, params, first, test_ids, configurations, args.out_dir)
    metadata['test_mean_snr_improvement_db'] = float(np.mean([r['snr_improvement_db'] for r in rows]))
    metadata['status'] = 'complete'
    metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Saved reusable checkpoint: {checkpoint}', flush=True)


if __name__ == '__main__':
    main()
