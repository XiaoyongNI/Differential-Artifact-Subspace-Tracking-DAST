"""Apply the pipeline harmonic stage to complete saved PBS DAST outputs.

Run after tests/run_pbs_dast_table.py:
    python tests/add_pbs_harmonic_filter.py

Requires unnormalized, unfiltered DAST outputs with no evaluation burn-in.
This preserves filter startup and is equivalent to enabling the harmonic stage
in run_pipeline with the per-configuration harmonic count below.
"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from algorithms import harmonic_rls_3d, build_harmonic_freqs
from data_loading import ROOT
from experiments.seizure_preservation.common import write_csv, write_json, array_sha256
from run_pbs_dast_table import CONDITIONS, reconstruction_metrics, render_table


def postfilter(x, fs, rate, harmonics=10, settling_time=.01):
    if not np.isfinite(fs) or fs <= 0 or not np.isfinite(rate) or rate < 0:
        raise ValueError('Require positive fs and nonnegative stimulation rate')
    if harmonics < 1 or not np.isfinite(settling_time) or settling_time <= 0:
        raise ValueError('Require positive harmonic count and settling time')
    count = len(build_harmonic_freqs(rate, fs, harmonics)) if rate else 0
    if count == 0:
        return x.copy(), count
    return harmonic_rls_3d(x, fs, rate, max_harmonics=count,
                           settling_time=settling_time), count


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir', type=Path, default=ROOT/'results/pbs_dast_table')
    p.add_argument('--output-dir', type=Path, default=ROOT/'results/pbs_dast_harmonic_table')
    p.add_argument('--harmonics', type=int, default=10)
    p.add_argument('--settling-time', type=float, default=.01, help='Seconds')
    args = p.parse_args()
    if args.source_dir.resolve() == args.output_dir.resolve():
        p.error('Source and output directories must differ')
    protocol = json.loads((args.source_dir/'protocol.json').read_text())
    config = protocol['pipeline']
    if (config['normalization'] != 'none' or config['harmonic_filter']
            or not config['use_derivative_for_tracking'] or config['derivative_order'] != 1
            or protocol['evaluation']['first_sample'] != 1):
        raise ValueError('Need complete first-difference DAST outputs, no normalization or harmonic stage')
    rows, channel_rows, settings = [], [], []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = {s['configuration']: s for s in protocol['sources']}
    for c in range(14):
        with np.load(args.source_dir/f'configuration_{c:02d}.npz') as saved:
            data = {key: saved[key] for key in saved.files}
        reference, before = data['reference'], data['cleaned']
        fs = float(data['fs'])
        np.testing.assert_array_equal(data['original_sample_indices'],
                                     np.arange(1, sources[c]['used_samples']))
        if array_sha256(reference) != sources[c]['clean_sha256']:
            raise ValueError(f'Config {c}: reference checksum differs from protocol')
        rate = sources[c]['stim_rate_hz']
        after, count = postfilter(before, fs, rate, args.harmonics, args.settling_time)
        metrics = reconstruction_metrics(after, reference)
        dast = reconstruction_metrics(before, reference)
        raw = reconstruction_metrics(data['contaminated'], reference)
        rows.append(dict(configuration=c, condition=CONDITIONS[c], **metrics,
                         dast_only_mse_uv2=dast['mse_uv2'], dast_only_snr_db=dast['snr_db'],
                         harmonic_snr_change_db=metrics['snr_db']-dast['snr_db'],
                         input_mse_uv2=raw['mse_uv2'], input_snr_db=raw['snr_db'],
                         snr_improvement_db=metrics['snr_db']-raw['snr_db'],
                         stim_rate_hz=rate, harmonics=count, settling_time_s=args.settling_time,
                         n_samples=reference.shape[-1], n_channels=reference.shape[1], fs=fs))
        for ch in range(reference.shape[1]):
            channel_rows.append(dict(configuration=c, recording_channel=ch,
                                      **reconstruction_metrics(after[:, ch], reference[:, ch])))
        data.update(after_subspace=before, cleaned=after)
        np.savez_compressed(args.output_dir/f'configuration_{c:02d}.npz', **data)
        settings.append(dict(configuration=c, harmonic_filter=count > 0,
                             harmonic_max_harmonics=count, stim_rate_hz=rate,
                             source_dast_sha256=array_sha256(before)))
        print(f"{c:2}: MSE {metrics['mse_uv2']:.2f}; SNR {metrics['snr_db']:.2f}; "
              f"change {metrics['snr_db']-dast['snr_db']:+.2f} dB", flush=True)
    write_csv(args.output_dir/'pbs_dast_harmonic_metrics.csv', rows)
    write_csv(args.output_dir/'pbs_dast_harmonic_per_channel.csv', channel_rows)
    table = render_table(rows).replace('PBS configurations and cancellation performance.',
                                     'PBS configurations and DAST with harmonic filtering performance.')
    table = table.replace(r'Config.~13 has $A_1=A_2=0$.',
                         r'Config.~13 has $A_1=A_2=0$; its harmonic stage is bypassed.'+'\n'+
                         f'Harmonic filtering uses up to {args.harmonics} harmonics below Nyquist '+
                         f'and a {1000*args.settling_time:g}~ms settling time.')
    (args.output_dir/'pbs_configurations.tex').write_text(table)
    write_json(args.output_dir/'protocol.json', dict(
        source_directory=str(args.source_dir.resolve()), source_protocol=protocol,
        harmonic_implementation='algorithms.harmonic_rls_3d',
        settling_time_s=args.settling_time, requested_harmonics=args.harmonics,
        configurations=settings,
        notes=['Existing harmonic implementation subtracts the per-channel input mean; it is not restored.',
               'Clean reference and scoring are unchanged; startup is included.',
               'Configuration 13 retains DAST, with harmonic filtering bypassed (no stimulation).',
               'Per-configuration harmonics are capped at Nyquist.']))
    (args.output_dir/'README.txt').write_text(
        'DAST followed by the existing harmonic RLS stage. See protocol.json for all settings.\n'
        'Reproduce: python tests/run_pbs_dast_table.py && python tests/add_pbs_harmonic_filter.py\n'
        'Previous DAST outputs are reused in full, preserving harmonic filter startup.\n'
        'The harmonic implementation removes the input mean. The clean reference is unchanged.\n'
        'Metrics are pooled across channels and samples; SNR is output SNR, not improvement.\n'
        'Config 13 has no harmonic stage because it has no stimulation.\n')
    print('Table:', args.output_dir/'pbs_configurations.tex')


if __name__ == '__main__':
    main()
