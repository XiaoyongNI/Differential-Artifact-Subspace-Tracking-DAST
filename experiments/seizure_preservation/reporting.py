"""Patient-level metrics, paired inference, and publication plots."""
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
from plotting import set_tbme_style
from .common import write_csv

METRICS = ('accuracy', 'weighted_f1', 'kappa')
RESTORATION = ('restoration_accuracy', 'restoration_f1', 'restoration_kappa')


def compute_restoration_ratio(performance, clean, artifact, epsilon=1e-6):
    """Unclipped ratio; a negative clean-artifact gap retains its sign."""
    if epsilon <= 0:
        raise ValueError('epsilon must be positive')
    gap = clean-artifact
    if not np.isfinite([performance, clean, artifact]).all() or abs(gap) <= epsilon:
        return float('nan')
    return float((performance-artifact)/gap)


def add_restoration(rows, epsilon):
    for decoder in sorted({r['decoder'] for r in rows}):
        decoder_rows = [r for r in rows if r['decoder']==decoder]
        reference = {r['method']:r for r in decoder_rows}
        for row in decoder_rows:
            for metric, restored in zip(METRICS, RESTORATION):
                row[restored] = compute_restoration_ratio(row[metric], reference['Clean'][metric],
                                                          reference['Contaminated'][metric], epsilon)
    return rows


def aggregate_results(rows):
    summaries = []
    for decoder, method in sorted({(r['decoder'],r['method']) for r in rows}):
        subset = [r for r in rows if (r['decoder'],r['method'])==(decoder,method)]
        if len({r['patient'] for r in subset}) != len(subset):
            raise ValueError('Duplicate patient/decoder/method observations')
        summary = dict(decoder=decoder, method=method, n_patients=len(subset))
        for metric in (*METRICS, *RESTORATION):
            values = np.array([r[metric] for r in subset], dtype=float)
            values = values[np.isfinite(values)]
            mean = float(values.mean()) if len(values) else float('nan')
            sd = float(values.std(ddof=1)) if len(values)>1 else float('nan')
            summary.update({metric+'_mean':mean, metric+'_sd':sd, metric+'_n':len(values),
                            metric+'_mean_sd':f'{mean:.6g} ± {sd:.6g}'})
        summaries.append(summary)
    return summaries


def _holm(pvalues):
    pvalues = np.array(pvalues, dtype=float)
    result = np.full(len(pvalues), np.nan)
    indices = np.flatnonzero(np.isfinite(pvalues))
    indices = indices[np.argsort(pvalues[indices])]
    if len(indices):
        result[indices] = np.minimum(1, np.maximum.accumulate(pvalues[indices]*(len(indices)-np.arange(len(indices)))))
    return result


def run_statistical_analysis(rows, config, output_dir):
    """Two-sided paired Wilcoxon on patients, global Holm correction.

    One family spans all decoders x comparators x three raw metrics. Missing
    pairs are excluded pairwise and explicitly counted; all-zero differences
    give p=1 rather than a SciPy exception. No window-level inference.
    """
    stats = []
    comparators = [m for m in config['methods'] if m not in ('Clean','DAST')]
    for decoder in config['decoders']:
        indexed = {(r['patient'],r['method']):r for r in rows if r['decoder']==decoder}
        if len(indexed) != sum(r['decoder']==decoder for r in rows):
            raise ValueError('Duplicate observations in paired analysis')
        patients = sorted({p for p,m in indexed if m=='DAST'})
        for baseline in comparators:
            for metric in METRICS:
                pairs = [(p, indexed[p,'DAST'][metric], indexed[p,baseline][metric])
                         for p in patients if (p,baseline) in indexed]
                pairs = [(p,a,b) for p,a,b in pairs if np.isfinite([a,b]).all()]
                delta = np.round([a-b for p,a,b in pairs], config['statistics']['difference_round_decimals'])
                if len(delta)<2:
                    statistic, pvalue = np.nan, np.nan
                elif np.all(delta==0):
                    statistic, pvalue = 0.0, 1.0
                else:
                    result = wilcoxon(delta, alternative='two-sided', zero_method='wilcox', method='auto')
                    statistic, pvalue = float(result.statistic), float(result.pvalue)
                stats.append(dict(decoder=decoder, baseline=baseline, metric=metric, n_pairs=len(pairs),
                    n_nonzero_pairs=int(np.count_nonzero(delta)), patients=';'.join(p for p,a,b in pairs),
                    mean_dast_minus_baseline=float(np.mean(delta)) if len(delta) else np.nan,
                    statistic=statistic, p_raw=pvalue))
    adjusted = _holm([r['p_raw'] for r in stats])
    for row,p in zip(stats, adjusted):
        row.update(p_corrected=float(p), correction='holm',
                   family=config['statistics']['family'], reject=bool(p<config['statistics']['alpha']))
    write_csv(Path(output_dir)/'paired_statistics.csv', stats)
    return stats


def plot_results(rows, config, output_dir):
    import matplotlib.pyplot as plt
    set_tbme_style()
    directory = Path(output_dir)/'figures'
    directory.mkdir(parents=True, exist_ok=True)
    labels = dict(accuracy='Accuracy', weighted_f1='Weighted F1', kappa="Cohen’s κ",
                  restoration_accuracy='Accuracy restoration', restoration_f1='Weighted F1 restoration',
                  restoration_kappa='κ restoration')
    decoders = config['decoders']
    def panel(ax, decoder, metric, methods):
        patients = sorted({r['patient'] for r in rows if r['decoder']==decoder})
        lookup = {(r['patient'],r['method']):r[metric] for r in rows if r['decoder']==decoder}
        values = np.array([[lookup.get((p,m),np.nan) for m in methods] for p in patients])
        positions = np.arange(len(methods))
        for i,line in enumerate(values):
            # Fixed deterministic horizontal offset preserves patient pairing.
            offset = (i-(len(patients)-1)/2)*0.012
            ax.plot(positions+offset, line, '-o', color='0.4', alpha=config['plots']['patient_alpha'],
                    markersize=3, linewidth=.55, zorder=1)
        means, sd = [], []
        for column in values.T:
            finite = column[np.isfinite(column)]
            means.append(finite.mean() if len(finite) else np.nan)
            sd.append(finite.std(ddof=1) if len(finite)>1 else 0)
        ax.errorbar(positions, means, yerr=sd, fmt='D', color='#1f77b4', capsize=3,
                    markersize=5, linewidth=1.4, label='Patient mean ± SD', zorder=3)
        ax.set_xticks(positions, [m.replace('_','\n') for m in methods], rotation=40, ha='right')
        ax.set_title(f'{decoder} (n={len(patients)} patients)')
        ax.set_ylabel(labels[metric])
        if metric.startswith('restoration'):
            ax.axhline(0, color='0.5', linestyle=':', linewidth=.8)
            ax.axhline(1, color='0.3', linestyle='--', linewidth=.8)
        else:
            ax.set_ylim((-1.03,1.03) if metric=='kappa' else (-.03,1.03))
    for metric in METRICS:
        fig, axes = plt.subplots(1,len(decoders), squeeze=False,
            figsize=(config['plots']['width_per_decoder']*len(decoders), config['plots']['height']), constrained_layout=True)
        for ax,decoder in zip(axes[0],decoders):
            panel(ax, decoder, metric, config['methods'])
        axes[0,0].legend(fontsize=8)
        for ext in ('pdf','png'):
            fig.savefig(directory/f'{metric}.{ext}', dpi=config['plots']['dpi'], bbox_inches='tight')
        plt.close(fig)
    methods = [m for m in config['methods'] if m not in ('Clean','Contaminated')]
    fig, axes = plt.subplots(3,len(decoders), squeeze=False,
        figsize=(config['plots']['width_per_decoder']*len(decoders),3*config['plots']['height']), constrained_layout=True)
    for i,metric in enumerate(RESTORATION):
        for ax,decoder in zip(axes[i],decoders):
            panel(ax,decoder,metric,methods)
    for ext in ('pdf','png'):
        fig.savefig(directory/f'restoration.{ext}', dpi=config['plots']['dpi'], bbox_inches='tight')
    plt.close(fig)
