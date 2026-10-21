"""Verify cached harmonic postprocessing matches the full pipeline."""
import sys
from pathlib import Path
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from add_pbs_harmonic_filter import postfilter
from data_loading import Dataset
from pipeline import PipelineConfig, run_pipeline


def test_cached_postfilter_matches_pipeline():
    rng = np.random.default_rng(42)
    clean = rng.normal(size=(1, 3, 512)) + 2
    x = clean + 5*np.sin(2*np.pi*100*np.arange(512)/2048)[None, None, :]
    ds = Dataset(name='test', X=x, y_clean=clean, fs=2048., stim_rate=100.,
                 stim_channels=[0], metadata={})
    config = PipelineConfig(n_test=1, preprocessing='none', normalization='none',
                            use_derivative_for_tracking=True, derivative_order=1,
                            tracker='PASTd', rank=1, beta=.999, adaptive_k=False,
                            harmonic_filter=False, compute_ideal_u=False,
                            removal_method='projection')
    before = run_pipeline(ds, config).stages['cleaned']
    cached, count = postfilter(before, ds.fs, ds.stim_rate)
    expected = run_pipeline(ds, replace(config, harmonic_filter=True,
                                        harmonic_max_harmonics=count)).stages['cleaned']
    np.testing.assert_array_equal(cached, expected)


def test_control_bypass_and_nyquist_limit():
    x = np.random.default_rng(1).normal(size=(1, 2, 100))
    result, count = postfilter(x, 10240., 0.)
    assert count == 0
    np.testing.assert_array_equal(result, x)
    _, count = postfilter(x, 10240., 1000/1.5)
    assert count == 7
