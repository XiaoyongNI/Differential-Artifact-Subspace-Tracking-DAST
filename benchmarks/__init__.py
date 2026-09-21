"""Benchmarks compatible with the TBME dataset loaders."""
from .lrr import fit_lrr, OnlineLRR, apply_lrr_batch
from .methods import lrr, fit_lrr_trials
from .methods import (BenchmarkConfig, BenchmarkResult, METHODS, run_benchmark,
                      linear_interpolation, template_subtraction, window_svd, window_ica, pulse)

__all__ = ["fit_lrr", "OnlineLRR", "apply_lrr_batch", "lrr", "fit_lrr_trials",
           "BenchmarkConfig", "BenchmarkResult", "METHODS", "run_benchmark",
           "linear_interpolation", "template_subtraction", "window_svd", "window_ica", "pulse"]
