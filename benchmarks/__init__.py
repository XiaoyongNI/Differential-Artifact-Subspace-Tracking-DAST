"""Benchmarks compatible with the TBME dataset loaders."""
from .methods import (BenchmarkConfig, BenchmarkResult, METHODS, run_benchmark,
                      linear_interpolation, template_subtraction, window_svd, window_ica, pulse)

__all__ = ["BenchmarkConfig", "BenchmarkResult", "METHODS", "run_benchmark",
           "linear_interpolation", "template_subtraction", "window_svd", "window_ica", "pulse"]
