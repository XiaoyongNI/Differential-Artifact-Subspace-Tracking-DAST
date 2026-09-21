"""Benchmarks compatible with the TBME dataset loaders."""
from .lrr import fit_lrr, OnlineLRR, apply_lrr_batch
from .asar import OnlineASAR, calibrate_asar
from .methods import asar
from .dictionary_learning import ArtifactDictionary, fit_dictionary
from .methods import dictionary_learning
from .methods import lrr, fit_lrr_trials
from .methods import (BenchmarkConfig, BenchmarkResult, METHODS, run_benchmark,
                      linear_interpolation, average_template_subtraction, svd_template_subtraction,
                      window_svd, window_ica, pulse, low_rank_tv)

__all__ = ["ArtifactDictionary", "fit_dictionary", "dictionary_learning",
           "OnlineASAR", "calibrate_asar", "asar", "fit_lrr", "OnlineLRR", "apply_lrr_batch", "lrr", "fit_lrr_trials",
           "BenchmarkConfig", "BenchmarkResult", "METHODS", "run_benchmark",
           "linear_interpolation", "average_template_subtraction", "svd_template_subtraction",
           "window_svd", "window_ica", "pulse", "low_rank_tv"]
