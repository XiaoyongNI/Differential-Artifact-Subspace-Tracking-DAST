"""Public modular API for the seizure-information preservation experiment."""
from .data import load_patient_data, split_patient_data, extract_windows
from .cancellation import generate_synthetic_artifact, apply_artifact_cancellation
from .training import train_eegnet, train_gru, train_chrononet, evaluate_decoder
from .reporting import compute_restoration_ratio, run_statistical_analysis, plot_results

__all__ = [
    'load_patient_data', 'split_patient_data', 'extract_windows',
    'generate_synthetic_artifact', 'train_eegnet', 'train_gru', 'train_chrononet',
    'apply_artifact_cancellation', 'evaluate_decoder', 'compute_restoration_ratio',
    'run_statistical_analysis', 'plot_results',
]
