"""Check metric definitions and completeness of the manuscript table."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_pbs_dast_table import reconstruction_metrics,render_table


def test_output_snr_and_pooled_mse():
    reference=np.array([[1.,2.],[3.,4.]])
    estimate=reference+1
    result=reconstruction_metrics(estimate,reference)
    assert result['mse_uv2']==1
    assert result['reference_power_uv2']==7.5
    assert result['snr_db']==pytest.approx(10*np.log10(7.5))


def test_exact_reconstruction_and_invalid_pairs():
    x=np.arange(5.)
    assert np.isposinf(reconstruction_metrics(x,x)['snr_db'])
    with pytest.raises(ValueError):
        reconstruction_metrics(x,x[:-1])
    with pytest.raises(ValueError):
        reconstruction_metrics(x,np.zeros(5))


def test_table_preserves_all_stimulated_configs_and_separate_control():
    rows=[dict(configuration=c,mse_uv2=c+.125,snr_db=10+c) for c in range(14)]
    table=render_table(rows)
    assert '13 & Config.~11 without stimulation & 13.12 & 23.00' in table
    assert table.count(r'\midrule')==2
    assert r'\label{tab:pbs_configurations}' in table
    assert r'$D_1$--$D_2$--$D_1$' in table
    with pytest.raises(ValueError):
        render_table(rows[:-1])
