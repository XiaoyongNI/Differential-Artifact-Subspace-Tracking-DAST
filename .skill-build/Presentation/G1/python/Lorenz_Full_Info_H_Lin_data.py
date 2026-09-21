"""Recorded G1 results, translated from Lorenz_Full_Info_H_Lin_data.m.

Rows index observation variances; columns index process/observation
standard-deviation ratios. MSE values are already in dB.
"""
import numpy as np

m = n = 3
T = 30
dt = 0.02
J = 5
r2_arr = np.array([1e0, 1e-2, 1e-4, 1e-6])
r2_arr_dB = 10 * np.log10(r2_arr)
Nr = len(r2_arr)
q_arr = np.array([1e0, 1e-1, 1e-2])
q_arr_dB = 10 * np.log10(q_arr)  # Original variable; legend uses 20*log10(q_arr).
Nq = len(q_arr)
MSE = np.array([
    [(-2.139702406575807 - 2.174910340581381) / 2,
     (-10.12700061958431 - 10.041765463691107 - 10.046306035119015) / 3,
     -18.655000054478997],
    [(-22.171000440546468 - 22.179474683248245) / 2,
     (-30.047837611679185 - 30.048575443957823 - 30.06414986057351) / 3,
     -38.71761413752751],
    [(-42.15506489346719 - 42.160531806450976) / 2,
     (-50.129902295386906 - 50.12249452724808 - 50.05043492427429) / 3,
     -58.675050532717414],
    [(-62.184757704142385 - 62.19432778198879) / 2,
     (-70.0706231763961 - 70.09860178279828 - 70.05804642126097) / 3,
     -78.64752565319313],
])
MSE_KNET = np.array([
    [-2.0543682716637126, -10.00331639589777, -17.07254696350233],
    [-22.040053245709046, -29.89401070972432, -36.96423234005554],
    [-41.93072918077107, -49.14118426626317, -56.145213769428906],
    [-62.07994395416154, -68.63837128341339, -75.69891740764328],
])


def load_data():
    """Return independent copies so callers cannot mutate the saved results."""
    names = ('m', 'n', 'T', 'dt', 'J', 'r2_arr', 'r2_arr_dB', 'Nr',
             'q_arr', 'q_arr_dB', 'Nq', 'MSE', 'MSE_KNET')
    return {name: globals()[name].copy() if isinstance(globals()[name], np.ndarray)
            else globals()[name] for name in names}


if __name__ == '__main__':
    print('EKF MSE [dB]:\n', MSE)
    print('KalmanNet MSE [dB]:\n', MSE_KNET)
