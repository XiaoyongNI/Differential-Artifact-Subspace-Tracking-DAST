# G1: Python conversion

All six MATLAB source files have Python counterparts in this folder. The
original MATLAB files and saved figures are unchanged. The actual source path
uses `ETH_master/research/wireless_comm_AI`, rather than the split directory
names in the original request.

## Run

From this `python` folder:

```powershell
python -m pip install -r requirements.txt
python Lorenz_Full_Info_H_Lin_plot.py
```

Your existing Anaconda Python already has the required libraries:

```powershell
& 'C:\Users\xiaoy\anaconda3\python.exe' .\Lorenz_Full_Info_H_Lin_plot.py
```

Save a figure without opening a window:

```powershell
python Lorenz_Full_Info_H_Lin_plot.py --no-show --save G1_results.png
python Lorenz_Full_Info_H_Lin_plot.py --no-show --save G1_results.pdf
```

The plotting script loads its own data; there is no need to execute the data
script first. Paths supplied to `--save` are relative to your working directory.
Imports also work as a package (`python.Lorenz_Full_Info_H_Lin_plot`) from G1.

## Files and behavior

| MATLAB file | Python counterpart | Purpose |
| --- | --- | --- |
| `Lorenz_Full_Info_H_Lin_data.m` | `Lorenz_Full_Info_H_Lin_data.py` | Model metadata and all 24 recorded MSE values |
| `Lorenz_Full_Info_H_Lin_plot.m` | `Lorenz_Full_Info_H_Lin_plot.py` | EKF/KalmanNet comparison and noise floor |
| `style.m` | `style.py` | Original palette, line styles, markers, math labels |
| `myLabel.m` | `myLabel.py` | Axis labeling function |
| `rgb2hex.m` | `rgb2hex.py` | RGB conversion with MATLAB-compatible rounding |
| `magnifyOnFigure.m` | `magnifyOnFigure.py` | Matplotlib magnifier adaptation |

This directory contains recorded experiment results, not the Lorenz dynamics,
EKF implementation, or KalmanNet training. No simulation is performed by these
scripts. MSE arrays are `(4, 3)` NumPy arrays, matching MATLAB's effective shape
(the third dimension in the assignments is singleton). Python indexing is
zero-based. `load_data()` returns independent array copies.

The default figure retains the first two ratio columns (0 and -20 dB), all four
noise levels, five plotted curves, original colors/markers/font sizes, and
limits x=[0,40], y=[-51,0]. Thus points at x=60 and the third ratio column are
still omitted from the displayed view, just as in MATLAB. A larger canvas keeps
the presentation-sized legend readable. Math uses Matplotlib mathtext; a LaTeX
installation is unnecessary. The original `q_arr_dB = 10*log10(q_arr)` variable
is retained, while the squared-ratio legend uses `20*log10(q_arr)`.

## Magnifier

```python
import matplotlib.pyplot as plt
from Lorenz_Full_Info_H_Lin_plot import plot_results
from magnifyOnFigure import magnifyOnFigure

fig, ax = plot_results()
tool = magnifyOnFigure(
    ax,
    secondaryAxesXLim=[15, 25],
    secondaryAxesYLim=[-30, -20],
    initialPositionSecondaryAxes=[250, 100, 400, 250],
    displayLinkStyle='edges',
)
plt.show()
```

Accepts an Axes, Figure, or no target (current axes), and either MATLAB-style
property/value pairs or keyword arguments. Positions are `[left, bottom, width,
height]` in figure pixels. Supported properties are `magnifierShape`,
`secondaryAxesFaceColor`, `edgeWidth`, `edgeColor`, `displayLinkStyle`, `mode`,
`units`, `initialPositionSecondaryAxes`, `initialPositionMagnifier`,
`secondaryAxesXLim`, `secondaryAxesYLim`, and `frozenZoomAspectRatio`.

Drag the region or inset; click either to focus it. Tab switches focus.
Arrow keys move the region one pixel; Shift+arrows resize it by 10%.
Ctrl+arrows move the inset; Alt+arrows resize it. PageUp/PageDown change
additional horizontal zoom; Shift applies to vertical zoom. Ctrl+Q resets
additional zoom; Ctrl+I toggles the identifier; Ctrl+A prints positions;
Ctrl+D deletes the focused magnifier. `tool.remove()` also cleans it up.
Interactive controls require a GUI Matplotlib backend and an inactive pan/zoom
toolbar. Backend/window-manager shortcuts may intercept some keys.

This helper is a Matplotlib adaptation, not a MATLAB graphics compatibility
layer: it supports 2D Cartesian line and image axes (including reversed axes),
but rejects scatter/collection/patch and 3D axes. It does not accept saved MATLAB
handle/struct state. Connecting lines use nearest box corners; they do not
reproduce every original edge/ellipse intersection rule. Artists are copied at
creation time. `.fig` files are not converted. The main G1 plot does not call
this optional helper.

Source attribution: `magnifyOnFigure.m` by David Fernandez Prim (2009-2010),
`rgb2hex.m` by Chad A. Greene (2014). Python adaptations retain these credits.

## Validation

```powershell
python -m unittest -v
```

The unmodified `Lorenz_Full_Info_H_Lin_data.m` was executed with MATLAB R2023a.
Its arrays were saved in `matlab_reference.mat`; the same values are provided
in `matlab_reference.json` so regression tests need only NumPy and Matplotlib.
The tests compare all recorded MSE values, noise arrays, and model metadata
against this reference at absolute tolerance 1e-13. They also check plotted
curves/limits, RGB rounding and validation, magnifier movement/resize/zoom,
multiple independent insets, focus/deletion, and reversed image axes.

Tested using Python 3.8.8, NumPy 1.23.5, Matplotlib 3.3.4. Magnifier events were
tested programmatically with the noninteractive Agg backend; desktop keyboard
and mouse behavior has not been manually exercised.
