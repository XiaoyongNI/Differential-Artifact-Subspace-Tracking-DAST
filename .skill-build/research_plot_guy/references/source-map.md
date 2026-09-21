# Source map and reproduction notes

Source inspected on 2026-09-20:
`C:/Users/xiaoy/Documents/learning/ETH_master/research/wireless_comm_AI/loeliger/code/matlab/Presentation`

This path is provenance and an optional local source, not a dependency for styling new figures. If reproducing an experiment on another machine, obtain its source/data first. No measured datasets are bundled in this skill.

## File roles

Each experiment folder has a `*_data.m` script with recorded values and arithmetic combinations, a `*_plot.m` script consuming workspace variables, `style.m` defining palette/styles/legend fragments, and `myLabel.m` setting axis labels. `rgb2hex.m` and `magnifyOnFigure.m` are utilities. Saved `.fig` and `.jpg` files are historical presentation artifacts and need not match current scripts.

The data scripts describe a three-dimensional Lorenz system, typically `m=n=3`, `T=30`, `dt=0.02`, and generating order `J=5`. They are result tables, not a dynamics simulator, filter implementation, or training pipeline.

## Active plot recipes

Indices below are MATLAB one-based. Use the named data script before its matching plot script, in the same folder.

| Folder; filename stem (`_data.m` / `_plot.m`) | Active series, in legend order | Limits x; y |
| --- | --- | --- |
| G1; `Lorenz_Full_Info_H_Lin` | EKF column 1, KalmanNet column 1, EKF column 2, KalmanNet column 2, noise reference; ratio labels 0 and -20 dB | [0,40]; [-51,0] |
| G2; `Lorenz_H_NL` | EKF columns 1,2,3 (labels 10,0,-10 dB), KalmanNet column 2 (0 dB); no noise reference | [-10,30]; automatic |
| G3; `Lorenz_noise` | EKF columns 1,2,3: matched noise, process mismatch -6 dB, process -6 dB plus observation +6 dB; KalmanNet vector; noise reference | [0,40]; [-42.5,2.5] |
| G4; `Lorenz_Jmdl` | EKF columns 1,3 (`Jmdl_arr=[5,3,2]'`, hence orders 5,2), KalmanNet column 3 (order 2), noise reference | [0,40]; [-50,0] |
| G5/EKF_Baseline; `Lorenz_H_rot` | EKF columns 1-5, rotation labels 0,0.1,1,5,10 degrees; noise reference | automatic; automatic |
| G5/KalmanNet; `Lorenz_H_rot_KNet` | EKF columns 1,2 (rotation 0,1), `MSE_KNet(:)` at rotation 1, noise reference labeled "Noise Level" | [0,40]; [-51,0] |

Most groups contain x samples [0,20,40,60], so x=60 is outside the limits in several scripts. G2 contains x=[-10,0,10,20,40,60], with its last two samples outside the view. G1 stores three ratio columns but plots only two. G3 stores four EKF columns but plots only three. Do not silently extend those selections in exact reproduction.

## Colors of the selected curves

Color names resolve to the hex palette in SKILL.md. All EKF markers use the selected style slot, which may differ from the data column index.

| Recipe | EKF colors / markers in plot order | KalmanNet colors / markers |
| --- | --- | --- |
| G1 | Gold /^, light blue /v | Green /o, blue /d (interleaved with EKF) |
| G2 | Green /^, gold /v, red /d | Blue /d |
| G3 | Green /^, light blue /v, gold /d | Blue /o |
| G4 | Green /^, gold /v | Blue /d |
| G5/EKF_Baseline | Green /^, light green /v, blue /d, light blue /h, gold /s | None |
| G5/KalmanNet | Green /^, gold /v | Blue /d |

`G3/style0.m` is an alternative legacy style (width 2, markers 10 and other color assignments), not the active `style.m`. Current G5/KalmanNet style italicizes KalmanNet; most other groups use upright mathematical text.

## Data and implementation caveats

- G1 `q_arr=[1,0.1,0.01]` has historical `q_arr_dB=10*log10(q_arr)`, but the displayed squared-ratio labels are 0,-20 dB for its first two columns. Its Python port uses `20*log10(q_arr)` for those labels. G2 explicitly labels its values 10,0,-10 dB. Do not impose the G1 rule on every group.
- G2 includes extrapolated EKF values for its first ratio, arithmetic means of recorded dB numbers, and a KalmanNet entry explicitly copied from an EKF entry. Other KalmanNet columns remain initialized to zero. Preserve these facts in reproduction; do not describe every value as an independent measured run.
- G4 initializes `MSE_KNET` from a zero array and fills only selected columns; column 2 is not a valid recorded comparison just because it exists.
- Averaging dB values differs from averaging linear MSE then converting to dB. Preserve source arithmetic for reproduction; for new research use the aggregation definition appropriate to the experiment and label it accurately.
- MATLAB variable spelling matters: most groups use `MSE_KNET`; G5/KalmanNet uses `MSE_KNet`.
- Legacy legends use `num2str` and sometimes character indexing. Generate new labels from scalar numeric values (`sprintf` or formatted Python strings), especially for multidigit or decimal parameters.
- G5/KalmanNet uses the loop's final `Jidx` for the learned curve's style. Make that style choice explicit in new code.
- The inspected G3 JPEG includes a title, inset, and broader limits than its current plot script. A saved image is visual evidence, not proof that the current script recreates every saved setting.

## Other plotting files

`G4/animation_T2000.m` reads `data/target_mean.txt`, reshapes 6000 floats to a 3x2000 trajectory, and extends a red `plot3` line with `pause(0.002)`, square axes, grid, and x(t)/y(t)/z(t) labels. Despite naming its array `EKF_mean`, the filename says target: inspect provenance before labeling it an estimate. It performs playback rather than simulation. For adaptation, verify the sample count, close the input file, and use MATLAB-compatible column-major reshape in NumPy (`order='F'`).

`G1/python/` contains Python equivalents of all six G1 MATLAB files plus tests and numeric reference artifacts. It is the preferred existing Python starting point for G1; G2-G5 have no equivalent Python ports in the inspected tree. Read its README for detailed magnifier behavior and tests.
