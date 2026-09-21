---
name: research_plot_guy
description: Create, restyle, or reproduce MATLAB and Python scientific comparison plots using the Loeliger Presentation folder conventions, including EKF/KalmanNet MSE curves, parameter sweeps, noise references, and optional zoom insets. Use for research figures in this style or explicit research_plot_guy requests.
---

# Research Plot Guy

Turn experiment results into clear, reproducible scientific figures using the plotting conventions distilled from the Loeliger Presentation collection. Support MATLAB and Python/Matplotlib; follow the user's requested language and output format.

## Choose the task

- For a new figure, reuse the visual grammar below with the user's actual data, methods, variables, and units. The Lorenz experiment settings are examples, not defaults for unrelated research.
- For reproducing G1-G5, read [the source map](references/source-map.md) for data selection, curve order, palette differences, limits, and source caveats.
- For execution, language conversion, export, or magnifiers, read [implementation guidance](references/implementation.md).

## Visual grammar

Use a white background, visible grid, strong line and marker contrast, mathematical axis labels, explicit units, and a concise legend describing both method and varied condition. Keep the same condition-to-style mapping across related figures.

| Element | Presentation default |
| --- | --- |
| Baseline / EKF | Dashed line; markers `^`, `v`, `d`, `h`, `s`, `o` |
| Learned method / KalmanNet | Solid line; markers `o`, `d`, `d`, `h`, `s`, `o` |
| Reference noise curve | Red dash-dot line, no marker |
| Main palette | Green `#77AC30`, blue `#0072BD`, gold `#F9A602`, light blue `#5BCFF4` |
| Additional palette | Light green `#95F985`, purple `#67025E`, near-black `#111111` |
| Lines / markers | Width 4, marker size 15, black marker edges, unfilled markers |
| Text | Tick labels 30 pt, legend 35 pt, axis labels 36 pt |

These sizes fit large presentation canvases; scale them coherently for paper columns or smaller panels. G5/EKF_Baseline uses width 2, markers 10, ticks 24, and legend 20. Do not assume one universal palette ordering: use the source map for reproduction. Avoid indexing beyond unequal legacy style arrays; define an explicit style per requested series.

The active comparison scripts leave titles disabled in `myLabel.m`; include a title or experimental metadata only when useful or requested. Historical saved images can have titles, insets, and different limits absent from current scripts. Establish which artifact is the reproduction target.

## Preserve scientific meaning

Keep data loading, series selection, style configuration, and export separate. Use explicit figure/axes handles and a callable plotting function when creating new code. Build legend entries from plotted series and actual parameter values, in plotting order.

The source comparison figures use `x = -r2_arr_dB`, where `r2_arr_dB = 10*log10(r2_arr)`, with x labeled inverse observation-noise variance in dB and y labeled `MSE [dB]`. Stored MSE values are already in dB. Plot them directly on linear axes; do not apply another log transform. The red reference is `y = r2_arr_dB = -x`; add it only when meaningful for the data. Its historical label "Noise Floor" does not establish a universal estimator lower bound.

For new data, distinguish power/variance ratios (`10*log10`) from amplitude/standard-deviation ratios (`20*log10`). The legacy `q_arr` conventions differ between examples, so verify metadata rather than inferring units from its name. Preserve historical numeric calculations when reproducing, and disclose uncertainties or approximations rather than silently repairing them.

Retain requested series and limits for reproduction, noting hidden data. For new figures, choose limits from the selected data so results are not silently clipped. Never treat zero-initialized, unfilled legacy columns as measured results or invent missing measurements.

## Finish

Check x/y lengths, selected columns, units, legend correspondence, and visible range. Render and inspect the exported figure for clipped labels, overlapping legends, indistinguishable styles, and inset placement. Deliver reusable plotting code plus the requested figure outputs; state relevant data omissions or execution limitations. Do not claim simulation, training, or experimental validation from these plotting files.
