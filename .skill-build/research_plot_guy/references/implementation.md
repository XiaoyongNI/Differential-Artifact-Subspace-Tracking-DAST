# Execution, conversion, and export

## MATLAB

Legacy data files begin with `clear all`, `close all`, and `clc`; plot files also close figures. Run exact reproduction in a separate MATLAB process/session when preserving an interactive workspace matters. Each group has same-named helpers with different contents: run from the selected experiment directory rather than adding all groups recursively to the path.

Example from G1's directory:

```matlab
Lorenz_Full_Info_H_Lin_data
Lorenz_Full_Info_H_Lin_plot
```

For reusable new code, accept data and optional axes as function arguments, return graphics handles, use `grid(ax,'on')` rather than toggling grid, and avoid global workspace resets. Set `MarkerFaceColor` to `none` explicitly. Use MATLAB's LaTeX interpreter for mathematical labels. Bind the legend to explicit line handles so later inset/reference artists cannot change its meaning.

Export a vector PDF for line-art publication and a PNG for quick inspection when appropriate; preserve `.fig` if MATLAB editability is requested. Use facilities supported by the installed MATLAB version and inspect the actual export, especially with insets. The current scripts mainly create figures interactively and do not supply a consistent export pipeline.

## Python / Matplotlib

The existing G1 port loads its own data, creates a 24x14 inch canvas, uses mathtext (no external LaTeX installation), returns `(figure, axes)` from `plot_results(data=None, ax=None)`, and accepts:

```powershell
python Lorenz_Full_Info_H_Lin_plot.py --no-show --save G1_results.pdf
python Lorenz_Full_Info_H_Lin_plot.py --no-show --save G1_results.png
```

These commands run from `G1/python`; select an available Python interpreter with NumPy and Matplotlib. Its PNG export uses 160 dpi and a tight bounding box. For new figures choose resolution/canvas to fit the requested output rather than requiring those settings universally.

When adapting:

- Convert one-based selection indices to zero-based explicitly.
- Preserve rows as noise samples and columns as conditions; remove only known singleton dimensions. Flatten only actual vector series.
- Use `markerfacecolor='none'`, `markeredgecolor='k'`, explicit colors/styles, and one label per plotted line.
- Translate MATLAB `$$...$$` / `\textrm` label syntax into Matplotlib-supported single-dollar mathtext and ordinary unit text.
- Use a noninteractive backend for headless rendering and save before closing the figure.
- Preserve source numbers and averaging expressions. A port should not perform additional simulation or infer missing results.

The existing G1 tests can be run with `python -m unittest -v` from that directory when modifying the port. They include numerical references and curve checks. For new changes, validate the affected data/plot behavior and inspect the exported image; do not claim the existing tests verify newly ported groups.

## Magnifiers

`magnifyOnFigure.m` is an optional interactive zoom-inset helper by David Fernandez Prim (2009-2010). `rgb2hex.m` is by Chad A. Greene (2014). Preserve attribution if copying or adapting either utility. RGB conversion accepts Nx3 arrays on a 0-1 or 0-255 scale; its scale decision is made across the input, and its rounding should be preserved if numerical compatibility matters.

Only add an inset where it reveals a meaningful separation. Set its region and position deliberately and verify exported connectors/labels. The active comparison scripts do not call the magnifier automatically.

The MATLAB helper accepts a figure or axes handle and property/value pairs. Relevant options include `secondaryAxesXLim`, `secondaryAxesYLim`, `initialPositionSecondaryAxes`, `initialPositionMagnifier`, `displayLinkStyle` (`none`, `straight`, `edges`), `magnifierShape` (`rectangle`, `ellipse`), and `mode` (`manual`, `interactive`). Positions use [left,bottom,width,height] in pixels. It has known limitations around pan/zoom updates, large datasets, and printed positioning.

G1's Python adaptation accepts corresponding keywords but is not a general MATLAB graphics compatibility layer. It supports 2D Cartesian line/image axes, not 3D or scatter/collection/patch axes, and does not load MATLAB `.fig` or saved handle state. Artists are copied at creation; GUI interaction needs an interactive backend. For a new static figure, a native inset may be simpler than copying the interactive helper.
