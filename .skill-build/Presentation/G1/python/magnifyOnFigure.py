"""Matplotlib adaptation of David Fernandez Prim's magnifyOnFigure (2009-2010).

Supports 2D lines/images, MATLAB property/value pairs or keyword properties,
rectangular/elliptical regions, draggable insets, and the original hotkeys.
Returns a Magnifier object; MATLAB graphics handles/structs are not accepted.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Ellipse, ConnectionPatch


class Magnifier:
    def __init__(self, ax, **options):
        defaults = dict(magnifiershape='rectangle', secondaryaxesfacecolor='white',
                        edgewidth=1, edgecolor='black', displaylinkstyle='straight',
                        mode='interactive', units='pixels',
                        initialpositionsecondaryaxes=None, initialpositionmagnifier=None,
                        secondaryaxesxlim=None, secondaryaxesylim=None,
                        frozenzoomaspectratio='off')
        supplied = {key.lower(): value for key, value in options.items()}
        unknown = supplied.keys() - defaults.keys()
        if unknown:
            raise TypeError('Unknown properties: ' + ', '.join(sorted(unknown)))
        defaults.update(supplied)
        self.options = o = defaults
        for key, allowed in [('magnifiershape', ('rectangle', 'ellipse')),
                             ('displaylinkstyle', ('straight', 'edges', 'none')),
                             ('mode', ('interactive', 'manual')),
                             ('units', ('pixels',)),
                             ('frozenzoomaspectratio', ('on', 'off'))]:
            o[key] = str(o[key]).lower()
            if o[key] not in allowed:
                raise ValueError(f'{key} must be one of {allowed}')
        if ax.name != 'rectilinear':
            raise ValueError('Magnifier supports 2D Cartesian axes only')
        if ax.collections or ax.patches:
            raise ValueError('This port supports line and image artists; use a line/image axes')
        for key, length in [('initialpositionsecondaryaxes', 4),
                            ('initialpositionmagnifier', 4),
                            ('secondaryaxesxlim', 2), ('secondaryaxesylim', 2)]:
            if o[key] is not None:
                value = np.asarray(o[key], dtype=float)
                if value.shape != (length,) or not np.all(np.isfinite(value)):
                    raise ValueError(f'{key} must contain {length} finite numbers')
                if (length == 4 and np.any(value[2:] <= 0)) or (length == 2 and value[0] == value[1]):
                    raise ValueError(f'{key} must have nonzero limits or positive dimensions')
                o[key] = value
        self.ax, self.fig = ax, ax.figure
        self.fig.canvas.draw()
        bbox = ax.bbox
        self.zoom = np.zeros(2)
        self.drag = None
        self.connections = []
        self.links = []
        self.removed = False
        self.frozen = o['frozenzoomaspectratio'] == 'on'
        self.tools = getattr(self.fig, '_g1_magnifiers', [])
        self.fig._g1_magnifiers = self.tools
        for tool in self.tools:
            tool.focus = False
        self.focus = True
        self.tools.append(self)
        p = o['initialpositionsecondaryaxes']
        if p is None:
            p = [bbox.x0 + 0.7*bbox.width - 10, bbox.y0 + 0.7*bbox.height - 10,
                 0.3*bbox.width, 0.3*bbox.height]
        self.inset = self.fig.add_axes(self._normalized(p), facecolor=o['secondaryaxesfacecolor'], label=f'_g1_{id(self)}')
        self.inset.set_xscale(ax.get_xscale())
        self.inset.set_yscale(ax.get_yscale())
        for line in ax.lines:
            self.inset.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(),
                            linestyle=line.get_linestyle(), linewidth=2,
                            marker=line.get_marker(), markersize=line.get_markersize(),
                            markerfacecolor=line.get_markerfacecolor(),
                            markeredgecolor=line.get_markeredgecolor(), alpha=line.get_alpha(),
                            drawstyle=line.get_drawstyle())
        for im in ax.images:
            self.inset.imshow(im.get_array(), cmap=im.get_cmap(), norm=im.norm,
                              origin=im.origin, extent=im.get_extent(),
                              interpolation=im.get_interpolation(), alpha=im.get_alpha(),
                              aspect='auto')
        self.inset.tick_params(colors=o['edgecolor'])
        for spine in self.inset.spines.values():
            spine.set_color(o['edgecolor'])
            spine.set_linewidth(o['edgewidth'])
        p = o['initialpositionmagnifier']
        if p is None:
            height = (0.3 if ax.lines else 0.1) * bbox.height
            p = [bbox.x0 + 0.45*bbox.width, bbox.y0 + (bbox.height-height)/2,
                 0.1*bbox.width, height]
        self.region = np.asarray(self._normalized(p))
        patch_cls = Rectangle if o['magnifiershape'] == 'rectangle' else Ellipse
        self.patch = patch_cls((0, 0), 1, 1, fill=False, edgecolor=o['edgecolor'],
                               linewidth=o['edgewidth'], transform=self.fig.transFigure)
        self.fig.add_artist(self.patch)
        self.identifier = self.inset.text(0.5, 0.5, str(len(self.tools)),
                                         transform=self.inset.transAxes, ha='center',
                                         va='center', color='white', fontsize=20, visible=False)
        self._update()
        if o['secondaryaxesxlim'] is not None:
            self.inset.set_xlim(o['secondaryaxesxlim'])
        if o['secondaryaxesylim'] is not None:
            self.inset.set_ylim(o['secondaryaxesylim'])
        self.connections.append(self.fig.canvas.mpl_connect('resize_event', self._update))
        if o['mode'] == 'interactive':
            for event, callback in [('button_press_event', self._press),
                                    ('button_release_event', self._release),
                                    ('motion_notify_event', self._motion),
                                    ('key_press_event', self._key)]:
                self.connections.append(self.fig.canvas.mpl_connect(event, callback))
        plt.sca(ax)

    def _normalized(self, position):
        return np.asarray(position, dtype=float) / np.tile(self.fig.bbox.size, 2)

    def _update(self, event=None):
        if self.removed:
            return
        x, y, w, h = self.region
        if isinstance(self.patch, Rectangle):
            self.patch.set_bounds(x, y, w, h)
        else:
            self.patch.center = (x+w/2, y+h/2)
            self.patch.width, self.patch.height = w, h
        corners = self.fig.transFigure.transform([[x, y], [x+w, y+h]])
        limits = self.ax.transData.inverted().transform(corners)
        center = limits.mean(axis=0)
        halfspan = (limits[1] - limits[0]) * (1-self.zoom) / 2
        self.inset.set_xlim(center[0]-halfspan[0], center[0]+halfspan[0])
        self.inset.set_ylim(center[1]-halfspan[1], center[1]+halfspan[1])
        for link in self.links:
            link.remove()
        self.links = []
        if self.options['displaylinkstyle'] != 'none':
            region_points = np.array([[x,y], [x+w,y], [x,y+h], [x+w,y+h]])
            b = self.inset.get_position()
            inset_points = np.array([[b.x0,b.y0], [b.x1,b.y0], [b.x0,b.y1], [b.x1,b.y1]])
            distances = np.sum(((region_points[:,None,:]-inset_points[None,:,:]) * self.fig.bbox.size)**2, axis=2)
            used_a, used_b = set(), set()
            count = 2 if self.options['displaylinkstyle'] == 'edges' else 1
            for index in np.argsort(distances, axis=None):
                i, j = np.unravel_index(index, distances.shape)
                if i in used_a or j in used_b:
                    continue
                link = ConnectionPatch(region_points[i], inset_points[j],
                                       coordsA=self.fig.transFigure, coordsB=self.fig.transFigure,
                                       color=self.options['edgecolor'], linewidth=self.options['edgewidth'])
                self.fig.add_artist(link)
                self.links.append(link)
                used_a.add(i)
                used_b.add(j)
                if len(self.links) == count:
                    break
        self.identifier.set_bbox(dict(facecolor='red' if self.focus else 'black'))
        self.fig.canvas.draw_idle()

    def _press(self, event):
        if event.button != 1 or event.x is None or event.y is None:
            return
        if getattr(self.fig.canvas, 'toolbar', None) and self.fig.canvas.toolbar.mode:
            return
        point = self.fig.transFigure.inverted().transform((event.x, event.y))
        x, y, w, h = self.region
        hit = x <= point[0] <= x+w and y <= point[1] <= y+h
        in_inset = event.inaxes is self.inset
        if not (hit or in_inset):
            return
        for tool in self.tools:
            tool.focus = tool is self
        self.drag = (point, 'inset' if in_inset else 'region',
                     np.array(self.inset.get_position().bounds) if in_inset else self.region.copy())

    def _motion(self, event):
        if self.drag is None or event.x is None or event.y is None:
            return
        point = self.fig.transFigure.inverted().transform((event.x, event.y))
        start, target, original = self.drag
        position = original.copy()
        position[:2] += point-start
        if target == 'inset':
            self.inset.set_position(position)
        else:
            self.region = position
        self._update()

    def _release(self, event):
        self.drag = None

    def _key(self, event):
        if not self.focus or getattr(event, '_g1_handled', False):
            return
        # One event must not be processed again after Tab changes focus.
        event._g1_handled = True
        key = (event.key or '').replace('control+', 'ctrl+')
        if key == 'tab':
            interactive = [t for t in self.tools if t.options['mode'] == 'interactive']
            self.focus = False
            interactive[(interactive.index(self)+1) % len(interactive)].focus = True
            for tool in interactive:
                tool._update()
            return
        if key == 'ctrl+d':
            self.remove()
            return
        if key == 'ctrl+i':
            self.identifier.set_visible(not self.identifier.get_visible())
        elif key == 'ctrl+a':
            print('Magnifier position:', self.region * np.tile(self.fig.bbox.size, 2))
            print('Secondary axes position:', np.array(self.inset.get_position().bounds) * np.tile(self.fig.bbox.size, 2))
        elif key == 'ctrl+q':
            self.zoom[:] = 0
        elif key.split('+')[-1] in ('pageup', 'pagedown'):
            component = 1 if 'shift+' in key else 0
            change = 0.1 if key.endswith('pageup') else -0.1
            indices = [0, 1] if self.frozen else [component]
            self.zoom[indices] = np.minimum(self.zoom[indices]+change, 0.9)
        elif key.split('+')[-1] in ('left', 'right', 'up', 'down'):
            direction = key.split('+')[-1]
            component = 0 if direction in ('left', 'right') else 1
            sign = -1 if direction in ('left', 'down') else 1
            is_inset = 'ctrl+' in key or 'alt+' in key
            position = np.array(self.inset.get_position().bounds) if is_inset else self.region.copy()
            if 'shift+' in key or 'alt+' in key:
                indices = [0, 1] if self.frozen else [component]
                for dim in indices:
                    old = position[dim+2]
                    position[dim+2] *= 1+sign*0.1
                    position[dim] += (old-position[dim+2])/2
            else:
                position[component] += sign/self.fig.bbox.size[component]
            if is_inset:
                self.inset.set_position(position)
            else:
                self.region = position
        self._update()

    def remove(self):
        """Delete this magnifier and disconnect its event handlers."""
        if self.removed:
            return
        for cid in self.connections:
            self.fig.canvas.mpl_disconnect(cid)
        for artist in self.links + [self.patch]:
            artist.remove()
        self.inset.remove()
        self.tools.remove(self)
        self.removed = True
        if self.tools and self.focus:
            self.tools[-1].focus = True
        self.fig.canvas.draw_idle()


def magnifyOnFigure(target=None, *args, **kwargs):
    """Create a magnifier from an Axes, Figure, or the current axes.

    Example: magnifyOnFigure(ax, 'mode', 'manual', secondaryAxesXLim=[10,20],
                            secondaryAxesYLim=[-30,-20]).
    Positions use [left, bottom, width, height] in figure pixels.
    """
    if len(args) % 2:
        raise TypeError('Positional properties must be name/value pairs')
    options = dict(zip(args[::2], args[1::2]))
    options.update(kwargs)
    if target is None:
        target = plt.gca()
    if isinstance(target, Figure):
        target = target.axes[0] if target.axes else target.add_subplot(111)
    if not isinstance(target, Axes):
        raise TypeError('Target must be a Matplotlib Axes or Figure')
    return Magnifier(target, **options)


magnify_on_figure = magnifyOnFigure

