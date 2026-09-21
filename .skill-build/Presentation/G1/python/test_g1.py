"""Numerical and rendering regression checks. Run: python -m unittest -v."""
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent, MouseEvent
from Lorenz_Full_Info_H_Lin_data import load_data
from Lorenz_Full_Info_H_Lin_plot import plot_results
from rgb2hex import rgb2hex
from magnifyOnFigure import magnifyOnFigure


class G1Tests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_matlab_reference(self):
        # Exported by executing the unmodified original script in MATLAB R2023a.
        reference = json.loads(Path(__file__).with_name('matlab_reference.json').read_text(encoding='utf-8'))
        actual = load_data()
        for key in ('MSE', 'MSE_KNET', 'r2_arr', 'r2_arr_dB', 'q_arr', 'q_arr_dB', 'm', 'n', 'T', 'dt', 'J'):
            with self.subTest(key=key):
                np.testing.assert_allclose(actual[key], reference[key], rtol=0, atol=1e-13)

    def test_independent_data(self):
        data = load_data()
        data['MSE'][0, 0] = 999
        self.assertLess(load_data()['MSE'][0, 0], 0)
        self.assertEqual(data['MSE'].shape, (4, 3))

    def test_rgb_matlab_rounding_and_scale(self):
        self.assertEqual(rgb2hex([0, 1, 0]), '#00FF00')
        self.assertEqual(rgb2hex([0, 255, 0]), '#00FF00')
        self.assertEqual(rgb2hex([[.2,.3,.4], [.5,.6,.7], [.8,.6,.2], [.2,.2,.9]]),
                         ['#334D66', '#8099B3', '#CC9933', '#3333E6'])
        self.assertEqual(rgb2hex([[0,1,0], [255,0,0]]), ['#000100', '#FF0000'])

    def test_rgb_invalid(self):
        for value in ([], [1,2], [-1,0,0], [256,0,0], [np.nan,0,0], [np.inf,0,0]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                rgb2hex(value)
        with self.assertRaises(TypeError):
            rgb2hex(['r','g','b'])

    def test_plot_curves_and_limits(self):
        fig, ax = plot_results()
        fig.canvas.draw()
        self.assertEqual(len(ax.lines), 5)
        data = load_data()
        for i, key in enumerate(('MSE', 'MSE_KNET', 'MSE', 'MSE_KNET')):
            np.testing.assert_array_equal(ax.lines[i].get_xdata(), [0,20,40,60])
            np.testing.assert_array_equal(ax.lines[i].get_ydata(), data[key][:, i//2])
        np.testing.assert_array_equal(ax.lines[4].get_ydata(), [0,-20,-40,-60])
        self.assertEqual(ax.get_xlim(), (0,40))
        self.assertEqual(ax.get_ylim(), (-51,0))
        self.assertEqual(len(ax.get_legend().get_texts()), 5)

    def make_axes(self):
        fig, ax = plt.subplots()
        ax.plot([0,1,2], [2,1,0])
        ax.set(xlim=(0,2), ylim=(0,2))
        return fig, ax

    def test_manual_magnifier(self):
        fig, ax = self.make_axes()
        tool = magnifyOnFigure(ax, 'mode', 'manual', secondaryAxesXLim=[.2,.5],
                               secondaryAxesYLim=[.4,.9], displayLinkStyle='edges')
        fig.canvas.draw()
        self.assertEqual(len(tool.inset.lines), 1)
        self.assertEqual(len(tool.links), 2)
        np.testing.assert_allclose(tool.inset.get_xlim(), [.2,.5])
        np.testing.assert_allclose(tool.inset.get_ylim(), [.4,.9])
        tool.remove()
        tool.remove()
        self.assertEqual(len(fig.axes), 1)
        self.assertEqual(len(fig._g1_magnifiers), 0)

    def test_magnifier_keys_focus_and_delete(self):
        fig, ax = self.make_axes()
        first = magnifyOnFigure(ax)
        second = magnifyOnFigure(ax)
        self.assertIsNot(first.inset, second.inset)
        self.assertEqual(len(fig.axes), 3)
        before = second.region.copy()
        fig.canvas.callbacks.process('key_press_event', KeyEvent('key_press_event', fig.canvas, 'right'))
        self.assertAlmostEqual(second.region[0]-before[0], 1/fig.bbox.width)
        fig.canvas.callbacks.process('key_press_event', KeyEvent('key_press_event', fig.canvas, 'tab'))
        self.assertTrue(first.focus)
        self.assertFalse(second.focus)
        first._key(KeyEvent('key_press_event', fig.canvas, 'pageup'))
        self.assertAlmostEqual(first.zoom[0], .1)
        first._key(KeyEvent('key_press_event', fig.canvas, 'ctrl+q'))
        np.testing.assert_array_equal(first.zoom, [0,0])
        first._key(KeyEvent('key_press_event', fig.canvas, 'ctrl+d'))
        self.assertTrue(first.removed)
        self.assertTrue(second.focus)

    def test_magnifier_drag_and_resize(self):
        fig, ax = self.make_axes()
        tool = magnifyOnFigure(ax)
        x,y,w,h = tool.region
        start = fig.transFigure.transform([x+w/2,y+h/2])
        tool._press(MouseEvent('button_press_event', fig.canvas, *start, button=1))
        self.assertIsNotNone(tool.drag)
        before = tool.region.copy()
        tool._motion(MouseEvent('motion_notify_event', fig.canvas, *(start+[10,5]), button=1))
        np.testing.assert_allclose(tool.region[:2]-before[:2], [10/fig.bbox.width,5/fig.bbox.height])
        tool._release(None)
        self.assertIsNone(tool.drag)
        before = tool.region.copy()
        fig.set_size_inches(10,8)
        tool._update()
        np.testing.assert_array_equal(tool.region, before)

    def test_image_ellipse_reversed_axes(self):
        fig, ax = plt.subplots()
        ax.imshow(np.arange(100).reshape(10,10))
        tool = magnifyOnFigure(fig, magnifierShape='ellipse', displayLinkStyle='none',
                               frozenZoomAspectRatio='on')
        fig.canvas.draw()
        self.assertEqual(len(tool.inset.images), 1)
        self.assertTrue(tool.inset.yaxis_inverted())
        self.assertEqual(len(tool.links), 0)
        before = tool.region[2:].copy()
        tool._key(KeyEvent('key_press_event', fig.canvas, 'shift+right'))
        np.testing.assert_allclose(tool.region[2:], before*1.1)

    def test_invalid_options(self):
        fig, ax = self.make_axes()
        for options in ({'units':'inches'}, {'mode':'bad'},
                        {'initialPositionMagnifier':[0,0,-1,4]}):
            with self.assertRaises(ValueError):
                magnifyOnFigure(ax, **options)
        with self.assertRaises(TypeError):
            magnifyOnFigure(ax, unknown=True)


if __name__ == '__main__':
    unittest.main()
