"""G1 colors, line styles and Matplotlib-compatible math labels."""
import numpy as np
try:
    from .rgb2hex import rgb2hex
except ImportError:
    from rgb2hex import rgb2hex

_PALETTE = {
    'yellow': (253,230,75), 'canary': (249,200,2), 'gold': (249,166,2),
    'daffodil': (253,238,135), 'flaxen': (214,183,90), 'butter': (254,226,39),
    'lemon': (239,253,95), 'mustard': (232,184,40), 'corn': (228,205,5),
    'medallion': (227,177,4), 'dandelion': (253,206,42), 'yellowfire': (253,165,15),
    'bumblebee': (252,226,5), 'banana': (252,244,163), 'butterscotch': (252,188,2),
    'dijon': (194,146,0), 'honey': (255,195,11), 'blonde': (254,235,117),
    'pineapple': (254,226,39), 'tuscansun': (252,209,42),
}
globals().update({name: np.array(rgb) / 255 for name, rgb in _PALETTE.items()})
lightgreen = '#95F985'
myGr = '#77AC30'
myBl = '#0072BD'
myYl = rgb2hex(gold)
myGray = np.array([17,17,17]) / 255
lightBlue = rgb2hex([91,207,244])
purple = rgb2hex([103,2,94])
KF_COLOR = [myYl, lightBlue, myBl, lightBlue, myYl, 'r', 'r']
KF_LINE = ['--'] * 6
KF_MARKER = ['^', 'v', 'd', 'h', 's', 'o']
KF_LINE_WIDTH = KNET_LINE_WIDTH = 4
KF_MARKER_SIZE = KNET_MARKER_SIZE = 15
KNET_COLOR = [myGr, myBl, myBl, myGr, myBl]
KNET_LINE = ['-'] * 6
KNET_MARKER = ['o', 'd', 'd', 'h', 's', 'o']
q_str = r'$\frac{q^2}{r^2} =$'
J_mdl_str = r'$J_{mdl} =$'
EKF_str = r'$\mathrm{EKF}$'
KNET_str = r'$\mathrm{KalmanNet}$'
del_str = ' - '
del2_str = ' : '
space_str = ' '
db_str = r'$\mathrm{[dB]}$'
BL_str = 'Noise Floor'
q_mdl_str = r'$q_{mdl}^2 - q_{gen}^2 =$'
r_mdl_str = r'$r_{mdl}^2 - r_{gen}^2 =$'
