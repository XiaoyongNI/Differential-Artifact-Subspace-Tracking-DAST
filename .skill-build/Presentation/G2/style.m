%%%%%%%%%%%%%%%
%%% Coloros %%%
%%%%%%%%%%%%%%%

yellow=[253/255,230/255,75/255];
canary=[249/255,200/255,2/255];
gold=[249/255,166/255,2/255];
daffodil=[253/255,238/255,135/255];
flaxen=[214/255,183/255,90/255];
butter=[254/255,226/255,39/255];
lemon=[239/255,253/255,95/255];
mustard=[232/255,184/255,40/255];
corn=[228/255,205/255,5/255];
medallion=[227/255,177/255,4/255];
dandelion=[253/255,206/255,42/255];
yellowfire=[253/255,165/255,15/255];
bumblebee=[252/255,226/255,5/255];
banana=[252/255,244/255,163/255];
butterscotch=[252/255,188/255,2/255];
dijon=[194/255,146/255,0/255];
honey=[255/255,195/255,11/255];
blonde=[254/255,235/255,117/255];
pineapple=[254/255,226/255,39/255];
tuscansun=[252/255,209/255,42/255];

lightgreen = '#95F985';
myGr = "#77AC30";
myBl = "#0072BD";
myYl = convertCharsToStrings(rgb2hex(gold)) ;

myGray = [17 17 17]/255 ;

lightBlue = [91, 207, 244] / 255; 
lightBlue = convertCharsToStrings(rgb2hex(lightBlue)) ;

purple = [103, 2, 94] / 255;
purple = convertCharsToStrings(rgb2hex(purple)) ;

KF_COLOR = [myGr, myYl, 'r', lightBlue, myYl, 'r', 'r'];
KF_LINE   = ["--", "--", "--", "--", "--", "--"];  
KF_MARKER   = ["^", "v", "d", "h", 's', 'o'];  
KF_LINE_WIDTH = 4 ;
KF_MARKER_SIZE = 15 ;

KNET_COLOR =  [myBl, myBl, myBl, myGr, myBl];
KNET_LINE =   ['-', '-', '-', '-', '-', '-'];
KNET_MARKER   = ["o", "d", "d", "h", 's', 'o'];  
KNET_LINE_WIDTH = 4 ;
KNET_MARKER_SIZE = 15 ;


q_str = "$$\mathrm{\frac{q^2}{r^2} =}$$" ;
J_mdl_str = "$$\mathrm{J_{mdl}} =$$" ; 
EKF_str = "$$\mathrm{EKF}$$" ;
KNET_str = "$$\mathrm{KalmanNet}$$" ;
del_str = " - " ;
del2_str = " : ";
space_str = " ";
db_str = "$$\mathrm{[dB]}$$" ;
BL_str = "Noise Floor" ;

q_mdl_str = "$$\mathrm{q_{mdl}^2 - q_{gen}^2 =}$$" ;
r_mdl_str = "$$\mathrm{r_{mdl}^2 - r_{gen}^2 =}$$" ;