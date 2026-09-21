    
    xl = xlabel("$$\mathrm{\frac{1}{r^2}} \textrm{ [dB]}$$", 'interpreter','latex') ;
    xl.FontSize = 36 ;

    yl = ylabel("MSE [dB]") ;
    yl.FontSize = 36 ;


    delim = " ; " ;
    modelSize = "$$m$$x$$n$$" + " $$=$$ " + m + "x" + n ;
    T_str = "$$\mathrm{T} = $$" + " " + T;
    dt_str = "$$\mathrm{\Delta t = }$$" + " " + dt ;
    Jgen_str = "$$\mathrm{J_{gen} = }$$" + " " + J ;
    q_str = "$$\mathrm{\frac{q}{r}[dB] = }$$" + " " + q_arr_dB(qIdx) ;

%     myTitle = "Lorenz Attractor - Non Linear: " + modelSize + delim + T_str + delim + dt_str  + delim + Jgen_str ; %+ delim + q_str ;
% 
%     tl = title(myTitle, 'Interpreter', 'latex') ;
%     tl.FontSize = 36 ; 