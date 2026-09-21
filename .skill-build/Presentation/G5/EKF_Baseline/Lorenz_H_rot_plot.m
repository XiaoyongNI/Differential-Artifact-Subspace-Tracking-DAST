close all

style


H_rot_work_EKF = [1, 2, 3, 4, 5] ;

J_work_KNET = [1, 3] ;

%%%%%%%%%%%%%%%%%%%%
%%% Define Legend %%
%%%%%%%%%%%%%%%%%%%%

J_num_EKF = num2str(H_rot_arr(H_rot_work_EKF)) ;
J_num_KNET = num2str(H_rot_arr(J_work_KNET)) ;

EKF_J_mdl_str = "$$\mathrm{EKF: \Delta\theta} = $$ " ;
deg_str = "$$^\circ$$";

myLegend = [
    EKF_J_mdl_str + J_num_EKF(1, :) + deg_str,
    EKF_J_mdl_str + J_num_EKF(2, :) + deg_str,
    EKF_J_mdl_str + J_num_EKF(3, :) + deg_str,
    EKF_J_mdl_str + J_num_EKF(4, :) + deg_str,
    EKF_J_mdl_str + J_num_EKF(5, :) + deg_str,
    BL_str
] ;

Data = MSE ;


%%%%%%%%%%%%
%%% Plot %%%
%%%%%%%%%%%%'

figure

set(gca,'FontSize',60)
    

%%%%%%%%%%%%%
%%% J_mdl %%%
%%%%%%%%%%%%%%

for j = 1:1:length(H_rot_work_EKF)

    Jidx = H_rot_work_EKF(j) ;

    plot(-r2_arr_dB, Data(:, Jidx), 'Color', KF_COLOR(Jidx), 'LineStyle', KF_LINE(Jidx), 'LineWidth', KF_LINE_WIDTH, 'Marker', KF_MARKER(Jidx), 'MarkerSize', KF_MARKER_SIZE, 'MarkerEdgeColor', 'k') ;

    hold on

end
    
      
plot(-r2_arr_dB, r2_arr_dB, 'Color', 'r', 'LineStyle', '-.', 'LineWidth', KNET_LINE_WIDTH, 'MarkerEdgeColor', 'k') ;

%%%%%%%%%%%%%%%%%%%%%
%%% Figure Config %%%
%%%%%%%%%%%%%%%%%%%%%


%xlim([2, 20])

ax = gca ;
ax.XAxis.FontSize = 24;
ax.YAxis.FontSize = 24;

grid

%%%%%%%%%%%%%%
%%% Legend %%%
%%%%%%%%%%%%%%
lgd = legend(myLegend, 'Interpreter', 'latex');
lgd.FontSize = 20;


%%%%%%%%%%%%%
%%% Label %%%
%%%%%%%%%%%%%
myLabel

    




