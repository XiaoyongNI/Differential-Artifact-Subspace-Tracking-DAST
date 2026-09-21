close all

style


H_rot_work_EKF = [1, 2] ;


%%%%%%%%%%%%%%%%%%%%
%%% Define Legend %%
%%%%%%%%%%%%%%%%%%%%

J_num_EKF = num2str(H_rot_arr(H_rot_work_EKF)) ;
J_num_KNET = J_num_EKF ;

EKF_H_rot_str = "$$\mathrm{EKF: \Delta\theta} = $$ " ;
KNET_H_rot_str = "$$\mathrm{KalmanNet: \Delta\theta} = $$ " ;
BL_str = "Noise Level" ;


myLegend = [
    EKF_H_rot_str + J_num_EKF(1, :),
    EKF_H_rot_str + J_num_EKF(2, :),
    KNET_H_rot_str + J_num_KNET(2, :),
    BL_str
] ;

Data = MSE ;


%%%%%%%%%%%%
%%% Plot %%%
%%%%%%%%%%%%'

    figure

    set(gca,'FontSize',80)
    

%%%%%%%%%%%%%
%%% J_mdl %%%
%%%%%%%%%%%%%%

for j = 1:1:length(H_rot_work_EKF)

    Jidx = H_rot_work_EKF(j) ;

    plot(-r2_arr_dB, Data(:, Jidx), 'Color', KF_COLOR(Jidx), 'LineStyle', KF_LINE(Jidx), 'LineWidth', KF_LINE_WIDTH, 'Marker', KF_MARKER(Jidx), 'MarkerSize', KF_MARKER_SIZE, 'MarkerEdgeColor', 'k') ;

    hold on

end
    
plot(-r2_arr_dB, MSE_KNet(:), 'Color', KNET_COLOR(Jidx), 'LineStyle', KNET_LINE(Jidx), 'LineWidth', KNET_LINE_WIDTH, 'Marker', KNET_MARKER(Jidx), 'MarkerSize', KNET_MARKER_SIZE, 'MarkerEdgeColor', 'k') ;
    
hold on

plot(-r2_arr_dB, r2_arr_dB, 'Color', 'r', 'LineStyle', '-.', 'LineWidth', KNET_LINE_WIDTH, 'MarkerEdgeColor', 'k') ;

%%%%%%%%%%%%%%%%%%%%%
%%% Figure Config %%%
%%%%%%%%%%%%%%%%%%%%%


xlim([0, 40])
ylim([-51, 0])

ax = gca ;
ax.XAxis.FontSize = 30;
ax.YAxis.FontSize = 30;

grid

%%%%%%%%%%%%%%
%%% Legend %%%
%%%%%%%%%%%%%%
lgd = legend(myLegend, 'Interpreter', 'latex');
lgd.FontSize = 35;


%%%%%%%%%%%%%
%%% Label %%%
%%%%%%%%%%%%%
myLabel




    




