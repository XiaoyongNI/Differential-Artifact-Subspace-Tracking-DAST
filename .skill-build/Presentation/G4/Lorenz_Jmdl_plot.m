close all

style


J_work_EKF = [1, 3] ; %[1, 2, 3] ;

J_work_KNET = [3] ;% [1, 3] ;

%%%%%%%%%%%%%%%%%%%%
%%% Define Legend %%
%%%%%%%%%%%%%%%%%%%%

J_num_EKF = num2str(Jmdl_arr(J_work_EKF)) ;
J_num_KNET = num2str(Jmdl_arr(J_work_KNET)) ;


%%%%%%%%%%%%%%%%%%%%
%%% Define Legend %%
%%%%%%%%%%%%%%%%%%%%

% myLegend = [
%     EKF_str + del2_str + J_mdl_str + space_str + J_num_EKF(1),
%     EKF_str + del2_str + J_mdl_str + space_str + J_num_EKF(2),
%     EKF_str + del2_str + J_mdl_str + space_str + J_num_EKF(3),
%     BL_str
% ] ;

myLegend = [
    EKF_str + del2_str + J_mdl_str + space_str + J_num_EKF(1),
    EKF_str + del2_str + J_mdl_str + space_str + J_num_EKF(2),
    KNET_str + del2_str + J_mdl_str + space_str + J_num_KNET(1),
    BL_str
] ;


Data = MSE ;
Data_KNET = MSE_KNET ;




figure

set(gca,'FontSize',80)


%%%%%%%%%%%%%
%%% J_mdl %%%
%%%%%%%%%%%%%%

for j = 1:1:length(J_work_EKF)

    Jidx = J_work_EKF(j) ;

    plot(-r2_arr_dB, Data(:, Jidx), 'Color', KF_COLOR(j), 'LineStyle', KF_LINE(j), 'LineWidth', KF_LINE_WIDTH, 'Marker', KF_MARKER(j), 'MarkerSize', KF_MARKER_SIZE, 'MarkerEdgeColor', 'k') ;

    hold on

end
    
%%%%%%%%%%%%%
%%% J_mdl %%%
%%%%%%%%%%%%%

for j = 1:1:length(J_work_KNET)

    Jidx = J_work_KNET(j) ;

    plot(-r2_arr_dB, Data_KNET(:, Jidx), 'Color', KNET_COLOR(2), 'LineStyle', KNET_LINE(2), 'LineWidth', KNET_LINE_WIDTH, 'Marker', KNET_MARKER(2), 'MarkerSize', KNET_MARKER_SIZE, 'MarkerEdgeColor', 'k') ;

    hold on

end
    
    
plot(-r2_arr_dB, r2_arr_dB, 'Color', 'r', 'LineStyle', '-.', 'LineWidth', KNET_LINE_WIDTH, 'MarkerEdgeColor', 'k') ;

%%%%%%%%%%%%%%%%%%%%%
%%% Figure Config %%%
%%%%%%%%%%%%%%%%%%%%%


xlim([0, 40])
ylim([-50, 0])

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




    




