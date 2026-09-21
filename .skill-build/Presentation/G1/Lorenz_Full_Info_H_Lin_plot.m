close all

style


J_work_EKF = [1] ;

J_work_KNET = [1] ;

J_mdl_arr = [5] ;

%%%%%%%%%%%%%%%%%%%%
%%% Define Legend %%
%%%%%%%%%%%%%%%%%%%%

J_num_EKF = num2str(J_mdl_arr(J_work_EKF)) ;
J_num_KNET = num2str(J_mdl_arr(J_work_KNET)) ;

% myLegend = [
%     q_str + " 0" + db_str + del_str +  EKF_str + del2_str + J_mdl_str + space_str + J_num_EKF(1),
%     q_str + " 0" + db_str + del_str +  KNET_str + del2_str + J_mdl_str + space_str + J_num_EKF(1),
%     q_str + " -20" + db_str + del_str +  EKF_str + del2_str + J_mdl_str + space_str + J_num_EKF(1),
%     q_str + " -20" + db_str + del_str +  KNET_str + del2_str + J_mdl_str + space_str + J_num_EKF(1),
%     BL_str
% ] ;

myLegend = [
    q_str + " 0" + db_str + del_str +  EKF_str
    q_str + " 0" + db_str + del_str +  KNET_str
    q_str + " -20" + db_str + del_str +  EKF_str
    q_str + " -20" + db_str + del_str +  KNET_str
    BL_str
] ;

Data = MSE ;
Data_KNET = MSE_KNET ;


figure

set(gca,'FontSize',80)


for qIdx=1:1:2

%%%%%%%%%%%%
%%% Plot %%%
%%%%%%%%%%%%'

    plot(-r2_arr_dB, Data(:, qIdx), 'Color', KF_COLOR(qIdx), 'LineStyle', KF_LINE(qIdx), 'LineWidth', KF_LINE_WIDTH, 'Marker', KF_MARKER(qIdx), 'MarkerSize', KF_MARKER_SIZE, 'MarkerEdgeColor', 'k') ;

    hold on


    plot(-r2_arr_dB, Data_KNET(:, qIdx), 'Color', KNET_COLOR(qIdx), 'LineStyle', KNET_LINE(qIdx), 'LineWidth', KNET_LINE_WIDTH, 'Marker', KNET_MARKER(qIdx), 'MarkerSize', KNET_MARKER_SIZE, 'MarkerEdgeColor', 'k') ;

    hold on
 
end

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


    




