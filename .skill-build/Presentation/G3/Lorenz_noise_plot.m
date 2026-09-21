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


%%%%%%%%%%%%%%%%%%%%
%%% Define Legend %%
%%%%%%%%%%%%%%%%%%%%
% 
% myLegend = [
%     q_str + " 0" + db_str + del_str +  EKF_str + del2_str + J_mdl_str + space_str + J_num_EKF(1),
%     q_str + " 0" + db_str + del_str +  EKF_str + del2_str + J_mdl_str + space_str + J_num_EKF(1) + " , " + q_mdl_str + space_str + "-6" + db_str,
%     q_str + " 0" + db_str + del_str +  EKF_str + del2_str + J_mdl_str + space_str + J_num_EKF(1) + " , " + q_mdl_str + space_str + "-6" + db_str + " , " + r_mdl_str + space_str + "+6" + db_str,
%     q_str + " 0" + db_str + del_str +  KNET_str + del2_str + J_mdl_str + space_str + J_num_EKF(1),
%     BL_str
% ] ;

myLegend = [
    EKF_str
    EKF_str +  del2_str + "$$\Delta \mathrm{q}^2 = $$" +  " -6" + db_str,
    EKF_str +  del2_str + "$$\Delta \mathrm{q}^2 = $$" +  " -6" + db_str + " , " + "$$\Delta \mathrm{r}^2 = $$" + " +6" + db_str,
    KNET_str,
    BL_str
] ;

Data = MSE ;
Data_KNET = MSE_KNET ;

figure

set(gca,'FontSize',80)


for mIdx=1:1:3

%%%%%%%%%%%%
%%% Plot %%%
%%%%%%%%%%%%'

    plot(-r2_arr_dB, Data(:, mIdx), 'Color', KF_COLOR(mIdx), 'LineStyle', KF_LINE(mIdx), 'LineWidth', KF_LINE_WIDTH, 'Marker', KF_MARKER(mIdx), 'MarkerSize', KF_MARKER_SIZE, 'MarkerEdgeColor', 'k') ;

    hold on

end

mIdx = 2 ;

plot(-r2_arr_dB, Data_KNET(:), 'Color', KNET_COLOR(1), 'LineStyle', KNET_LINE(1), 'LineWidth', KNET_LINE_WIDTH, 'Marker', KNET_MARKER(1), 'MarkerSize', KNET_MARKER_SIZE, 'MarkerEdgeColor', 'k') ;
        
hold on
%     
plot(-r2_arr_dB, r2_arr_dB, 'Color', 'r', 'LineStyle', '-.', 'LineWidth', KNET_LINE_WIDTH, 'MarkerEdgeColor', 'k') ;

    %%%%%%%%%%%%%%%%%%%%%
    %%% Figure Config %%%
    %%%%%%%%%%%%%%%%%%%%%
    
    ax = gca ;
    ax.XAxis.FontSize = 30;
    ax.YAxis.FontSize = 30;

    grid
    
    xlim([0, 40]) ;
    ylim([-42.5, 2.5]) ;

    %%%%%%%%%%%%%%
    %%% Legend %%%
    %%%%%%%%%%%%%%
    lgd = legend(myLegend, 'Interpreter', 'latex');
    lgd.FontSize = 35;
    
    
    %%%%%%%%%%%%%
    %%% Label %%%
    %%%%%%%%%%%%%
    myLabel



