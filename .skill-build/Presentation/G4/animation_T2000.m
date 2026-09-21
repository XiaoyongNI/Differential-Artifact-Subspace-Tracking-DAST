fileID = fopen('data/target_mean.txt','r');
formatSpec = '%f';
EKF_mean = fscanf(fileID,formatSpec);
EKF_mean = reshape(EKF_mean,3,2000);
t = 1:2000;
x = EKF_mean(1,:);
y = EKF_mean(2,:);
z = EKF_mean(3,:);
view(3)
% axis([-150 100 -100 350 -500 500]);
axis square
xlabel('x(t)')
ylabel('y(t)')
zlabel('z(t)')
grid on
hold on
h = plot3(x(1),y(1),z(1),'r');
for k=2:length(t)
    set(h,'xdata',x(1:k),'ydata',y(1:k),'zdata',z(1:k));
    pause(0.002);
end
hold off
    