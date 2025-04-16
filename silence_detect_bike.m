function silence_activity = silence_detect_bike(data,ws,thresh)
%% Function to detect pedalling activity in cycling data. 
% Created by J.I.M. Parmentier
% February 2024
% Univeristy of Twente
% Contact: j.i.m.parmentier@utwente.nl

% Notes: Data used to create this function was collected from an IMU sensor 
% sampling at 200Hz, attached to the left crank arm, with the x-axis along 
% the axis of the crank arm.


% % Debug input
% data_temp = data_filt.accel(:,1);
% ws = window_size/5;
% thresh = 10;

% figure
% plot(data_temp); hold on;  
data = movvar(data,ws);
% plot(data);
data = movmean(data,ws);
% plot(data);

silence_activity = (data>= thresh);

% figure;
% plot(data_temp); hold on
% plot(silence*10)
end