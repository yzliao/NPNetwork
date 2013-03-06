% NonLinearLMSTest.m

clear all
close all

% Filter 1:
%
%
% H(Z) = 0.8+ 0.7*z^-1 + 0.6*z^-2
%
% Filter 2:
%
%                1
% H(z) = ----------------
%        1 - 0.8*z^-1
%
% Each filter has a sigmoid function at output
%
%
% sigmoid(x)= -1 + 2/(1+exp(-x))
%
% sigmoid(x)' = 0.5*(1-sigmoid(x)^2)
%

sigmoid = @(x) -1+2./(1+exp(-x));



% Parameters
L1 = 5;
L2 = 25;
N = 80000;
mu = 0.001;
M = 500;


% input
plant_input = randn(N+L2,1);
% plant_output_1 = sigmoid(filter([0.8,0.7,0.6],1,plant_input));
plant_output_2 = sigmoid(filter(1,[1,-0.8],plant_input));

% fixed weights
% fixed_weights_1 = randn(M,L1);
% fixed_weights_2 = randn(M,L2);
fixed_weights_2 = -1+2*rand(M,L2);

% training

% filter 1
% adaptive_weights_1 = zeros(M,1);
% error_vector_1 = zeros(N+L1,1);
% filter_output_vector_1 = zeros(N+L1,1);

% % streaming
% 
% for i = L1:N+L1,
%     xtdl = plant_input(i:-1:i-L1+1);
%     
%     % fixed layer
%     ytdl = fixed_weights_1*xtdl;
%     xtdl2 = sigmoid(ytdl);
%     
%     % adaptive layer
%     s = adaptive_weights_1'*xtdl2;
%     filter_output = sigmoid(s);
%     
%     filter_error = plant_output_1(i) - filter_output;
%     
%     sigmoid_dot = 0.5*(1-filter_output.^2);
%     
%     % update weights
%     % theta = theta + 2*mu*sigmoid_dot*error*xtdl2
%     adaptive_weights_1 = adaptive_weights_1 + 2*mu*sigmoid_dot*filter_error*xtdl2;
%     
%     error_vector_1(i) = filter_error;
%     filter_output_vector_1(i) = filter_output;
% end
% 
% figure(1)
% plot(error_vector_1.^2);
% ylabel('MSE')
% title('Filter 1');
% 
% figure(2)
% plot(plant_output_1,'r-o');
% hold on
% plot(filter_output_vector_1,'b');
% ylabel('Output');
% legend('Plant Output','Filter Output');
% title('Filter 1');


% filter 2
adaptive_weights_2 = zeros(M,1);
error_vector_2 = zeros(N+L2,1);
filter_output_vector_2 = zeros(N+L2,1);

% streaming
for i = L2:N+L2,
    xtdl = plant_input(i:-1:i-L2+1);
    
    % fixed layer
    ytdl = fixed_weights_2*xtdl;
    xtdl2 = sigmoid(ytdl);
    
    % adaptive layer
    s = adaptive_weights_2'*xtdl2;
    filter_output = sigmoid(s);
    
    filter_error = plant_output_2(i) - filter_output;
    
    sigmoid_dot = 0.5*(1-filter_output.^2);
    
    % update weights
    % theta = theta + 2*mu*sigmoid_dot*error*xtdl2
    adaptive_weights_2 = adaptive_weights_2 + 2*mu*sigmoid_dot*filter_error*xtdl2;
    
    error_vector_2(i) = filter_error;
    filter_output_vector_2(i) = filter_output;
end

figure(3)
plot(error_vector_2.^2);
ylabel('MSE')
title('Filter 2');

mean(error_vector_2(end*9/10:end).^2)

% figure(4)
% plot(plant_output_2,'r-o');
% hold on
% plot(filter_output_vector_2,'b');
% ylabel('Output');
% legend('Plant Output','Filter Output');
% title('Filter 2');
    
