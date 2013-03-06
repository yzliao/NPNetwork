% LMSTest.m

%clear all
close all

% Parameters
L = 5;
N = 1000;
mu = 0.05;
M = 100;

% input
% plant_input = randn(N+L,1);
% plant_output = filter([0.4,0.3,0.2],1,plant_input);

% training
adaptive_weights = zeros(L,1);
error_vector = zeros(N+L,1);
filter_output_vector = zeros(N+L,1);

% streaming

% for i = L:N+L,
%     xtdl = plant_input(i:-1:i-L+1);
%     
%     filter_output = adaptive_weights'*xtdl;
%     
%     filter_error = plant_output(i) - filter_output;
%     
%     % update weights
%     % theta_j = theta_j + mu*(y^(i) - theta_j'*x^(i))*x_j^(i)
%     adaptive_weights = adaptive_weights + mu*filter_error*xtdl;
%     
%     error_vector(i) = filter_error;
%     filter_output_vector(i) = filter_output;
%     
% end

% matrix
xTrainingMtx = zeros(L,N);
for i = L:N+L,
    xtdl = plant_input(i:-1:i-L+1);
    xTrainingMtx(:,i) = xtdl;
end

for i = L:N+L,
    xtdl = xTrainingMtx(:,i);
    
    filter_output = adaptive_weights'*xtdl;
    
    filter_error = plant_output(i) - filter_output;
    
    % update weights
    % theta_j = theta_j + mu*(y^(i) - theta_j'*x^(i))*x_j^(i)
    adaptive_weights = adaptive_weights + mu*filter_error*xtdl;
    
    error_vector(i) = filter_error;
    filter_output_vector(i) = filter_output;
end

figure(1)
plot(error_vector.^2);
ylabel('MSE');

figure(2)
plot(plant_output,'r-o');
hold on
plot(filter_output_vector,'b');
ylabel('Output');
legend('Plant Output','Filter Output');