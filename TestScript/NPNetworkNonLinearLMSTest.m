% NNPNetworkNonLinearLMSTest.m

clear all
close all

addpath('../');

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
N = 8e4;
mu = 0.001;
M = 500;

% Set up NPNetwork
myNetwork_1 = NPNetwork(L1,M,N);
myNetwork_2 = NPNetwork(L2,M,N);

myNetwork_1.setStepSize('step size',mu);
myNetwork_2.setStepSize('step size',mu);

% input
plant_input = randn(N+L2,1);
plant_output_1 = sigmoid(filter([0.8,0.7,0.6],1,plant_input));
plant_output_2 = sigmoid(filter(1,[1,-0.8],plant_input));

plant_input1 = streaming2mtx(plant_input,L1,N,L1);
plant_input2 = streaming2mtx(plant_input,L2,N,L2);

myNetwork_1.setTraining(plant_input1,plant_output_1);
myNetwork_2.setTraining(plant_input2,plant_output_2);

% fixed weights
myNetwork_1.initFixedWeights();
myNetwork_2.initFixedWeights();

% Training
myNetwork_1.Training('Nonlinear','Hidden Layer');
%myNetwork_2.Training('Nonlinear','Hidden Layer');

% Plot Error
[error_vector_1,filter_output_vector_1] = myNetwork_1.getOutputSignal('Training');
[error_vector_2,filter_output_vector_2] = myNetwork_2.getOutputSignal('Training');

figure(1)
plot(error_vector_1.^2);
ylabel('MSE')
title('Filter 1');

figure(2)
plot(plant_output_1,'r-o');
hold on
plot(filter_output_vector_1,'b');
ylabel('Output');
legend('Plant Output','Filter Output');
title('Filter 1');

% figure(3)
% plot(error_vector_2.^2);
% ylabel('MSE')
% title('Filter 2');
% 
% mean(error_vector_2(end*9/10:end).^2)
% 
% figure(4)
% plot(plant_output_2,'r-o');
% hold on
% plot(filter_output_vector_2,'b');
% ylabel('Output');
% legend('Plant Output','Filter Output');
% title('Filter 2');

% Test filter 1
test_input = randn(1e5,1);
test_output_1 = sigmoid(filter([0.8,0.7,0.6],1,test_input));
myNetwork_1.setTesting(streaming2mtx(test_input,L1,length(test_input)-L1,L1),test_output_1);

% Testing
myNetwork_1.Testing('Nonlinear','Hidden Layer');

% plot error
[testing_error_1,testing_output_1] = myNetwork_1.getOutputSignal('Testing');
figure(5)
plot(testing_error_1.^2);

% Train filter 1 by BP
[elms,ylms,hidden_weights_cell,output_weights] = myNetwork_1.BP_Training(0,'Nonlinear');

figure(6)
plot(elms.^2);
ylabel('MSE')
title('filter 1 BP');

figure(7)
plot(plant_output_1,'r-o');
hold on
plot(ylms,'b');
ylabel('Output');
legend('Plant Output','Filter Output');
title('Filter 1 BP');

[elms,ylms] = myNetwork_1.BP_Testing(hidden_weights_cell,output_weights,'Nonlinear');

figure(8)
plot(elms.^2);



