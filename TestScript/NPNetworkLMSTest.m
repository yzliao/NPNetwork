% NPNetworkLMSTest.m

clear all
close all

% Parameters
L = 5;
N = 1000;
mu = 0.05;
M = 100;

% Initalize NPNetwork
myNPNetwork = NPNetwork(L,M,N);
myNPNetwork.setStepSize('step size',mu);

% input
plant_input = randn(N+L,1);
plant_output = filter([0.4,0.3,0.2],1,plant_input);
myNPNetwork.setTraining(plant_input,plant_output);

% Training
myNPNetwork.Training('Linear','No Hidden Layer');

% Plot Error
[training_error,training_output] = myNPNetwork.getOutputSignal('Training');

figure(1)
plot(training_error.^2);

figure(2)
plot(plant_output,'r-o');
hold on
plot(training_output,'b');
