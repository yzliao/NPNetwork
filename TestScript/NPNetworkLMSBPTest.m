% NPNetworkLMSBPTest.m

clear all
close all

% Parameters
L = 5;
N = 200;
mu = 0.01;
M = [10];

addpath('..');

% Initalize NPNetwork
myNPNetwork = NPNetwork(L,M,N);
myNPNetwork.setStepSize('step size',mu);

% input
plant_input = randn(N+L,1);
plant_output = filter([0.4,0.3,0.5],1,plant_input);
myNPNetwork.setTraining(streaming2mtx(plant_input,L,N,L),plant_output);

% Training
%myNPNetwork.Training('Linear','No Hidden Layer');
[training_error,training_output,~,~] = myNPNetwork.BP_Training('Linear');

% Plot Error
%[training_error,training_output] = myNPNetwork.getOutputSignal('Training');

figure(1)
plot(training_error.^2);

figure(2)
plot(plant_output,'r-o');
hold on
plot(training_output,'b');
