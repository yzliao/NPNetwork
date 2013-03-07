% NPNetworkIIRLMSTest.m

% IIRLMSTest.m
%  
%  IIR Filter:
%
%               1
%  H(z) = ------------
%         1 - 0.8*z^-1
%

addpath('../')

clear all
close all

% Parameters
L = 30;
N = 1500;
mu = 0.05;
M = 100;

% Initialize NPNetwork
myNPNetwork = NPNetwork(L,M,N);
myNPNetwork.setStepSize('step size',mu);

% input
plant_input = randn(N+L,1);
plant_output = filter(1,[1,-0.8],plant_input);
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

% Testing
test_input = randn(1e5,1);
test_output = filter(1,[1,-0.8],test_input);
myNPNetwork.setTesting(test_input,test_output);

myNPNetwork.Testing('Linear','No Hidden Layer');

% Plot error
[testing_error,testing_output] = myNPNetwork.getOutputSignal('Testing');

figure(3)
plot(testing_error.^2);

figure(4)
plot(test_output,'r-o');
hold on
plot(testing_output,'b');


