% TestNPNetwork.m

clear all
close all
clc

% Basic Setup
L = 10;
N = 15;
M_vec = [10,10];

myobj = NPNetwork(L,M_vec,N);
myobj.setBasicParameter('N',30);
myobj.setStepSize('misadjustment',0.05);
myobj.setTrainingTimes(20);

% test distribution
myobj.setDistribution('Uniform');
%myobj.setDistribution('Possion');
myobj.showDistributionInfo();

%myobj.setDistribution('Uniform',1,1);
myobj.setDistribution('Uniform',-1,1);

% init weights
myobj.initFixedWeights();

% set training signal
x_training = [20,20];
d_training = [10,10];

myobj.setTraining(x_training,d_training);

% get output
[err,y] = myobj.getOutputSignal('Training')

% save object
myobj.saveObj('test.mat','Weights','Training');

% load object
myobj2 = loadObj('test.mat')

% load object to an object
myobj3 = NPNetwork();
myobj3.loadObj('test.mat');