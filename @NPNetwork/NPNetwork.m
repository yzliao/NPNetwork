classdef NPNetwork<handle
    % NPNetwork is a Matlab class which constructs and implements the no
    % propagation adaptive neural network
    
    % LYZ @ Jan 25th, 2013
    
    properties
        L                   = 0;        % Tapped delay line
        N                   = 0;        % Number of input patterns
        misadj              = 0;        % Misadjustment
        mu                  = 0;        % Step size
        NumOfHiddenLayer    = 1;        % Number of hidden layers
        TrainingIter        = 1;        % Training iteration
        
        M_vec               = 0;        % Number of neurons on each layer
                                        % length(M_vec) =
                                        % NumOfHiddenLayer
        outputLayer         = 0;        % Number of neurons on output layer
    end
    
    properties(Access = protected)
        % adaptive weights
        adaptive_weights = 0;
        
        % fixed weights
        fixed_weights_vec = 0;
    end
    
    properties(Access = protected)
        % random fixed weights distribution
        RndDistribution = 'Normal'      % random distribution
        RndVar1 = 0;                    % normal: mean; uniform: a
        RndVar2 = 1;                    % normal: variance; uniform: b
    end
    
    properties(Access = private)
        ifsetmu = false;                % if mu is set by the users
    end
    
    properties(Access = protected)
        % signal info
        
        % training
        x_training = 0;
        d_training = 0;
        
        y_training = 0;
        e_training = 0;
        
        y_BPtraining = 0;
        e_BPtraining = 0;
        
        
        % testing
        x_testing = 0;
        d_testing = 0;
        
        y_testing = 0;
        e_testing = 0; 
        
        y_BPtesting = 0;
        e_BPtesting = 0;
    end
    
    properties(Access = protected)
        % BP variables
        BP_hidden_weights = 0;
        BP_output_weights = 0;
    end
        
    
    
    methods
        % constructor
        function obj = NPNetwork(L,M_vec,N,output_layer)
            if nargin > 0,
                obj.L = L;
                obj.M_vec = M_vec;

                obj.N = N;

                obj.NumOfHiddenLayer = length(M_vec);
                
                obj.outputLayer = output_layer;

                % only the output layer is adaptive. 
                % the number of neurons on the output layer = the number of
                % neurons on the last hidden layer
                obj.adaptive_weights = zeros(M_vec(end),output_layer);
            else
                obj.L = 0;
                obj.M_vec = 0;
                obj.N = 0;
                obj.NumOfHiddenLayer = 0;
                obj.adaptive_weights = 0;
            end
        end
        
        % change constructor parameter
        function setBasicParameter(obj,var1,var2)
            switch var1,
                case 'L'
                    obj.L = var2;
                case {'M','M_vec'}
                    obj.M_vec = var2;
                case 'N',
                    obj.N = var2;
                otherwise
                    display('Basic Parameters: L, M/M_vec, N');
            end
        end
        
        % set mu/misadjustment
        function setStepSize(obj,var1,var2)
            if strcmp(var1,'misadjustment')
                obj.misadj = var2;
            elseif strcmp(var1,'step size')
                obj.mu = var2;
                obj.ifsetmu = true;
            end
        end
        
        % set training iteration
        function setTrainingTimes(obj,iter)
            obj.TrainingIter = iter;
        end
        
        % set fixed weights distribution
        function setDistribution(obj,RndDistribution,RndVar1,RndVar2)
            if nargin > 1,
                if ~strcmp(RndDistribution,'Normal') && ...
                        ~strcmp(RndDistribution,'Uniform')
                    error(...
                        'We only support Normal Distribution and Uniform Distribution');
                end
                obj.RndDistribution = RndDistribution;
            end
            
            if nargin > 2,
                if strcmp(obj.RndDistribution,'Uniform') && ...
                        ((RndVar1 >= RndVar2) || (RndVar1 >= obj.RndVar2))
                    error('RndVar1 < RndVar2 is required for Uniform Distribution');
                end
                obj.RndVar1 = RndVar1;
            end
            
            if nargin > 3,
                obj.RndVar2 = RndVar2;
            end
            
            % Print Info
            obj.showDistributionInfo();
        end
        
        % show distribution parameter
        function showDistributionInfo(obj)
            display(['The distribution is ',obj.RndDistribution,' Distribution.']);
            if strcmp(obj.RndDistribution, 'Normal')
                display(['Mean is ', num2str(obj.RndVar1),'.']);
                display(['Variance is ', num2str(obj.RndVar2),'.']);
            else
                display(['Interval: (',num2str(obj.RndVar1),',',num2str(obj.RndVar2),...
                    ')']);
            end
        end
                
        % initalize fixed weights
        function initFixedWeights(obj)
            weights = cell(obj.NumOfHiddenLayer,1);
            a = obj.RndVar1;
            b = obj.RndVar2;
            if strcmp(obj.RndDistribution,'Normal')
                weights{1} = a + sqrt(b) * ...
                    randn(obj.M_vec(1),obj.L);
                for i = 2:obj.NumOfHiddenLayer,
                    weights{i} = a + sqrt(b) * ...
                        randn(obj.M_vec(i),obj.M_vec(i-1));
                end
            elseif strcmp(obj.RndDistribution,'Uniform')
                weights{1} = a + (b-a)*rand(obj.M_vec(1),obj.L);
                for i = 2:obj.NumOfHiddenLayer,
                    weights{i} = a + (b-a)*rand(obj.M_vec(i),...
                        obj.M_vec(i-1));
                end
            end
            obj.fixed_weights_vec = weights;
            display(['Fixed weights are generated by ',obj.RndDistribution,...
                ' Distribution.']);
            obj.showDistributionInfo();
        end
        
        % initalize fixed weights by layer
        function initFixedWeightsbyLayer(obj,layer)
            a = obj.RndVar1;
            b = obj.RndVar2;
            if strcmp(obj.RndDistribution,'Normal')
                if layer == 1,
                    weights = a + sqrt(b) * randn(obj.M_vec(1),obj.L);
                else
                    weights = a + sqrt(b) * randn(obj.M_vec(layer),obj.M_vec(layer-1));
                end
            elseif strcmp(obj.RndDistribution,'Uniform')
                if layer == 1,
                    weights = a + (b-a)*rand(obj.M_vec(1),obj.L);
                else
                    weights = a + (b-a)*rand(obj.M_vec(layer),obj.M_vec(layer-1));
                end
            end
            obj.fixed_weights_vec{layer} = weights;
        end
        
        % get fixed weights
        function weights = getFixedWeights(obj)
            weights = obj.fixed_weights_vec;
        end
        
        % get adaptive weights
        function weights = getAdaptiveWeights(obj)
            weights = obj.adaptive_weights;
        end
        
        % set adaptive weights
        function setAdaptiveWeights(obj,var1)
            obj.adaptive_weights = var1;
        end
        
        % get BP weights
        function [hidden_weights, output_weights] = getBPWeights(obj)
            hidden_weights = obj.BP_hidden_weights;
            output_weights = obj.BP_output_weights;
        end
        
        % set BP weights
        function setBPWeights(obj,hidden_weights,output_weights)
            obj.BP_hidden_weights = hidden_weights;
            obj.BP_output_weights = output_weights;
        end
        
        
        % set Training signal
        function setTraining(obj,x_training,d_training)
            
%             if length(x_training) ~= length(d_training)
%                 error('not same length');
%             elseif length(x_training) < (obj.N+obj.L)
%                 error('input signal is too short');
%             end
            
            obj.x_training = x_training;
            %obj.d_training = d_training;
            
            [p,~] = size(d_training);
            if p == obj.outputLayer,
                obj.d_training = d_training;
            else
                obj.d_training = d_training';
            end
            
            
            obj.e_training = zeros(size(x_training));
            obj.y_training = zeros(size(x_training));
        end
        
        % set Testing signal
        function setTesting(obj,x_testing,d_testing)
            
%             if length(x_testing) ~= length(d_testing)
%                 error('not same length');
%             elseif length(x_testing) < (obj.N+obj.L)
%                 error('input signal is too short');
%             end
            
            obj.x_testing = x_testing;
            %obj.d_testing = d_testing;
            
            [p,~] = size(d_testing);
            if p == obj.outputLayer,
                obj.d_testing = d_testing;
            else
                obj.d_testing = d_testing';
            end
            
            obj.y_testing = zeros(size(x_testing));
            obj.e_testing = zeros(size(x_testing));
        end
        
        % get output signal
        function [err,y] = getOutputSignal(obj,par)
            if strcmp(par,'Training')
                err = obj.e_training;
                y = obj.y_training;
            elseif strcmp(par,'Testing')
                err = obj.e_testing;
                y = obj.y_testing;
            else
                error('Incorrect input variable');
            end
        end
        
        % get BP output signal
        function [err,y] = getBPOutputSignal(obj,par)
            if strcmp(par,'Training')
                err = obj.e_BPtraining;
                y = obj.y_BPtraining;
            elseif strcmp(par,'Testing')
                err = obj.e_BPtesting;
                y = obj.y_BPtesting;
            else
                error('Incorrect input variable');
            end
        end
        
        % get input signal
        function [x,d] = getInputSignal(obj,par)
            if strcmp(par,'Training')
                x = obj.x_training;
                d = obj.d_training;
            elseif strcmp(par,'Testing')
                x = obj.x_testing;
                d = obj.y_testing;
            else
                error('Incorrect input variable');
            end
        end
        
        
        
        % save object
        function saveObj(obj,fn,varargin)
            if nargin > 1,
                saveObj = NPNetwork(obj.L,obj.M_vec,obj.N);
                saveObj.setStepSize('misadjustment',obj.misadj);
                if obj.mu ~= 0,
                    saveObj.setStepSize('step size',obj.mu);
                end
                saveObj.setTrainingTimes(obj.TrainingIter);
            end
            
            if nargin > 2,
                for i = 1:nargin-2,
                    par = varargin{i};
                    switch par,
                        case 'Weights'
                            saveObj.adaptive_weights = obj.getAdaptiveWeights();
                            saveObj.fixed_weights_vec = obj.getFixedWeights();
                        case 'Training'
                            saveObj.setTraining(obj.x_training,obj.d_training);
                        case 'Testing'
                            saveObj.setTesting(obj.x_testing,obj.d_testing);
                        case 'All'
                            saveObj.adaptive_weights = obj.getAdaptiveWeights();
                            saveObj.fixed_weights_vec = obj.getFixedWeights();
                            saveObj.setTraining(obj.x_training,obj.d_training);
                            saveObj.setTesting(obj.x_testing,obj.d_testing);
                        otherwise
                            display(['The parameter ',par,' is not valid']);
                    end
                end
            end
            
            save(fn,'saveObj');
        end
        
        % load obj
        function obj = loadObj(fn)
            x = load(fn);
            obj = x.saveObj;
        end
        
        % clea input signal
        function cleanInputSignal(obj,var1)
            if strcmp(var1, 'Testing')
                obj.setTesting(0,0);
            elseif strcmp(var1,'Training')
                obj.setTraining(0,0);
            end
        end
        
        % clear output signal
        function clearOutputSignal(obj,var1)
            if strcmp(var1,'Testing')
                obj.e_testing = 0;
                obj.y_testing = 0;
            elseif strcmp(var1,'Training')
                obj.e_training = 0;
                obj.y_training = 0;
            end
        end
        
    end
    
end

