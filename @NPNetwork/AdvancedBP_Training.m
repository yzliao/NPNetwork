function [MSE_vec,MSEPer_vec,output_weights_history] = ...
    AdvancedBP_Training(obj,TestingInputMtx,sys_fun,mu,output_scale,ifcont)
    
    if nargin < 6,
        ifcont = false;
    end

    sigmoid_func = @(x) -1 + 2./(1+exp(-x));
    
    % get parameters from object properties
    L = obj.L;
    N = obj.N;
    M_vec = obj.M_vec;
    x_training = obj.x_training;
    d_training = obj.d_training;
    
    if mu == 0,
        mu = obj.mu;
    end
    
    [p,q] = size(x_training);
    if p == L,
        xTrainingMtx = x_training;
    elseif q == L,
        xTrainingMtx = x_training';
    else
        display('Incorrect training input size');
        return;
    end
    
    
    [p,q] = size(TestingInputMtx);
    if p == N,
        
    elseif q == N,
        TestingInputMtx = TestingInputMtx';
    else
        display('Incorrect testing input size');
        return;
    end
    
    
    
    
    if ifcont,
        [hidden_weights_cell,output_weights] = obj.getBPWeights();
    else

        hidden_weights_cell = obj.getFixedWeights();

        %%%%
        %output_weights = zeros(1,obj.M_vec(end));
        output_weights = -0.05+0.1*rand(1,obj.M_vec(end));
        %%%%
    end
    
    
    output_weights_history = zeros(N,M_vec(end));
    MSE_vec = zeros(N,1);
    MSEPer_vec = zeros(N,1);
    
    idx = 1;
    tic
    
    % BP training algorithm
    for i = L+1:N+L,

        layer_input = xTrainingMtx(:,i);
        layer_output_cell = cell(obj.NumOfHiddenLayer,1);
        hidden_weights_new = cell(obj.NumOfHiddenLayer,1);
        delta_cell = cell(obj.NumOfHiddenLayer,1);

        % input from plant
        xtdl = layer_input;

        % feedforward
        % hidden layer
        for layer = 1:obj.NumOfHiddenLayer,
            layer_weights = hidden_weights_cell{layer};
            layer_output = layer_weights*layer_input;
            layer_output_sgm = sigmoid_func(layer_output);
            layer_output_cell{layer} = layer_output_sgm;

            layer_input = layer_output_sgm;
        end

        % output layer
        output_layer_output = output_weights*layer_input;
        

        % compute output layer error -- linear error

        delta = d_training(i) - output_layer_output;
        ylms(i) = output_layer_output;
        elms(i) = delta;

        % update output layer weights -- linear 
        output_weights_new = output_weights + mu*delta*layer_input';

        % feedback 
        % last hidden layer

        % compute last hidden layer error -- non linear
        epsilon = delta*output_weights;
        sigmoid_dot = 0.5*(1-layer_output_cell{end}.^2);
        delta = epsilon.*sigmoid_dot';
        delta_cell{end} = delta;

        % update last hidden layer weights -- non linear
        layer_weights = hidden_weights_cell{end};
        if obj.NumOfHiddenLayer == 1, % only one hidden layer
            layer_input = xtdl;
        else
            layer_input = layer_output_cell{end-1};
        end
        xx = delta'*layer_input';
        layer_weights = layer_weights + 2*mu*xx;
        
        
        
        hidden_weights_new{end} = layer_weights;

        % n-1th hidden layer to 1st hidden layer
        for layer = obj.NumOfHiddenLayer-1:-1:1,
            next_layer_weights = hidden_weights_cell{layer+1};
            if layer == 1,
                layer_input = xtdl;
            else
                layer_input = layer_output_cell{layer-1};
            end

            layer_output = layer_output_cell{layer};

            % compute hidden layer error -- non linear
            epsilon = delta_cell{layer+1}*next_layer_weights;
            sigmoid_dot = 0.5*(1-layer_output.^2);
            delta = epsilon.*sigmoid_dot';
            delta_cell{layer} = delta;


            % update hidden layer weights -- non linear
            layer_weights = hidden_weights_cell{layer};
            xx = delta'*layer_input';
            layer_weights = layer_weights + 2*mu*xx;
            
            hidden_weights_new{layer} = layer_weights;
        end

        hidden_weights_cell = hidden_weights_new;
        output_weights = output_weights_new;
        
        % Test Part
        testing_input = TestingInputMtx(idx,:);
        testing_output = sys_fun(testing_input)/output_scale;
        testing_input_mtx = streaming2mtx(testing_input,L,length(testing_input)-L,L);
        obj.setTesting(testing_input_mtx,testing_output');
        [test_error,~] = obj.BP_Testing(hidden_weights_cell,output_weights,'Linear');
        MSE_vec(idx) = mean(test_error(L+1:end).^2);
        MSEPer_vec(idx) = MSE_vec(idx)/mean(testing_output(L+1:end).^2);
        
        output_weights_history(idx,:) = output_weights;
        
        [idx toc]
        idx = idx + 1;


    end
    obj.y_BPtraining = ylms;
    obj.e_BPtraining = elms;
    obj.setBPWeights(hidden_weights_cell,output_weights);
end
    
    
    
    
    