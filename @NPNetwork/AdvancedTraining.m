function [MSE_vec,MSEPer_vec,weights_history] = ...
    AdvancedTraining(obj,TestingInputMtx,sys_fun,output_scale,ifcont)

    if nargin < 5,
        ifcont = false;
    end
    
    % calculating the MSE by testing at each training cycle
    sigmoid_func = @(x) -1+2./(1+exp(-x));
    
    % get parameters from object properties
    L = obj.L;
    N = obj.N;
    M_vec = obj.M_vec;
    x_training = obj.x_training;
    d_training = obj.d_training;
    
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
    
    
    
    fixedWeightVec = obj.getFixedWeights();
    xtdl_sig = xTrainingMtx;
    for i = 1:obj.NumOfHiddenLayer,
        fixed_weights = fixedWeightVec{i};
        xtdl = fixed_weights*xtdl_sig;
        xtdl_sig = sigmoid_func(xtdl);
    end
    
    fixedLayerMtx = xtdl_sig;
    
    % calculate the power
    if ~obj.ifsetmu
        RMatrix = fixedLayerMtx(:,L:N+L)* fixedLayerMtx(:,L:N+L)';
        RMatrix = RMatrix./N;
        trR = trace(RMatrix);
        mu = obj.misadj/trR;
        obj.setStepSize('step size',mu);
    else
        mu = obj.mu;
    end
    
    % adaptive layer
    elms = zeros(N+L,1);
    ylms = zeros(N+L,1);
    weights_history = zeros(N,M_vec(end));
    MSE_vec = zeros(N,1);
    MSEPer_vec = zeros(N,1);
    
    %%%%  
    if ifcont
        adaptive_weights = obj.getAdaptiveWeights();    % from last hidden layer
    else
        adaptive_weights = -0.05+0.1*rand(size(obj.getAdaptiveWeights()));
    end
    %%%%

    
    idx = 1;
    tic
    for i = L+1:N+L,
        xtdl = fixedLayerMtx(:,i);
        s = adaptive_weights'*xtdl;
        filter_output = s;
        
        filter_error = d_training(i) - filter_output;
        ylms(i) = filter_output;
        elms(i) = filter_error;
        
        adaptive_weights = adaptive_weights + mu*filter_error*xtdl;
        obj.setAdaptiveWeights(adaptive_weights);
        
        weights_history(idx,:) = adaptive_weights;
        
        % Test part
        testing_input = TestingInputMtx(idx,:);
        testing_output = sys_fun(testing_input)/output_scale;
        testing_input_mtx = streaming2mtx(testing_input,L,length(testing_input)-L,L);
        obj.setTesting(testing_input_mtx,testing_output');
        obj.Testing('Linear','Hidden Layer');
        [test_error,~] = obj.getOutputSignal('Testing');
        MSE_vec(idx) = mean(test_error(L+1:end).^2);
        MSEPer_vec(idx) = MSE_vec(idx)/mean(testing_output(L+1:end).^2);
        
        [idx toc]
        idx = idx + 1;
        
    end
    obj.y_training = ylms;
    obj.e_training = elms;
    obj.setAdaptiveWeights(adaptive_weights);
    
end
    
    
    