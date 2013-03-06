function Training(obj,varargin)
    sigmoid_func = @(x) -1+2./(1+exp(-x));
    
    if nargin == 1 % default
        ifLinear        = false;
        ifHiddenLayer   = true;
    else
        for i = 1:nargin-1
            var1 = varargin{i};
            switch var1,
                case 'Linear',
                    ifLinear = true;
                case 'Nonlinear',
                    ifLinear = false;
                case 'Hidden Layer',
                    ifHiddenLayer = true;
                case 'No Hidden Layer',
                    ifHiddenLayer = false;
                otherwise,
                    display(['Argument ',var1,' is not correct']);
            end
        end
    end
    
    % display argument
    %display(['We are training', ifLinear, ' system with', ifHiddenLayer]);
    
    % get parameters from object properties
    L = obj.L;
    N = obj.N;
    x_training = obj.x_training;
    d_training = obj.d_training;
    
    % convert streaming to matrix
    xTrainingMtx = zeros(L,N+L);
    for i = L:N+L,
        xtdl = x_training(i:-1:i-L+1);
        xTrainingMtx(:,i) = xtdl;
    end
    
    % fixed layer
    if ifHiddenLayer,
        fixedWeightVec = obj.getFixedWeights();
        xtdl = xTrainingMtx;
        
        for i = 1:obj.NumOfHiddenLayer,
            fixed_weights = fixedWeightVec{i};
            xtdl = fixed_weights*xtdl;
            xtdl = sigmoid_func(xtdl);
        end
        
        fixedLayerMtx = xtdl;
    else
        fixedLayerMtx = xTrainingMtx;
    end
    
    % calculate the power
    if ~obj.ifsetmu
        RMatrix = fixedLayerMtx* fixedLayerMtx';
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
    
    if ifHiddenLayer,
        adaptive_weights = obj.getAdaptiveWeights();    % from last hidden layer
    else
        adaptive_weights = zeros(L,1);                  % from tapped delay line
    end
    

    for n = 1:obj.TrainingIter,
        for i = L:N+L,
            xtdl = fixedLayerMtx(:,i);
            s = adaptive_weights'*xtdl;
            if ifLinear,
                filter_output = s;
            else
                filter_output = sigmoid_func(s);
            end
            
            filter_error = d_training(i)-filter_output;
            
            ylms(i) = filter_output;
            elms(i) = filter_error;
            
            if ifLinear,
                % theta_j = theta_j + mu*(y^(i) - theta_j'*x^(i))*x_j^(i)
                adaptive_weights = adaptive_weights + mu*filter_error*xtdl;
            else
                sigmoid_dot = 0.5*(1-filter_output.^2);
                % theta = theta + 2*mu*sigmoid_dot*error*xtdl
                adaptive_weights = adaptive_weights + 2*mu*sigmoid_dot*...
                    filter_error*xtdl;
            end
     
        end
    end
    
    obj.setAdaptiveWeights(adaptive_weights);
    obj.y_training = ylms;
    obj.e_training = elms;
end
            
    
    
    
    
        
                