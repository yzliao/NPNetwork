function Training(obj,varargin)
    sigmoid_func = @(x) -1+2./(1+exp(-x));
    
    ifCheck         = false;
    ifPrintMsg      = false;
    
    if nargin == 1 % default
        ifLinear        = false;
        ifHiddenLayer   = true;
        ifCheck         = false;
        ifPrintMsg      = false;
    else
        for i = 1:nargin-1
            var1 = varargin{i};
            switch var1,
                case 'Linear',
                    ifLinear = true;
                    display('Linear');
                case 'Nonlinear',
                    ifLinear = false;
                    display('Nonlinear');
                case 'Hidden Layer',
                    ifHiddenLayer = true;
                    display('Hidden Layer');
                case 'No Hidden Layer',
                    ifHiddenLayer = false;
                    display('No Hidden Layer');
                case 'Check Rank',
                    ifCheck = true;
                    display('Check Rank');
                case 'Print Message',
                    ifPrintMsg = true;
                    display('Print Message');
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
%     xTrainingMtx = zeros(L,N+L);
%    
%     for i = L:N+L,
%         xtdl = x_training(i:-1:i-L+1);
%         xTrainingMtx(:,i) = xtdl;
%     end

    [p,q] = size(x_training);
    if p == L,
        xTrainingMtx = x_training;
    elseif q == L,
        xTrainingMtx = x_training';
    else
        display('Incorrect training input size');
        return;
    end
      
    
    % fixed layer
    if ifHiddenLayer,
        fixedWeightVec = obj.getFixedWeights();
        xtdl_sig = xTrainingMtx;
        for i = 1:obj.NumOfHiddenLayer,
            fixed_weights = fixedWeightVec{i};
            xtdl = fixed_weights*xtdl_sig;
            xtdl_sig = sigmoid_func(xtdl);
        end
        
        
        % test purpose
%         xx = xtdl(:,L:N+L);
%         p = length(find(abs(xx(:)>1)))/length(xx(:));
%         display(['Largest:',num2str(max(abs(xx(:))))]);
%         display(['Num distribution',num2str(p)]);
        
        fixedLayerMtx = xtdl_sig;
    else
        fixedLayerMtx = xTrainingMtx;
    end
    
    if ifPrintMsg,
        display('Completed Fixed layer computation');
    end
    
    if ifCheck,
        display(['Rank of fixed layer output ',num2str(rank(fixedLayerMtx))]);
        return;
    end
    
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
    
    if ifPrintMsg,
        display('Completed step size adjustment');
    end
    
    % adaptive layer
    elms = zeros(N+L,1);
    ylms = zeros(N+L,1);
    
    if ifHiddenLayer,
        adaptive_weights = obj.getAdaptiveWeights();    % from last hidden layer
    else
        %adaptive_weights = -0.1+0.2*rand(L,1);                  % from tapped delay line
        adaptive_weights = zeros(L,1);
    end
    

    for n = 1:obj.TrainingIter,
        for i = L+1:N+L,
        %for i = 1:N %%%
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
    
    if ifPrintMsg,
        display('Completed adaptive layer computation');
    end
    
    obj.setAdaptiveWeights(adaptive_weights);
    obj.y_training = ylms;
    obj.e_training = elms;
    
    if ifPrintMsg,
        display('Completed results saving');
    end
end
            
    
    
    
    
        
                