function Training(obj,varargin)
    sigC = @(x) -1+2./(1+exp(-x));
    
    if nargin < 1 % default
        ifLinear        = true;
        ifHiddenLayer   = true;
    else
        for i = 1:nargin
            var1 = varargin{i};
            switch var1,
                case 'Linear',
                    ifLinear = true;
                case 'Nonlinear',
                    ifLinear = false;
                case 'Hidden Layer',
                    ifHiddenLayer = true;
                case 'Non Hidden Layer',
                    ifHiddenLayer = false;
                otherwise,
                    display(['Argument ',var1,' is not correct']);
            end
        end
    end
    
    % display argument
    display(['We are training', ifLinear, ' system with', ifHiddenLayer]);
    
    % convert streaming to matrix
    xTrainingMtx = zeros(L,N);
    for i = L:N+L-1,
        xtdl = x_training(i:-1:i-L+1);
        xTrainingMtx(:,i-L+1) = xtdl;
    end
    
    % fixed layer
    if ifHiddenLayer,
        fixedWeightVec = obj.getFixedWeights();
        xtdl = xTrainingMtx;
        
        for i = 1:obj.NumOfHiddenLayer,
            fixed_weights = fixedWeightVec{i};
            xtdl = fixed_weights*xtdl;
            xtdl = sigC(xtdl);
        end
        
        fixedLayerMtx = xtdl;
    else
        fixedLayerMtx = xTrainingMtx;
    end
    
    % calculate the power
    if ~obj.setmu
        RMatrix = xTrainingMtx * xTrainingMtx';
        RMatrix = RMatrix./N;
        trR = trace(RMatrix);
        mu = obj.misadj/trR;
        obj.setStepSize('step size',mu);
    else
        mu = obj.mu;
    end
    
    % adaptive layer
    elms = zeros(N,1);
    ylms = zeros(N,1);
    
    if ifHiddenLayer,
        adaptive_weights = obj.getAdaptiveWeights();
    else
        adaptive_weights = zeros(1,L);
    end
    
    d_training = obj.d_training;
    for n = 1:obj.TrainingIter,
        for i = 1:N,
            xtdl = xTrainingMtx(:,i);
            s = adaptive_weights*xtdl;
            if ifLinear,
                y = s;
            else
                y = sigC(s);
            end
            
            err = d_training(i+L)-y;
            
            ylms(i) = y;
            elms(i) = err;
            
            adaptive_weights = adaptive_weights + mu*err*xtdl';
        end
    end
    
    obj.setAdaptiveWeights(adaptive_weights);
    obj.y_training = ylms;
    obj.e_training = elms;
end
            
    
    
    
    
        
                