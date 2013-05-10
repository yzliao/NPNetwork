function [elms,ylms] = BP_Testing ...
    (obj,hidden_weights_cell,output_weights,var1)

    sigmoid_func = @(x) -1 + 2./(1+exp(-x));
    switch var1,
        case 'Linear',
            ifLinear = true;
            display('Linear');
        case 'Nonlinear',
            ifLinear = false;
            display('Non Linear');
        otherwise,
            ifLinear = false;
            display('Non Linear');
    end
    
    % get parameters from object properties
    L = obj.L;
    x_testing = obj.x_testing;
    d_testing = obj.d_testing;
    
    [p,~] = size(x_testing);
    if p == L,
        xTestingMtx = x_testing;
    elseif q == L,
        xTestingMtx = x_testing';
    else
        display('Incorrect testing input size');
    end
    [~,N] = size(xTestingMtx);
    N = N - L;  % correct size
    
    % hidden layer
    xtdl = xTestingMtx;
    for layer = 1:length(hidden_weights_cell),
        weights = hidden_weights_cell{layer};
        xtdl = weights*xtdl;
        xtdl = sigmoid_func(xtdl);
    end
    
    % output layer
    hiddenLayerMtx = xtdl;
    elms = zeros(N+L,1);
   % ylms = zeros(N+L,1);
    
    layer_output = output_weights*hiddenLayerMtx;
    
    if ifLinear,
        ylms = layer_output';
    else
        ylms = sigmoid_func(layer_output');
    end
    
    elms(L+1:end) = d_testing(L+1:end) - ylms(L+1:end);
    
end