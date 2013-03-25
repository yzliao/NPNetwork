function [elms,ylms,hidden_weights_cell,output_weights] = ...
    BP_Training(obj,varargin)

    sigmoid_func = @(x) -1 + 2./(1+exp(-x));
    
    if nargin == 1 % default
        ifLinear        = false;
    else 
        for i = 1:nargin-1,
            var1 = varargin{i};
            switch var1,
                case 'Linear',
                    ifLinear = true;
                    display('Linear');
                case 'Nonlinear',
                    ifLinear = false;
                    display('Nonlinear');
                otherwise,
                    display(['Argument ',var1,' is not correct']);
            end
        end
    end
    
    % get parameters from object properties
    L = obj.L;
    N = obj.N;
    x_training = obj.x_training;
    d_training = obj.d_training;
    mu = obj.mu;
    
    [p,q] = size(x_training);
    if p == L,
        xTrainingMtx = x_training;
    elseif q == L,
        xTrainingMtx = x_training';
    else
        display('Incorrect training input size');
        return;
    end
    
    % initalize hidden layer weights
    hidden_weights_cell = cell(length(obj.M_vec),1);
    
    % 1st hidden layer
    weights_vec = -1 + 2*rand(obj.M_vec(1),L);
    hidden_weights_cell{1} = weights_vec;
    
    % 2nd hidden layer to nth (last) hidden layer
    for layer = 2:length(obj.M_vec),
        weights_vec = -1 + 2*rand(obj.M_vec(layer),obj.M_vec(layer-1));
        hidden_weights_cell{layer} = weights_vec;
    end
    
    % output layer
    output_weights = -1+2*rand(1,obj.M_vec(end));

    % output vector
    ylms = zeros(N+L,1);
    elms = zeros(N+L,1);
    
    % BP training algorithm
    for i = L:N+L,
        layer_input = xTrainingMtx(:,i);
        layer_output_cell = cell(length(obj.M_vec),1);
        hidden_weights_new = cell(length(obj.M_vec),1);
        delta_cell = cell(length(obj.M_vec),1);

        % input from plant
        xtdl = layer_input;

        % feedforward
        % hidden layer
        for layer = 1:length(obj.M_vec),
            layer_weights = hidden_weights_cell{layer};
            layer_output = layer_weights*layer_input;
            layer_output_sgm = sigmoid_func(layer_output);
            layer_output_cell{layer} = layer_output_sgm;

            layer_input = layer_output_sgm;
        end

        % output layer
        output_layer_output = output_weights*layer_input;
        

        % compute output layer error -- linear error
        if ifLinear,
            delta = d_training(i) - output_layer_output;
            ylms(i) = output_layer_output;
        else
            delta = d_training(i) - sigmoid_func(output_layer_output);
            output_layer_output = sigmoid_func(output_layer_output);
            ylms(i) = output_layer_output;
        end
        elms(i) = delta;

        % update output layer weights -- linear 
        if ifLinear,
            output_weights_new = output_weights + mu*delta*layer_input';
        else
            output_weights_new = output_weights + 2*mu*delta'*layer_input';
        end

        % feedback 
        % last hidden layer

        % compute last hidden layer error -- non linear
        epsilon = delta*output_weights;
        sigmoid_dot = 0.5*(1-layer_output_cell{end}.^2);
        delta = epsilon.*sigmoid_dot';
        delta_cell{end} = delta;

        % update last hidden layer weights -- non linear
        layer_weights = hidden_weights_cell{end};
        if layer == 1,
            layer_input = xtdl;
        else
            layer_input = layer_output_cell{end-1};
        end
        xx = delta'*layer_input';
        layer_weights = layer_weights + 2*mu*xx;

        hidden_weights_new{end} = layer_weights;

        % n-1th hidden layer to 1st hidden layer
        for layer = length(obj.M_vec)-1:-1:1,
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


    end
    


end

