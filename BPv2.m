% BPv2.m

clear all
close all

sigmoid_func = @(x) -1 + 2./(1+exp(-x));



L = 100;             % tapped line
N = 1000;            % number of patterns
M_vec = [800 1000];      % number of neurons in hidden layer
mu = 0.01;

plant_input = rand(L,N);
%sys = [-3 -4 -5 3 7 9 1 2 2 2]';
sys = randn(L,1);
plant_output = sys'*plant_input/20;

% BP Network

% initalize hidden layer weights
hidden_weights_cell = cell(length(M_vec),1);

% 1st hidden layer
weights_vec = -1 + 2*rand(M_vec(1),L);
hidden_weights_cell{1} = weights_vec;

% 2nd hidden layer to nth (last) hidden layer
for layer = 2:length(M_vec),
    weights_vec = -1+2*rand(M_vec(layer),M_vec(layer-1));
    hidden_weights_cell{layer} = weights_vec;
end

% output layer
output_weights = -1+2*rand(1,M_vec(end));


% error vector
error_vector = zeros(N,1);

% BP training algorithm
for i = 1:N,
    layer_input = plant_input(:,i);
    layer_output_cell = cell(length(M_vec),1);
    hidden_weights_new = cell(length(M_vec),1);
    delta_cell = cell(length(M_vec),1);
    
    % input from plant
    xtdl = plant_input(:,i);
    
    % feedforward
    % hidden layer
    for layer = 1:length(M_vec),
        layer_weights = hidden_weights_cell{layer};
        layer_output = layer_weights*layer_input;
        layer_output_sgm = sigmoid_func(layer_output);
        layer_output_cell{layer} = layer_output_sgm;
        
        layer_input = layer_output_sgm;
    end
    
    % output layer
    output_layer_output = output_weights*layer_input;
    
    % compute output layer error -- linear error
    delta = plant_output(i) - output_layer_output;
    error_vector(i) = delta;
    
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
    layer_input = layer_output_cell{end-1};
    xx = delta'*layer_input';
    layer_weights = layer_weights + 2*mu*xx;
    
    hidden_weights_new{end} = layer_weights;
    
    % n-1th hidden layer to 1st hidden layer
    for layer = length(M_vec)-1:-1:1,
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
    
    