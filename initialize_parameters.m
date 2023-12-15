function parameters = initialize_parameters(layer_dims)
    % Initializes the weights using He initialization and biases for a 3 layer
    % neural network
    %
    % Arguments:
    % layer_dims -- a vector containing the number of neurons in each layer
    %
    % Returns:
    % parameters -- a struct containing weights and biases for each layer

    rng(3); % Setting a random seed for comparability
    L = length(layer_dims); % Number of layers in the network
    fprintf('Number of layers (including input): %d\n', L);

    for l = 1:L-1
        parameters.(['W', num2str(l)]) = randn(layer_dims(l+1), layer_dims(l)) * sqrt(2. / layer_dims(l));
        parameters.(['b', num2str(l)]) = zeros(layer_dims(l+1), 1);
        
        % Debug prints
        fprintf('Layer %d initialized with dimensions: ', l);
        fprintf('W%d: %dx%d, ', l, size(parameters.(['W', num2str(l)]), 1), size(parameters.(['W', num2str(l)]), 2));
        fprintf('b%d: %dx%d\n', l, size(parameters.(['b', num2str(l)]), 1), size(parameters.(['b', num2str(l)]), 2));
    end

end
