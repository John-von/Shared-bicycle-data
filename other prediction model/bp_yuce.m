clc; clear; close all;

data = xlsread('test.xls','sheet3','a1:a128')';

for i = 1:120
    y(i,1) = data(i+8);
    for j = 1:8
        x(i,j) = data(i+j-1);
    end
end

% Read data
input = x;
output = y;

% Training set and test set
input_train = input(1:119,:)';
output_train = output(1:119,:)';
input_test = input(120:end,:)';
output_test = output(120:end)';

% Data normalization
[inputn, inputs] = mapminmax(input_train, 0, 1); % Normalize to (0,1) range
[outputn, outputs] = mapminmax(output_train);
inputn_test = mapminmax('apply', input_test, inputs); % 'inputs' stores the mapping info

% Build BP neural network
net = newff(inputn, outputn, 8);

% Network parameters
net.trainparam.epochs = 10000; % Number of training iterations
net.trainparam.lr = 0.0001;    % Learning rate
net.dividefcn = '';            % Disable data division

% Train the BP neural network
net = train(net, inputn, outputn);

% Test the BP neural network
an = sim(net, inputn_test); % Simulate using the trained model
test_simu = mapminmax('reverse', an, outputs); % Reverse normalization of prediction
error2 = test_simu - output_test; % Error between predicted and actual values
