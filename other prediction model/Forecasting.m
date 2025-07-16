%% Clear environment variables
clc;
clear all
close all
nntwarn off;

%% Load data
aa = xlsread('test.xls','sheet3','a1:a128');
lag = 4;         % Autoregressive order
iinput = aa;     % x is the original sequence (row vector)
n = length(iinput);

% Prepare input and output data
inputs = zeros(lag, n - lag); % Create a lag-row, (n-lag)-column zero matrix. For 2nd-order, two data points are used to predict the third.
for i = 1:n - lag
    inputs(:, i) = iinput(i:i + lag - 1)';
end
targets = aa(lag + 1:end);
targets = targets'; % Transpose the target matrix to align with the inputs matrix (depends on original data orientation)

% Create neural network
hiddenLayerSize = 10; % Number of hidden layer neurons
net = fitnet(hiddenLayerSize); % Create network with specified number of hidden layer neurons

% To avoid overfitting, set the ratio of training, validation, and testing data
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Train the network
[net, tr] = train(net, inputs, targets); % 'train' function: train(network, input data, target data)

%% Evaluate fitting performance with plots
yn = net(inputs);
errors = targets - yn;
figure, ploterrcorr(errors)                       % Plot autocorrelation of errors (20 lags)
figure, parcorr(errors)                           % Plot partial autocorrelation of errors
% [h, pValue, stat, cValue] = lbqtest(errors)     % Ljung-Box Q test (20 lags)
figure, plotresponse(con2seq(targets), con2seq(yn)) % Compare predicted trend with original trend
% figure, ploterrhist(errors)                     % Error histogram
% figure, plotperform(tr)                         % Performance curve

%% Predict future time steps
fn = 10;  % e.g., forecast next fn steps
f_in = iinput(n - lag + 1:end); % Use the last 'lag' points from original data as the initial input
% If data is a column vector, transpose may be needed: f_in = iinput(n-lag+1:end)';
f_out = zeros(1, fn); % Prediction output

% For multi-step prediction, feed the network output back into the input
for i = 1:fn
    f_out(i) = net(f_in);
    f_in = [f_in(2:end); f_out(i)];
end

% Plot the prediction results
figure, plot(1:n, iinput, 'b-s', n:n+fn, [iinput(end), f_out], 'r-d')
