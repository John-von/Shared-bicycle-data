clc;
clear all
close all

%% Load data
% The output is a cell array where each element is a single time step. Reshape the data into a row vector.
data = xlsread('test.xls','sheet3','a1:a128')';

figure
plot(data)
xlabel("TIME")
ylabel("NUMBER")
title("Observer")

%% Partition the data into training and testing sets
% Use the first 94% of the sequence for training, and the last 6% for testing.
numTimeStepsTrain = floor(0.94 * numel(data));
dataTrain = data(1:numTimeStepsTrain + 1);
dataTest = data(numTimeStepsTrain:end);

%% Standardize the data
% To obtain better fitting and prevent divergence during training, standardize the training data to zero mean and unit variance.
% During prediction, use the same parameters to standardize the test data.
mu = mean(dataTrain);
sig = std(dataTrain);
dataTrainStandardized = (dataTrain - mu) / sig;

%% Prepare predictors and responses
% To predict future time steps in the sequence, specify the response as the training sequence shifted by one time step.
% That is, at each time step of the input sequence, the LSTM network learns to predict the value of the next time step.
% The predictors are the training sequence without the last time step.
XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);

%% Define LSTM network architecture
% Create an LSTM regression network. Specify 200 hidden units for the LSTM layer.
numFeatures = 1;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

% Specify training options
% Set the solver to 'adam' and train for 250 epochs.
% To prevent gradient explosion, set the gradient threshold to 1.
% Set initial learning rate to 0.005 and reduce it by a factor of 0.2 after 125 epochs.
options = trainingOptions('adam', ...
    'MaxEpochs', 250, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', 0.005, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 125, ...
    'LearnRateDropFactor', 0.2, ...
    'Verbose', 0, ...
    'Plots', 'training-progress');

%% Train LSTM network
% Train the LSTM network using trainNetwork with the specified options.
net = trainNetwork(XTrain, YTrain, layers, options);

%% Predict future time steps
% To predict multiple future time steps, use predictAndUpdateState one step at a time and update the network state at each step.
% Standardize the test data using the same parameters as the training data.
dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1);

% Initialize the network state by predicting on the training data.
% Then use the last training response YTrain(end) for the first prediction.
% Loop through remaining predictions, feeding the previous prediction into predictAndUpdateState.
% For large datasets, long sequences, or large networks, GPU inference is often faster. Otherwise, use CPU for single-step prediction.
% To force CPU prediction, set 'ExecutionEnvironment' to 'cpu' in predictAndUpdateState.
net = predictAndUpdateState(net, XTrain);
[net, YPred] = predictAndUpdateState(net, YTrain(end));

numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net, YPred(:,i)] = predictAndUpdateState(net, YPred(:,i-1), 'ExecutionEnvironment', 'cpu');
end

% Unstandardize the predictions using previously calculated parameters.
YPred = sig * YPred + mu;

% RMSE in training progress plot is based on standardized data. Calculate RMSE using unstandardized predictions.
YTest = dataTest(2:end);
rmse = sqrt(mean((YPred - YTest).^2));

% Plot the forecasted time series
figure
plot(dataTrain(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain + numTimeStepsTest);
plot(idx, [data(numTimeStepsTrain) YPred], '.-')
hold off
xlabel("TIME")
ylabel("NUMBER")
title("Forecast")
legend(["Observed", "Forecast"])

% Compare predictions to test data
figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred, '.-')
hold off
legend(["Observed", "Forecast"])
ylabel("NUMBER")
title("Forecast")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("TIME")
ylabel("Error")
title("RMSE = " + rmse)

%% Update network state using observed values
% If actual values between prediction steps are available, use them instead of predictions to update the network state.
% First, reset network state to prepare for new sequences using resetState.
% This prevents previous predictions from affecting new data.
net = resetState(net);
net = predictAndUpdateState(net, XTrain);

% Predict at each time step using previous actual observation.
% Set 'ExecutionEnvironment' to 'cpu' for prediction.
YPred = [];
numTimeStepsTest = numel(XTest);
for i = 1:numTimeStepsTest
    [net, YPred(:,i)] = predictAndUpdateState(net, XTest(:,i), 'ExecutionEnvironment', 'cpu');
end

% Unstandardize predictions using previously calculated parameters.
YPred = sig * YPred + mu;

% Calculate RMSE
rmse = sqrt(mean((YPred - YTest).^2));

% Compare predictions to test data
figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred, '.-')
hold off
legend(["Observed", "Predicted"])
ylabel("NUMBER")
title("Forecast with Updates")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("TIME")
ylabel("Error")
title("RMSE = " + rmse)
