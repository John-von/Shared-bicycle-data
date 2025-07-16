%% Initialization 
clear
close all
clc
format short

%% Read data
data = xlsread('statistical data.xlsx', 'Sheet1', 'L2:O33'); %% Use xlsread function to read the corresponding range of data from EXCEL  

% Input and output data
input = data(:,1:end-1);    % The first column to the second last column of data are feature indicators
output = data(:,end);  % The last column of data is the output indicator value

N = length(output);   % Total number of samples
testNum = 8;   % Set the number of test samples
trainNum = N - testNum;    % Calculate the number of training samples

%% Split into training set and test set
input_train = input(1:trainNum,:)';
output_train = output(1:trainNum)';
input_test = input(trainNum+1:trainNum+testNum,:)';
output_test = output(trainNum+1:trainNum+testNum)';

%% Data normalization
[inputn, inputps] = mapminmax(input_train, 0, 1);
[outputn, outputps] = mapminmax(output_train);
inputn_test = mapminmax('apply', input_test, inputps);

%% Get the number of input and output layer nodes
inputnum = size(input, 2);
outputnum = size(output, 2);
disp('/////////////////////////////////')
disp('Neural Network Structure...')
disp(['Number of nodes in the input layer: ', num2str(inputnum)])
disp(['Number of nodes in the output layer: ', num2str(outputnum)])
disp(' ')
disp('Determining the number of nodes in the hidden layer...')

% Determine the number of nodes in the hidden layer
% Using the empirical formula hiddennum = sqrt(m + n) + a, where m is the number of input layer nodes, 
% n is the number of output layer nodes, and a is generally an integer between 1 and 10
MSE = 1e+5; % Initialize the minimum error
transform_func = {'tansig', 'purelin'}; % Activation functions
train_func = 'trainlm';   % Training algorithm
for hiddennum = fix(sqrt(inputnum + outputnum)) + 1 : fix(sqrt(inputnum + outputnum)) + 10
    
    % Build the network
    net = newff(inputn, outputn, hiddennum, transform_func, train_func);
    % Network parameters
    net.trainParam.epochs = 1000;         % Number of training iterations
    net.trainParam.lr = 0.01;                   % Learning rate
    net.trainParam.goal = 0.000001;        % Minimum training error target
    % Network training
    net = train(net, inputn, outputn);
    an0 = sim(net, inputn);  % Simulation result
    mse0 = mse(outputn, an0);  % Mean squared error of simulation
    disp(['When the number of hidden layer nodes is ', num2str(hiddennum), ', the MSE of the training set is: ', num2str(mse0)])
    
    % Update the best number of hidden layer nodes
    if mse0 < MSE
        MSE = mse0;
        hiddennum_best = hiddennum;
    end
end
disp(['The best number of hidden layer nodes is: ', num2str(hiddennum_best), ', corresponding MSE: ', num2str(MSE)])

%% Build the BP neural network with the best hidden layer nodes
net = newff(inputn, outputn, hiddennum_best, transform_func, train_func);

% Network parameters
net.trainParam.epochs = 1000;         % Number of training iterations
net.trainParam.lr = 0.01;                   % Learning rate
net.trainParam.goal = 0.000001;        % Minimum training error target

%% Network training
net = train(net, inputn, outputn);

%% Network testing
an = sim(net, inputn_test); % Simulate using the trained model
test_simu = mapminmax('reverse', an, outputps); % Reverse the normalization of the predicted results

error = test_simu - output_test;      % Error between predicted and actual values

%% Compare actual values and predicted values error
figure
plot(output_test, 'bo-', 'linewidth', 1.2)
hold on
plot(test_simu, 'r*--', 'linewidth', 1.2)
legend('Actual Value', 'Predicted Value')
xlabel('Time'), ylabel('Vehicle Count')
set(gca, 'fontsize', 12)
%set(gca, 'XTicklabel', {'6.00~6.30', '6.30~7.00', '7.00~7.30', '7.30~8.00', '8.00~8.30', '8.30~9.00', '9.00~9.30', '9.30~10.00'})
set(gca, 'XTicklabel', {'6.30', '7.00', '7.30', '8.00', '8.30', '9.00', '9.30', '10.00'})
%%
figure
plot(error, 'ro-', 'linewidth', 1.2)
xlabel('Time'), ylabel('Prediction Deviation')
set(gca, 'XTicklabel', {'6.30', '7.00', '7.30', '8.00', '8.30', '9.00', '9.30', '10.00'})
set(gca, 'fontsize', 12)
ylim()
%%
% Calculate errors
[~, len] = size(output_test);
SSE1 = sum(error.^2);
MAE1 = sum(abs(error)) / len;
MSE1 = error * error' / len;
RMSE1 = MSE1^(1/2);
MAPE1 = mean(abs(error ./ output_test));
r = corrcoef(output_test, test_simu);    % corrcoef calculates the correlation coefficient matrix, including self-correlation and cross-correlation
R1 = r(1, 2);    

disp(' ')
disp('/////////////////////////////////')
disp('Prediction Error Analysis...')
disp(['Sum of squared errors (SSE):            ', num2str(SSE1)])
disp(['Mean absolute error (MAE):      ', num2str(MAE1)])
disp(['Mean squared error (MSE):              ', num2str(MSE1)])
disp(['Root mean squared error (RMSE):        ', num2str(RMSE1)])
disp(['Mean absolute percentage error (MAPE): ', num2str(MAPE1*100), '%'])
disp(['Correlation coefficient (R):                     ', num2str(R1)])

% Print results
disp(' ')
disp('/////////////////////////////////')
disp('Print test set prediction results...')
disp(['   ID          Actual Value         Predicted Value         Error'])
for i = 1:len
    disp([i, output_test(i), test_simu(i), error(i)])
end
