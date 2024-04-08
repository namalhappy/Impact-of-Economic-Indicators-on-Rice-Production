 function [rmse, rsquared, mse, mae, msle, mpe, medae, tunedParams, mdl] = gpr(trainFile, testFile, kernel)
    % Load train and test data from CSV files, skipping the first row (header)
    trainData = csvread(trainFile, 1, 0); % Assuming the format is [input_features, output_value] after skipping the header
    testData = csvread(testFile, 1, 0);   % Assuming the format is [input_features, output_value] after skipping the header

    % Extract input features (all but last column) and output values (last column) for training data
    Xtrain = trainData(:, 1:end-1); % Input features are all columns except the last
    ytrain = trainData(:, end);     % Output values are in the last column

    % Define the hyperparameter optimization options
    opt = struct('Optimizer', 'bayesopt', 'MaxObjectiveEvaluations', 30);

    % Create a Gaussian Process Regression model with the specified kernel and optimize hyperparameters
    gprMdl = fitrgp(Xtrain, ytrain, 'KernelFunction', kernel);

    % Get the tuned hyperparameters
    tunedParams = gprMdl.ModelParameters;
    
    mdl = gprMdl;
    % Predict on test data
    Xtest = testData(:, 1:end-1); % Input features for test data
    ytest = testData(:, end);     % Output values for test data
    ypred = predict(gprMdl, Xtest);
    
    data = [ypred, ytest];
    outfile =  strcat('Outputs/GPR_',kernel, '.csv');
    disp(outfile);
    csvwrite(outfile, data);

    % Evaluate the performance of the model
    rmse = sqrt(mean((ypred - ytest).^2)); % Root Mean Squared Error

    % Calculate R-squared
    R = corrcoef(ytest, ypred);
    rsquared = R(1, 2)^2;

    % Calculate MSE
    mse = mean((ypred - ytest).^2);

    % Calculate MAE
    mae = mean(abs(ypred - ytest));

    % Calculate MSLE
    msle = mean((log1p(ypred) - log1p(ytest)).^2);

    % Calculate MPE
    mpe = mean((ypred - ytest) ./ ytest) * 100;

    % Calculate MedAE
    medae = median(abs(ypred - ytest));
end
