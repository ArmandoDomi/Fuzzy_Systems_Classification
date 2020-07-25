format compact
clear
clc

%% Load data - Split data
data=readtable('../Datasets/epileptic_seizure_data.csv');
data=removevars(data,{'Var1'});
data= table2array(data);

preproc=1;
[trnData,chkData,tstData]=split_scale(data,preproc);
Perf=zeros(1,4);

%% Evaluation function
Rsq = @(ypred,y) 1-sum((ypred-y).^2)/sum((y-mean(y)).^2);

% Keep only the number of features we want and not all of them
% Specify their order and later use the ranks array
[ranks, weights] = relieff(data(:,1:end-1), data(:, end), 100);

%% FINAL TSK MODEL
fprintf('\n *** TSK Model with 15 features and radii 0.3 - Substractive Clustering\n');

f = 15;
radii = 0.3;

training_data_x = trnData(:,ranks(1:f));
training_data_y = trnData(:,end);

validation_data_x = chkData(:,ranks(1:f));
validation_data_y = chkData(:,end);

test_data_x = tstData(:,ranks(1:f));
test_data_y = tstData(:,end);%% TRAIN TSK MODEL

%% MODEL WITH 15 FEATURES AND 4 RULES

% Generate the FIS
fprintf('\n *** Generating the FIS\n');

% As input data I give the train_id's that came up with the
% partitioning and only the most important features
% As output data is just the last column of the test_data that
% are left
model = genfis2(training_data_x, training_data_y, radii);
rules = length(model.rule);
% Plot some input membership functions
figure;
plotMFs(model,size(training_data_x,2)-9);
suptitle('Final TSK model : some membership functions before training');
xlabel('x');
ylabel('Degree of membership');
saveas(gcf, 'Final_TSK_model/some_mf_before_training.png');

% Tune the fis
fprintf('\n *** Tuning the FIS\n');

% Set some options
% The fis structure already exists
% set the validation data to avoid overfitting
anfis_opt = anfisOptions('InitialFIS', model, 'EpochNumber', 150, 'DisplayANFISInformation', 0, 'DisplayErrorValues', 0, 'DisplayStepSize', 0, 'DisplayFinalResults', 0, 'ValidationData', [validation_data_x validation_data_y]);

[trnFis,trnError,~,valFis,valError] = anfis([training_data_x training_data_y], anfis_opt);

% Evaluate the fis
fprintf('\n *** Evaluating the FIS\n');

% No need to specify specific options for this, keep the defaults
Y = evalfis(test_data_x, valFis);
Y=round(Y);
diff=tstData(:,end)-Y;
Acc=(length(diff)-nnz(diff))/length(Y)*100;
%% METRICS
%% Error Matrix
error_matrix = confusionmat(tstData(:,end), Y);
pa = zeros(1, 2);
ua = zeros(1, 2);
%% confusion matrix
figure;
cm = confusionchart(tstData(:,end),Y);

error = Y - test_data_y;

%% Producer’s accuracy – User’s accuracy
N = length(tstData);
for i = 1 : 2
    pa(i) = error_matrix(i, i) / sum(error_matrix(:, i));
    ua(i) = error_matrix(i, i) / sum(error_matrix(i, :));
end

%% Overall accuracy
overall_acc = 0;
for i = 1 : 2
    overall_acc = overall_acc + error_matrix(i, i);
end
overall_acc = overall_acc / N;

%% k
p1 = sum(error_matrix(1, :)) * sum(error_matrix(:, 1)) / N ^ 2;
p2 = sum(error_matrix(2, :)) * sum(error_matrix(:, 2)) / N ^ 2;
pe = p1 + p2;
k = (overall_acc - pe) / (1 - pe);

% Plot the metrics
figure;
plot(1:length(test_data_y), test_data_y, '*r', 1:length(test_data_y), Y, '.b');
title('Output');
legend('Reference Outputs', 'Model Outputs');
saveas(gcf, 'Final_TSK_model/output.png')

figure;
plot(error);
title('Prediction Errors');
saveas(gcf, 'Final_TSK_model/error.png')

figure;
plot(1:length(trnError), trnError, 1:length(valError), valError);
title('Learning Curve');
legend('Traning Set', 'Check Set');
saveas(gcf, 'Final_TSK_model/learningcurves.png')

% Plot the input membership functions after training
figure;
plotMFs(valFis,size(training_data_x,2)-9);
suptitle('Final TSK model : some membership functions after training');
xlabel('x');
ylabel('Degree of membership');
saveas(gcf, 'Final_TSK_model/mf_after_training.png');









