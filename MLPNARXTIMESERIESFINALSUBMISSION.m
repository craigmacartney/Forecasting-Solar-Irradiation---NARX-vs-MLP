%% Comparison of MLP Backprop and NARX on Solar Irradiation Forecasting 

%% Initialisation
clear ; close all; clc
format compact; %Suppress the display of blank lines
rng default;    %Ensure repeatable results

%% Load data
% 1) SUNY Glo (Wh/m^2)
% 2) Zenith (deg)
% 3) Azimuth (deg)
% 4) day
% 5) hour

M = csvread('solarFINAL2.csv');

disp('View first 5 rows of original data');
disp(M(1:5,1:end)); % view first 5 rows
disp(' ');

% confirm there are no missing values
nMissing = sum(sum(ismissing(M)));
fprintf('Number of missing values = %i\n', nMissing);
fprintf('\n');

%% Initial data analysis and prepartion for MLP/NARX time series forecasting 
%Visualise time series
Solar_viz
%Display descriptive stats from Paper in code (see Solar_viz.m)
%Please see SumStats.m and CorrTable.m to verify figures in Tables 1 and 2
%from Paper

%Pre-Processing
    %NORMALIZE DATA
  for i=1:5 
 M(:,i) = (M(:,i)-min(M(:,i)))/(max(M(:,i))-min(M(:,i)));
  end

disp('View first 5 rows of standardised table')
disp(M(1:5,1:end)); % view first 5 rows of standardised table
disp(' ');

%CREATE TRAIN/TEST SPLITS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  N.B. since our data is time series we do not select
%  a random sample as our test set, but instead use the latest time
%  sequence. This is because we need to keep the time dependencies in our
%  test set - and taking the latest sequence allows us to more effectively
%  undertake cross validation (as we will need to sample test sequences
%  as part of our crossvalidation) 

%Create final training & test set
% Split data into training set (80%) and test set holdout (20%) 
solarirrad_finaltrain = M(1:3836,1:1);
X_input_finaltrain = M(1:3836,2:5);

solarirrad_finaltest = M(3837:end,1:1);
X_input_finaltest = M(3837:end,2:5);

%Maintain full dataset in its entirety
solarirrad_fulldata = M(:,1:1);
X_input_fulldata = M(:,2:5);

% The following Cross-Validation takes 6-7 hours to run, so we have
% commented it out and provided the following file:
    % FINALCROSSVALRESULTS.mat
        % CV Results for both MLP and NARX
        
 % We have also included our final train/test results. These can be run at
 % leisure:
    % FINALTESTTRAINPERFORMANCEMLP.mat
    % FINALTESTPERFORMANCENARX.mat
    % MLPperformance_ResultsInPaper.fig
    % NARXperformance_ResultsInPaper.fig
    

% 
% %CREATE TIME SERIES K-FOLD ROLLING WINDOWS FOR CROSS VALIDATION 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %Create ten folds each with test dataset sequence of 769 long (20% of the
% %remaining training data)
% 
%  X_input_crosstrainwindows = {};
%  X_input_crosstestwindows = {};
%  solarirrad_crosstrainwindows= {};
%  solarirrad_crosstestwindows={};
% 
% 
% for u=1:10
%     % create random number between 1 and 3067  
%     Randno = rand;
%     Splitno = round(Randno*3067);
%     
%     % fold test sequence of length 769
%     solarirrad_crosstestwindow = solarirrad_finaltrain(Splitno:Splitno+768,:);
%     X_input_crosstestwindow = X_input_finaltrain(Splitno:Splitno+768,:);
% 
%     solarirrad_crosstrainwindow = solarirrad_finaltrain;
%     X_input_crosstrainwindow = X_input_finaltrain;
% 
%     % fold training set of length 3067 long (all the remaining training
%     % data minus the test sequence)
%     solarirrad_crosstrainwindow(Splitno:Splitno+768,:)=[];
%     X_input_crosstrainwindow(Splitno:Splitno+768,:)=[];
%     
%     %training and test data for cross validation partition
%     X_input_crosstrainwindows{u} =  X_input_crosstrainwindow;
%     solarirrad_crosstrainwindows{u} = solarirrad_crosstrainwindow;
%     X_input_crosstestwindows{u} = X_input_crosstestwindow;
%     solarirrad_crosstestwindows{u} = solarirrad_crosstestwindow;
% 
% end
% 
% %% MLP Backprop
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%BACKPROP/MLP CROSS VALIDATION %%%%%%%%%%
% 
% % Variable pre-allocations difficult for OverallperformanceMLP and
% % performanceMLP (same for NARX equivalents)
% 
% OverallperformanceMLP = {};
% s=0;
% 
% % LOOP THROUGH EACH TIME SERIES FOLD PARTITION
% for a = 1:10
%     
%     % Get training and test set for a particular fold
%     train_xinput_window = X_input_crosstrainwindows{a};
%     train_solarirrad_window = solarirrad_crosstrainwindows{a};
%     test_xinput_window = X_input_crosstestwindows{a};
%     test_solarirrad_window = solarirrad_crosstestwindows{a};
% 
%     % construct lagged training input variables
%     train_solarirrad_no_lag = train_solarirrad_window(1:end-4,1:1);  
%     train_solarirrad_lag1 = train_solarirrad_window(2:end-3,1:1);
%     train_solarirrad_lag2 = train_solarirrad_window(3:end-2,1:1);
%     train_solarirrad_lag3 = train_solarirrad_window(4:end-1,1:1);
%     train_solarirrad_lag4 = train_solarirrad_window(5:end,1:1);
% 
%     train_xinput_no_lag = train_xinput_window(1:end-4,:);
%     train_xinput_lag1 = train_xinput_window(2:end-3,:);
%     train_xinput_lag2 = train_xinput_window(3:end-2,:);
%     train_xinput_lag3 = train_xinput_window(4:end-1,:);
%     train_xinput_lag4 = train_xinput_window(5:end,:);
% 
%     % construct lagged test input variables
%     test_solarirrad_no_lag = test_solarirrad_window(1:end-4,1:1);
%     test_solarirrad_lag1 = test_solarirrad_window(2:end-3,1:1);
%     test_solarirrad_lag2 = test_solarirrad_window(3:end-2,1:1);
%     test_solarirrad_lag3 = test_solarirrad_window(4:end-1,1:1);
%     test_solarirrad_lag4 = test_solarirrad_window(5:end,1:1);
% 
%     test_xinput_no_lag = test_xinput_window(1:end-4,:);
%     test_xinput_lag1 = test_xinput_window(2:end-3,:);
%     test_xinput_lag2 = test_xinput_window(3:end-2,:);
%     test_xinput_lag3 = test_xinput_window(4:end-1,:);
%     test_xinput_lag4 = test_xinput_window(5:end,:);
% 
%     % combine training lags to create comparable input variables to input delay/feedback delay combinations used in NARX
%     X1train = train_xinput_lag1;
%     X2train = [train_xinput_lag1 train_xinput_lag2];
%     X3train = [train_xinput_lag1 train_xinput_lag2 train_xinput_lag3];
%     X4train = [train_xinput_lag1 train_xinput_lag2 train_xinput_lag3 train_xinput_lag4];
% 
%     SOLAR1train = train_solarirrad_lag1;
%     SOLAR2train = [train_solarirrad_lag1 train_solarirrad_lag2];
%     SOLAR3train = [train_solarirrad_lag1 train_solarirrad_lag2 train_solarirrad_lag3];
%     SOLAR4train = [train_solarirrad_lag1 train_solarirrad_lag2 train_solarirrad_lag3 train_solarirrad_lag4];
% 
%     
%     % combine test lags to create comparable input variables to input delay/feedback delay combinations used in NARX    
%     X1test = test_xinput_lag1;
%     X2test = [test_xinput_lag1 test_xinput_lag2];
%     X3test = [test_xinput_lag1 test_xinput_lag2 test_xinput_lag3];
%     X4test = [test_xinput_lag1 test_xinput_lag2 test_xinput_lag3 test_xinput_lag4];
% 
%     SOLAR1test = test_solarirrad_lag1;
%     SOLAR2test = [test_solarirrad_lag1 test_solarirrad_lag2];
%     SOLAR3test = [test_solarirrad_lag1 test_solarirrad_lag2 test_solarirrad_lag3];
%     SOLAR4test = [test_solarirrad_lag1 test_solarirrad_lag2 test_solarirrad_lag3 test_solarirrad_lag4];
% 
%     % create sliding window size combinations to loop through in cross
%     % validation
%     Xinputstrain = {X2train,X3train,X4train};
%     Yinputstrain = {SOLAR2train,SOLAR3train,SOLAR4train};
%     Xinputstest = {X2test,X3test,X4test};
%     Yinputstest = {SOLAR2test,SOLAR3test,SOLAR4test};
% 
%     
%     % create sliding window size combinations to loop through in cross
%     % validation
%     HiddenLayerSizeArray = [20,30,40];
%     MomentumArray = [0.3,1,3];
%     LearningrateArray = [0.003,0.01,0.03];
% 
%     q=0;
%     % PERFORM HYPERPARAMETER GRID SEARCH WITHIN FOLD
%     for b = 1:3
%         for c=1:3
%             for d=1:3
%                 for e=1:3
%                     for f = 1:3
%                         % combine input training variables 
%                         MLP_X_Inputtrain = Xinputstrain{b};
%                         MLP_Y_Inputtrain = Yinputstrain{c};
%                         MLP_Input_train = [MLP_X_Inputtrain, MLP_Y_Inputtrain];
%                         % combine input test variables
%                         MLP_X_Inputtest = Xinputstest{b};
%                         MLP_Y_Inputtest = Yinputstest{c};
%                         MLP_Input_test = [MLP_X_Inputtest, MLP_Y_Inputtest];
%                        
%                         % configure MLP training with hyperparameter values
%                         trainFcn = 'trainlm';
%                         hiddenLayerSize = HiddenLayerSizeArray(d);
%                         net = fitnet(hiddenLayerSize,trainFcn);
%                         
%                         %set indices for training and crossvalidation
%                         net.divideFcn = 'divideind';
%                         net.divideParam.trainind = 1:3062;
%                         net.divideParam.valind = 3063:3836;
%                         %set learning rate and momentum
%                         net.trainParam.lr = LearningrateArray(e);
%                         net.trainParam.mc = MomentumArray(f);
%                         % set early stopping criteria
%                         net.trainParam.epochs = 30;                
%                         net.trainParam.min_grad = 1e-5;
%                         net.trainParam.max_fail = 10; 
%                         
%                         % combine training and test data for training in
%                         % MLP
%                         FULL_XInputdata = [MLP_Input_train; MLP_Input_test];
%                         FULL_Ylabeldata = [train_solarirrad_no_lag; test_solarirrad_no_lag];
% 
%                         %Train the network
%                         [net,tr] = train(net,FULL_XInputdata',FULL_Ylabeldata');
%                         % Test the Network
%                         y_MLPpredict = net(MLP_Input_test'); 
%                         q= q+1;
%                         
%                         % store HYPERPARAMETER VALUES FOR FOLD
%                         performanceMLP{q,1} = b; % or corresponding index in inputdelayarray in this case
%                         performanceMLP{q,2} = c; % or corresponding index in feedbackdelayarray in this case
%                         performanceMLP{q,3} = HiddenLayerSizeArray(d);
%                         performanceMLP{q,4} = LearningrateArray(e); 
%                         performanceMLP{q,5} = MomentumArray(e);
%                         % STORE TEST PERFORMANCE ACROSS ALL HYPER PARAMETER COMBINATIONS WITHIN A K-FOLD
%                         performanceMLP{q,6} = perform(net,test_solarirrad_no_lag,y_MLPpredict'); 
%                     end
%                 end
%             end
%         end
%     end
%     s= s+1;
%      % TEST MLP PERFORMANCE ACROSS ALL HYPER PARAMETER COMBINATIONS ACROSS ALL
%      % K-FOLDS
%     OverallperformanceMLP{s} = performanceMLP;
% 
% end
% 
% 
% AverageperformanceMLP = {};
% % AVERAGE MLP PERFORMANCE ACROSS ALL FOLDS
% for t=1:243
%    performanceFold1 = cell2mat(OverallperformanceMLP{1,1}(t,6));
%    performanceFold2 = cell2mat(OverallperformanceMLP{1,2}(t,6));
%    performanceFold3 = cell2mat(OverallperformanceMLP{1,3}(t,6));
%    performanceFold4 = cell2mat(OverallperformanceMLP{1,4}(t,6));
%    performanceFold5 = cell2mat(OverallperformanceMLP{1,5}(t,6));
%    performanceFold6 = cell2mat(OverallperformanceMLP{1,6}(t,6));
%    performanceFold7 = cell2mat(OverallperformanceMLP{1,7}(t,6));
%    performanceFold8 = cell2mat(OverallperformanceMLP{1,8}(t,6));
%    performanceFold9 = cell2mat(OverallperformanceMLP{1,9}(t,6));
%    performanceFold10 = cell2mat(OverallperformanceMLP{1,10}(t,6));
%    AverageperformanceMLP{t,1}= OverallperformanceMLP{1,1}(t,1); 
%    AverageperformanceMLP{t,2}= OverallperformanceMLP{1,1}(t,2);
%    AverageperformanceMLP{t,3}= OverallperformanceMLP{1,1}(t,3);
%    AverageperformanceMLP{t,4}= OverallperformanceMLP{1,1}(t,4);
%    AverageperformanceMLP{t,5}= OverallperformanceMLP{1,1}(t,5);
%    AverageperformanceMLP{t,6} = (performanceFold1 + performanceFold2 + performanceFold3+ performanceFold4 + performanceFold5 + performanceFold6 + performanceFold7 + performanceFold8 + performanceFold9 + performanceFold10)/10;
%  
% end
% 
% 
% %% NARX
% 
% %%%%%%%%%%%%%%%%%%%%% NARX%%%%%%%%%%%%%%%%%%%%%%%%%%
% OverallperformanceNARX = {};
% s=0;
% 
% % loop through to get data for a particular cross validation split
% for a =1:10
%      % Get training and test set for a particular fold
%     train_xinput_window = X_input_crosstrainwindows{a};
%     train_solarirrad_window = solarirrad_crosstrainwindows{a};
%     test_xinput_window = X_input_crosstestwindows{a};
%     test_solarirrad_window = solarirrad_crosstestwindows{a};
% 
%     fulldata_xinput_window = [X_input_crosstrainwindows{a}; X_input_crosstestwindows{a}];   
%     fulldata_solarirrad_window = [solarirrad_crosstrainwindows{a};solarirrad_crosstestwindows{a}];
% 
%     % create lagged input variables for training and testing
%     X_train = tonndata(train_xinput_window,false,false);
%     T_train = tonndata(train_solarirrad_window ,false,false);
%     X_test = tonndata(test_xinput_window,false,false);
%     T_test = tonndata(test_solarirrad_window,false,false);
%     X_FULLDATA = tonndata(fulldata_xinput_window,false,false);
%     T_FULLDATA = tonndata(fulldata_solarirrad_window,false,false);
% 
%     trainFcn = 'trainlm';  % Levenberg-Marquardt backpropagation.
% 
% 
%     q=0;
%     % Create a Nonlinear Autoregressive Network with External Input
%     % this creates 3^5 (243) entries for hyperparameter combinations
%     % this inludes two duplicates for each value as
%     % inputDelay,feedbackbackdelay [1:2],[1:2] is the same as feedbackdelay,inputdelay [1:2],[1:2] etc
%     % we resolve this by only keeping the first combination in our results
%     
%     %inputDelaysArray_NARX = {[1:2],[1:3],[1:4]}; %[] brackets not necessary
%     inputDelaysArray_NARX = {1:2,1:3,1:4};
%     %feedbackDelaysArray_NARX = {[1:2],[1:3],[1:4]}; %[] brackets not necessary
%     feedbackDelaysArray_NARX = {1:2,1:3,1:4};
%     
%     HiddenLayerSizeArray_NARX = [20,30,40];
%     MomentumArray_NARX = [0.3,1,3];
%     LearningrateArray_NARX = [0.003,0.01,0.03]; 
% 
%     for b = 1:3
%         for c=1:3
%             for d=1:3
%                 for e=1:3
%                     for f=1:3
% 
%                         %set hyperparameters for the split
%                         inputDelays = inputDelaysArray_NARX{b};
%                         feedbackDelays = feedbackDelaysArray_NARX{c};
%                         hiddenLayerSize = HiddenLayerSizeArray_NARX(d);
%                         net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'open',trainFcn);           
%                         
%                         
%                         % Prepare the Data for Training and Simulation
%                         % The function PREPARETS prepares timeseries data for a particular network,
%                         % shifting time by the minimum amount to fill input states and layer
%                         % states. Using PREPARETS allows you to keep your original time series data
%                         % unchanged, while easily customizing it for networks with differing
%                         % numbers of delays, with open loop or closed loop feedback modes.
%                         [x_full,xi_full,ai_full,t_full] = preparets(net,X_FULLDATA,{},T_FULLDATA); 
%                         [x_test,xi_test,ai_test,t_test] = preparets(net,X_test,{},T_test);
%                         [x_train,xi_train,ai_train,t_train] = preparets(net,X_train,{},T_train);
% 
%                         %adjust size of validation set depending on input
%                         %and feed back delay
%                         crossvalsplitstart = length(t_full) - length(t_test);
%                         crossvalsplitend = length(t_full); 
%                         
%                         %Setup Division of Data for Training, Validation, Testing  
%                         net.divideFcn = 'divideind';
%                         net.divideParam.trainind = 1:crossvalsplitstart-1;
%                         net.divideParam.valind = crossvalsplitstart:crossvalsplitend;
%                         
%                         % set learning rate and momentum hyperparameter
%                         % values
%                         net.trainParam.lr = LearningrateArray_NARX(e);
%                         net.trainParam.mc = MomentumArray_NARX(f);
%                         
%                         % set stopping criteria
%                         net.trainParam.epochs = 30;                
%                         net.trainParam.min_grad = 1e-5;
%                         net.trainParam.max_fail = 10; 
%                         
%                         % Train the Network
%                         [net,tr] = train(net,x_full,t_full);
%                         
%                         % Test the Network
%                         NARX_ypredict = net(x_test);
%                         q= q+1;
%                         performanceNARX{q} = perform(net,t_test,NARX_ypredict);    
%                     end
%                 end
%             end
%         end
%         
%     end
%     s=s+1;
%     OverallperformanceNARX{s} = {performanceNARX};
% end
%     
% AverageperformanceNARX = {};     
% %%% Calculate average performance in each fold for each parameter combination     
% % calculate average performance in each fold
% for t=1:243
%    performanceFold1NARX = OverallperformanceNARX{1,1}{1,1}{1,t};
%    performanceFold2NARX = OverallperformanceNARX{1,2}{1,1}{1,t};
%    performanceFold3NARX = OverallperformanceNARX{1,3}{1,1}{1,t};
%    performanceFold4NARX = OverallperformanceNARX{1,4}{1,1}{1,t};
%    performanceFold5NARX = OverallperformanceNARX{1,5}{1,1}{1,t};
%    performanceFold6NARX = OverallperformanceNARX{1,6}{1,1}{1,t};
%    performanceFold7NARX = OverallperformanceNARX{1,7}{1,1}{1,t};
%    performanceFold8NARX = OverallperformanceNARX{1,8}{1,1}{1,t};
%    performanceFold9NARX = OverallperformanceNARX{1,9}{1,1}{1,t};
%    performanceFold10NARX = OverallperformanceNARX{1,10}{1,1}{1,t};
%    % folds calculated in same order as MLP so can use "hyperparameter values" from NARX" 
%    AverageperformanceNARX{t,1}= OverallperformanceMLP{1,1}(t,1); 
%    AverageperformanceNARX{t,2}= OverallperformanceMLP{1,1}(t,2);
%    AverageperformanceNARX{t,3}= OverallperformanceMLP{1,1}(t,3);
%    AverageperformanceNARX{t,4}= OverallperformanceMLP{1,1}(t,4);
%    AverageperformanceNARX{t,5}= OverallperformanceMLP{1,1}(t,5);
%    AverageperformanceNARX{t,6} = (performanceFold1NARX + performanceFold2NARX + performanceFold3NARX+ performanceFold4NARX + performanceFold5NARX + performanceFold6NARX + performanceFold7NARX + performanceFold8NARX + performanceFold9NARX + performanceFold10NARX)/10;
%    
% end    
% 
% %% Consolidated CV
% % create final consolidated MLP and NARX crossvalidation performance table
% FinalperformanceMLPvsNARX= {};
% 
% for t =1:243
%     Finalinputdelayindex= cell2mat(AverageperformanceMLP{t,1});
%     Finalfeedbackdelayindex= cell2mat(AverageperformanceMLP{t,2});
%     FinalperformanceMLPvsNARX{t,1} = inputDelaysArray_NARX{1,Finalinputdelayindex};
%     FinalperformanceMLPvsNARX{t,2} =  feedbackDelaysArray_NARX{1,Finalfeedbackdelayindex};
%     FinalperformanceMLPvsNARX{t,3} = cell2mat(AverageperformanceMLP{t,3});
%     FinalperformanceMLPvsNARX{t,4} = cell2mat(AverageperformanceMLP{t,4});
%     FinalperformanceMLPvsNARX{t,5} = cell2mat(AverageperformanceMLP{t,5});
%     FinalperformanceMLPvsNARX{t,6} = AverageperformanceMLP{t,6};
%     FinalperformanceMLPvsNARX{t,7} = AverageperformanceNARX{t,6};
% end

%% Final Models
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%FINAL TRAIN/TESTING%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%MLP%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get training and test set for final training/testing
train_xinput_window = X_input_finaltrain;
train_solarirrad_window = solarirrad_finaltrain;
test_xinput_window = X_input_finaltest;
test_solarirrad_window = solarirrad_finaltest;

% construct lagged training input variables
train_solarirrad_no_lag = train_solarirrad_window(1:end-4,1:1);  
train_solarirrad_lag1 = train_solarirrad_window(2:end-3,1:1);
train_solarirrad_lag2 = train_solarirrad_window(3:end-2,1:1);
train_solarirrad_lag3 = train_solarirrad_window(4:end-1,1:1);

train_xinput_no_lag = train_xinput_window(1:end-4,:);
train_xinput_lag1 = train_xinput_window(2:end-3,:);
train_xinput_lag2 = train_xinput_window(3:end-2,:);

% construct lagged test input variables
test_solarirrad_no_lag = test_solarirrad_window(1:end-4,1:1);
test_solarirrad_lag1 = test_solarirrad_window(2:end-3,1:1);
test_solarirrad_lag2 = test_solarirrad_window(3:end-2,1:1);
test_solarirrad_lag3 = test_solarirrad_window(4:end-1,1:1);

test_xinput_no_lag = test_xinput_window(1:end-4,:);
test_xinput_lag1 = test_xinput_window(2:end-3,:);
test_xinput_lag2 = test_xinput_window(3:end-2,:);


% combine training lags to create comparable input variables for MLP
X2train = [train_xinput_lag1 train_xinput_lag2];
SOLAR3train = [train_solarirrad_lag1 train_solarirrad_lag2 train_solarirrad_lag3];
MLP_Input_finaltrain = [X2train, SOLAR3train];

% combine test lags to create comparable input variables for MLP
X2test = [test_xinput_lag1 test_xinput_lag2];
SOLAR3test = [test_solarirrad_lag1 test_solarirrad_lag2 test_solarirrad_lag3];
MLP_Input_finaltest = [X2test, SOLAR3test];


% configure training function and set hyperparmeters
trainFcn = 'trainlm';
hiddenLayerSize = 20;
net = fitnet(hiddenLayerSize,trainFcn);

%set indices for training,testing and crossvalidation
net.divideFcn = 'divideind';
net.divideParam.trainind = 1:3062;
net.divideParam.valind = 3063:3832;
net.divideParam.testind = 3833:4798;

% set learning rate and momentum hyperparameter values
net.trainParam.lr = 0.03;
net.trainParam.mc = 3;

% set stopping criteria
net.trainParam.epochs = 30;                
net.trainParam.min_grad = 1e-5;
net.trainParam.max_fail = 10;                      

% create combined dataset for testing training an MLP
FULL_XInputdata = [MLP_Input_finaltrain; MLP_Input_finaltest];
FULL_Ylabeldata = [train_solarirrad_no_lag; test_solarirrad_no_lag];

% train MLP
[net,tr] = train(net,FULL_XInputdata',FULL_Ylabeldata');
% plot testFinal training/test performance
figure
plotperform(tr)
savefig("MLPperformance")
% Get one step ahead test prediction
 y_MLPpredict = net(MLP_Input_finaltest'); 
% get final one step ahead MLP test performance
 FINALMLPperformance = perform(net,test_solarirrad_no_lag,y_MLPpredict');
 
 %clear net

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%NARX%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get training and test set for final train/test set
train_xinput_window =X_input_finaltrain;
train_solarirrad_window =solarirrad_finaltrain;
test_xinput_window = X_input_finaltest;
test_solarirrad_window = solarirrad_finaltest ;
fulldata_xinput_window = X_input_fulldata;  
fulldata_solarirrad_window =solarirrad_fulldata;

% create lagged input variables for training and testing
X_FINALNARXtrain = tonndata(train_xinput_window,false,false);
T_FINALNARXtrain = tonndata(train_solarirrad_window ,false,false);
X_FINALNARXtest = tonndata(test_xinput_window,false,false);
T_FINALNARXtest = tonndata(test_solarirrad_window,false,false);
X_FINALNARXFULLDATA = tonndata(fulldata_xinput_window,false,false);
T_FINALNARXFULLDATA = tonndata(fulldata_solarirrad_window,false,false);


% Levenberg-Marquardt backpropagation.
trainFcn = 'trainlm';  


 %set hyperparameters and configure narx
 inputDelays = 1:2;
 feedbackDelays = 1:2;
 hiddenLayerSize = 20;
 figure
 net = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'open',trainFcn);           
                        
                        
% Prepare the Data for Training and Simulation
% The function PREPARETS prepares timeseries data for a particular network,
% shifting time by the minimum amount to fill input states and layer
% states. Using PREPARETS allows you to keep your original time series data
% unchanged, while easily customizing it for networks with differing
% numbers of delays, with open loop or closed loop feedback modes.
[x_FINALNARXfull,xi_FINALNARXfull,ai_FINALNARXfull,t_FINALNARXfull] = preparets(net,X_FINALNARXFULLDATA,{},T_FINALNARXFULLDATA); 
[x_FINALNARXtest,xi_FINALNARXtest,ai_FINALNARXtest,t_FINALNARXtest] = preparets(net,X_FINALNARXtest,{},T_FINALNARXtest);
[x_FINALNARXtrain,xi_FINALNARXtrain,ai_FINALNARXtrain,t_FINALNARXtrain] = preparets(net,X_FINALNARXtrain,{},T_FINALNARXtrain);

                        
%Setup Division of Data for Training, Validation, Testing
% setting training test and crossvalidation indices   
net.divideFcn = 'divideind';
net.divideParam.trainind = 1:3062;
net.divideParam.valind = 3063:3836;
net.divideParam.testind = 3837:4796;

% setting learning rate and momentum hyperparameter values
net.trainParam.lr = 0.003;
net.trainParam.mc = 0.3;

% set stopping criteria
net.trainParam.epochs = 30;                
net.trainParam.min_grad = 1e-5;
net.trainParam.max_fail = 10; 

%Narx training
[net,tr] = train(net,x_FINALNARXfull,t_FINALNARXfull);

%Narx performance graph
plotperform(tr)
savefig("NARXperformance")

% Test the Network
FINALNARX_ypredict = net(x_FINALNARXtest);
performanceFINALNARX = perform(net, t_FINALNARXtest,FINALNARX_ypredict);
                     






