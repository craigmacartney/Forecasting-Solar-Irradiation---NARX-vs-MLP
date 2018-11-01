  
%% Descriptive Statistics of Trend of Solar Irradiance Over Time 

  %Read-in data
  Solar1 = readtable("solarFINAL2.csv");

  %Pre-Processing - identify variable columns
  Irrad = Solar1{1:end,1};
  Zenith = Solar1{1:end,2};
  Azimuth = Solar1{1:end,3};
  Day = Solar1{1:end,4};
  Hour = Solar1{1:end,5};
  
  
  
  %% Calculate Variable Summary Statistics
  SumStats = []; %initialize empty array
  
  SumStats(1,1) = min(Irrad); %minumum of SUNY Glo (Wh/m^2) (Global Solar Radiation) column
  SumStats(1,2) = max(Irrad); %maximum of SUNY Glo (Wh/m^2) (Global Solar Radiation) column
  SumStats(1,3) = mean(Irrad); %mean of SUNY Glo (Wh/m^2) (Global Solar Radiation) column
  SumStats(1,4) = var(Irrad); %variance of SUNY Glo (Wh/m^2) (Global Solar Radiation) column
  SumStats(1,5) = kurtosis(Irrad); %kurtosis of SUNY Glo (Wh/m^2) (Global Solar Radiation) column
  
  SumStats(2,1) = min(Zenith); %minumum of Zenith Angle column
  SumStats(2,2) = max(Zenith); %maximum of Zenith Angle column
  SumStats(2,3) = mean(Zenith); %mean of Zenith Angle column
  SumStats(2,4) = var(Zenith); %variance of Zenith Angle column
  SumStats(2,5) = kurtosis(Zenith); %kurtosis of Zenith Angle column
  
  SumStats(3,1) = min(Azimuth); %minumum of Azimuth Angle column
  SumStats(3,2) = max(Azimuth); %maximum of Azimuth Angle column
  SumStats(3,3) = mean(Azimuth); %mean of Azimuth Angle column
  SumStats(3,4) = var(Azimuth); %variance of Azimuth Angle column
  SumStats(3,5) = kurtosis(Azimuth); %kurtosis of Azimuth Angle column
  
  SumStats(4,1) = min(Day); %minumum of Day column
  SumStats(4,2) = max(Day); %maximum of Day column
  SumStats(4,3) = mean(Day); %mean of Day column
  SumStats(4,4) = var(Day); %variance of Day column
  SumStats(4,5) = kurtosis(Day); %kurtosis of Day column
  
  SumStats(5,1) = min(Hour); %minumum of Hour column
  SumStats(5,2) = max(Hour); %maximum of Hour column
  SumStats(5,3) = mean(Hour); %mean of Hour column 
  SumStats(5,4) = var(Hour); %variance of Hour column 
  SumStats(5,5) = kurtosis(Hour); %kurtosis of Hour column 
  
  
  
  %% Preparation of Time Series Lags for Correlation
  
  %Initialise  no lag
  Irrad_no_lag = Solar1{1:end-10,1};
  
  %Variable lags of 10 time points
  Irrad_lag_1 = Solar1{2:end-9,1};
  Irrad_lag_2 = Solar1{3:end-8,1};
  Irrad_lag_3 = Solar1{4:end-7,1};
  Irrad_lag_4 = Solar1{5:end-6,1};
  Irrad_lag_5 = Solar1{6:end-5,1};
  Irrad_lag_6 = Solar1{7:end-4,1};
  Irrad_lag_7 = Solar1{8:end-3,1};  
  Irrad_lag_8 = Solar1{9:end-2,1};
  Irrad_lag_9 = Solar1{10:end-1,1};
  Irrad_lag_10 = Solar1{11:end,1};
  
  Azimuth_lag_1 = Solar1{2:end-9,2};
  Azimuth_lag_2 = Solar1{3:end-8,2};
  Azimuth_lag_3  = Solar1{4:end-7,2};
  Azimuth_lag_4 = Solar1{5:end-6,2};
  Azimuth_lag_5 = Solar1{6:end-5,2};
  Azimuth_lag_6 = Solar1{7:end-4,2};
  Azimuth_lag_7 = Solar1{8:end-3,2};
  Azimuth_lag_8 = Solar1{9:end-2,2};
  Azimuth_lag_9 = Solar1{10:end-1,2};
  Azimuth_lag_10 = Solar1{11:end,2};
  
  Zenith_lag_1 = Solar1{2:end-9,3};
  Zenith_lag_2 = Solar1{3:end-8,3};
  Zenith_lag_3  = Solar1{4:end-7,3};
  Zenith_lag_4 = Solar1{5:end-6,3};
  Zenith_lag_5 = Solar1{6:end-5,3};
  Zenith_lag_6 = Solar1{7:end-4,3};
  Zenith_lag_7 = Solar1{8:end-3,3};
  Zenith_lag_8 = Solar1{9:end-2,3};
  Zenith_lag_9 = Solar1{10:end-1,3};
  Zenith_lag_10 = Solar1{11:end,3};
 
  
  Hour_lag_1 = Solar1{2:end-9,4}; 
  Hour_lag_2 = Solar1{3:end-8,4};
  Hour_lag_3 = Solar1{4:end-7,4};
  Hour_lag_4 = Solar1{5:end-6,4};
  Hour_lag_5 = Solar1{6:end-5,4};
  Hour_lag_6 = Solar1{7:end-4,4};
  Hour_lag_7 = Solar1{8:end-3,4};
  Hour_lag_8 = Solar1{9:end-2,4};
  Hour_lag_9 = Solar1{10:end-1,4};
  Hour_lag_10 = Solar1{11:end,4};
  
  Day_lag_1 = Solar1{2:end-9,5}; 
  Day_lag_2 = Solar1{3:end-8,5};
  Day_lag_3 = Solar1{4:end-7,5};
  Day_lag_4 = Solar1{5:end-6,5};
  Day_lag_5 = Solar1{6:end-5,5};
  Day_lag_6 = Solar1{7:end-4,5};
  Day_lag_7 = Solar1{8:end-3,5};
  Day_lag_8 = Solar1{9:end-2,5};
  Day_lag_9 = Solar1{10:end-1,5};
  Day_lag_10 = Solar1{11:end,5};
  
  %% Correlation between solar irradiance and lagged predictors
  
  CorrTable = [];
  
  CorrTable(1,1) = corr(Irrad_no_lag,Irrad_lag_1);
  CorrTable(2,1) = corr(Irrad_no_lag,Irrad_lag_2);
  CorrTable(3,1) = corr(Irrad_no_lag,Irrad_lag_3);
  CorrTable(4,1) = corr(Irrad_no_lag,Irrad_lag_4);
  CorrTable(5,1) = corr(Irrad_no_lag,Irrad_lag_5); 
  CorrTable(6,1) = corr(Irrad_no_lag,Irrad_lag_6); 
  CorrTable(7,1) = corr(Irrad_no_lag,Irrad_lag_7); 
  CorrTable(8,1) = corr(Irrad_no_lag,Irrad_lag_8); 
  CorrTable(9,1) = corr(Irrad_no_lag,Irrad_lag_9); 
  CorrTable(10,1) = corr(Irrad_no_lag,Irrad_lag_10); 
 
  CorrTable(1,2) = corr(Irrad_no_lag,Azimuth_lag_1);  
  CorrTable(2,2) = corr(Irrad_no_lag,Azimuth_lag_2);
  CorrTable(3,2) = corr(Irrad_no_lag,Azimuth_lag_3);
  CorrTable(4,2) = corr(Irrad_no_lag,Azimuth_lag_4);
  CorrTable(5,2) = corr(Irrad_no_lag,Azimuth_lag_5);
  CorrTable(6,2) = corr(Irrad_no_lag,Azimuth_lag_6);
  CorrTable(7,2) = corr(Irrad_no_lag,Azimuth_lag_7);
  CorrTable(8,2) = corr(Irrad_no_lag,Azimuth_lag_8);
  CorrTable(9,2) = corr(Irrad_no_lag,Azimuth_lag_9);
  CorrTable(10,2) = corr(Irrad_no_lag,Azimuth_lag_10);

  CorrTable(1,3) = corr(Irrad_no_lag,Zenith_lag_1); 
  CorrTable(2,3) = corr(Irrad_no_lag,Zenith_lag_2);
  CorrTable(3,3) = corr(Irrad_no_lag,Zenith_lag_3);
  CorrTable(4,3) = corr(Irrad_no_lag,Zenith_lag_4);
  CorrTable(5,3) = corr(Irrad_no_lag,Zenith_lag_5);
  CorrTable(6,3) = corr(Irrad_no_lag,Zenith_lag_6);
  CorrTable(7,3) = corr(Irrad_no_lag,Zenith_lag_7);
  CorrTable(8,3) = corr(Irrad_no_lag,Zenith_lag_8);
  CorrTable(9,3) = corr(Irrad_no_lag,Zenith_lag_9);
  CorrTable(10,3) = corr(Irrad_no_lag,Zenith_lag_10);

  CorrTable(1,4) = corr(Irrad_no_lag,Day_lag_1);  
  CorrTable(2,4)= corr(Irrad_no_lag,Day_lag_2); 
  CorrTable(3,4)= corr(Irrad_no_lag,Day_lag_3);
  CorrTable(4,4) = corr(Irrad_no_lag,Day_lag_4);
  CorrTable(5,4) = corr(Irrad_no_lag,Day_lag_5);
  CorrTable(6,4) = corr(Irrad_no_lag,Day_lag_6);
  CorrTable(7,4) = corr(Irrad_no_lag,Day_lag_7);
  CorrTable(8,4) = corr(Irrad_no_lag,Day_lag_8);
  CorrTable(9,4) = corr(Irrad_no_lag,Day_lag_9);
  CorrTable(10,4) = corr(Irrad_no_lag,Day_lag_10);

  
  CorrTable(1,5) = corr(Irrad_no_lag,Hour_lag_1); 
  CorrTable(2,5) = corr(Irrad_no_lag,Hour_lag_2);
  CorrTable(3,5) = corr(Irrad_no_lag,Hour_lag_3);
  CorrTable(4,5) = corr(Irrad_no_lag,Hour_lag_4);
  CorrTable(5,5) = corr(Irrad_no_lag,Hour_lag_5); 
  CorrTable(6,5) = corr(Irrad_no_lag,Hour_lag_6); 
  CorrTable(7,5) = corr(Irrad_no_lag,Hour_lag_7); 
  CorrTable(8,5) = corr(Irrad_no_lag,Hour_lag_8); 
  CorrTable(9,5) = corr(Irrad_no_lag,Hour_lag_9); 
  CorrTable(10,5) = corr(Irrad_no_lag,Hour_lag_10); 

  
  %% Visualisation of Trend of Solar Irradiance Over Time 
  %Normalise data
  Irrad = (Irrad - min(Irrad))/(max(Irrad)-min(Irrad));
  Zenith = (Zenith - min(Zenith))/(max(Zenith)-min(Zenith));
  Azimuth = (Azimuth - min(Azimuth))/(max(Azimuth)-min(Azimuth));
  
  %Plot Trend of Solar Irradiance Over Time
  Day= Day* 24 - 24;
  Hours = Day+Hour;
  Hours = Hours/24;
  plot(Hours,Irrad);
  %title('2-D Line Plot')
  xlabel('Day in 2009');
  ylabel('Solar Irradiance');
  xlim([0 365]);
 
  