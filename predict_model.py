#### https://www.kaggle.com/neelkudu28/covid-19-visualizations-predictions-forecasting  ---- Covid Predictions used for Polynomial regression and Holt prediction
## https://www.kaggle.com/saga21/covid-global-forecast-sir-model-ml-regressions   ----- Covid predictions used for Linear Lagged prediction model
import warnings
warnings.filterwarnings('ignore')                   #currently warnings ignored, you can see the warnings if you comment this
import pandas as pd
import numpy as np
#import datetime as dt
#from datetime import timedelta
#import time
import pickle
import os

##from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression,Ridge,Lasso,LassoLars       #pip install sklearn
#from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import mean_squared_error,r2_score,explained_variance_score,max_error,mean_absolute_error


####################################################################################################

def calculate_lag(df, lag_list, column):			#use of df.shift to create additional columns of t-1,t-2,...,t-lag_size
    for lag in lag_list:
        column_lag = column + "_" + str(lag)                #deal with 'Close_1','Close_2','Close_3',... columns
        df[column_lag] = df[column].shift(lag, fill_value=0)
    return df

def calculate_leads(df, lag_list, column):			#use of df.shift to create additional columns of t+1,t+2,...,t+num_days_ahead
    for lag in lag_list:
        column_lag = column + "p" + str(lag)                #deal with 'Closep1','Closep2',Closep3',... columns
        df[column_lag] = df[column].shift(-lag, fill_value=0)
    return df

##################################################################################################################

def predict_on_lastupdate():
	THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
	my_stockcsv = os.path.join(THIS_FOLDER, 'StockData.csv')
	df = pd.read_csv(my_stockcsv)
	#df=pd.read_csv("NSEI.csv")   #read the covid data from file
	#df=pd.read_csv('StockData.csv')
	df_NSE = df[['Date','Close','Open','High','Low','Volume']]#.iloc[-730:,:]#.reset_index()   #work the predictions only for the column 'Close' in the rest of code
	#df_NSE = df[['Date','Close']]		#Use date columns for housekeeping and Close column is the target but older values are forked to populate lag columns which are used as regressor features
	df_NSE.dropna(axis=0,how='any', inplace=True) #drop rows with null values
	df_NSE['Date'] = pd.to_datetime(df_NSE['Date']) #change date values to standard date format


	df_NSE['Days Since'] = list(range(0, df_NSE.shape[0]))		#for house keeping

	################################################################################################################################################################################

	#model_scores = []

	num_days_ahead=2		# t,t+1,t+2,...,t+num_days_ahead-1 values are predicted
	num_predictdays = 100		# number of days of testing, as testing days are increased r2score is seen to improve	
	lag_size = 15      #Older values of Close column are populated in additional columns

	lagpred_data_features = df_NSE.copy()       #work with local copy, needed to do store inplace the predicted out and to compare with reference

	lagpred_data_features = calculate_lag(lagpred_data_features, range(1, lag_size), 'Close')       #update the Close_1,Close_2 etc columns
	lagpred_data_features = calculate_lag(lagpred_data_features, range(1, lag_size), 'Open')       #update the Close_1,Close_2 etc columns
	lagpred_data_features = calculate_lag(lagpred_data_features, range(1, lag_size), 'High')       #update the Close_1,Close_2 etc columns
	lagpred_data_features = calculate_lag(lagpred_data_features, range(1, lag_size), 'Low')       #update the Close_1,Close_2 etc columns
	lagpred_data_features = calculate_lag(lagpred_data_features, range(1, lag_size), 'Volume')       #update the Close_1,Close_2 etc columns
	idx = lagpred_data_features[0:lag_size].index
	lagpred_data_features = lagpred_data_features.drop(idx,axis=0)
	lagpred_data_features = calculate_leads(lagpred_data_features, range(1, num_days_ahead), 'Close')       #update the Close_1,Close_2 etc columns
	lagpred_data_features = calculate_leads(lagpred_data_features, range(1, num_days_ahead), 'Open')       #update the Close_1,Close_2 etc columns
	lagpred_data_features = calculate_leads(lagpred_data_features, range(1, num_days_ahead), 'High')       #update the Close_1,Close_2 etc columns
	lagpred_data_features = calculate_leads(lagpred_data_features, range(1, num_days_ahead), 'Low')       #update the Close_1,Close_2 etc columns
	lagpred_data_features = calculate_leads(lagpred_data_features, range(1, num_days_ahead), 'Volume')       #update the Close_1,Close_2 etc columns
	#idx = lagpred_data_features[-num_days_ahead:].index
	#lagpred_data_features = lagpred_data_features.drop(idx,axis=0)

	#train_ml = lagpred_data_features.iloc[:int(lagpred_data_features.shape[0] * 0.95)]
	train_ml = lagpred_data_features.iloc[:-num_predictdays]		#Data to feed the fitter
	#valid_ml = lagpred_data_features.iloc[int(lagpred_data_features.shape[0] * 0.95):]
	valid_ml = lagpred_data_features.iloc[-num_predictdays:]		#Data to feed tester/predictor

	filter_col_Close = [col for col in lagpred_data_features if col.startswith('Close_')]		#list of 'Close_' column names
	filter_col_Open = [col for col in lagpred_data_features if col.startswith('Open_')]		#list of 'Close_' column names
	filter_col_High = [col for col in lagpred_data_features if col.startswith('High_')]		#list of 'Close_' column names
	filter_col_Low = [col for col in lagpred_data_features if col.startswith('Low_')]		#list of 'Close_' column names
	filter_col_Volume = [col for col in lagpred_data_features if col.startswith('Volume_')]		#list of 'Close_' column names

	filter_col_depvar = ['Open'] + ['High'] + ['Low'] + ['Close'] + filter_col_Open + filter_col_High + filter_col_Low + filter_col_Close + filter_col_Volume

	filter_col_Closep = [col for col in lagpred_data_features if col.startswith('Closep')]		#list of 'Closep' column names
	filter_col_Openp = [col for col in lagpred_data_features if col.startswith('Openp')]		#list of 'Closep' column names
	filter_col_Highp = [col for col in lagpred_data_features if col.startswith('Highp')]		#list of 'Closep' column names
	filter_col_Lowp = [col for col in lagpred_data_features if col.startswith('Lowp')]		#list of 'Closep' column names
	#filter_col_Volumep = [col for col in lagpred_data_features if col.startswith('Volumep')]		#list of 'Closep' column names

	target_cols = filter_col_Openp + filter_col_Highp + filter_col_Lowp + filter_col_Closep# + ['Volume'] + filter_col_Volumep

	#fitx = np.array(train_ml[filter_col_depvar]).reshape(-1,len(filter_col_depvar))	#number of features lag_size-1
	#fity = np.array(train_ml[target_cols]).reshape(-1,len(target_cols))	#number of predictions per prediction input is num_days_ahead

	#regr = LinearRegression(normalize=True)
	#regr.fit(fitx,fity)
	my_picklepath = os.path.join(THIS_FOLDER, 'model_fit_predict.pkl')
	regr = pickle.load(open(my_picklepath,"rb"))
	#regr = pickle.load(open('model_fit_predict.pkl', 'rb'))

	whole_test_input = valid_ml[filter_col_depvar]
	xin_predict = np.array(whole_test_input.iloc[-1,:]).reshape(-1,len(filter_col_depvar))	#number of predictor input values lag_size-1

	y_pred = regr.predict(xin_predict)

	predout = pd.DataFrame(zip(target_cols,y_pred[-1,:]),columns=["target_vars","Share_predicted_value"])
	#print(predout)
	#breakpoint()
	return np.round(y_pred[0],2)

def getdate_csv(stock_name):
        THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
        my_stockcsv = os.path.join(THIS_FOLDER, 'StockData.csv')
        df = pd.read_csv(my_stockcsv)
        print("Stock Name is ",stock_name)
	#df=pd.read_csv('StockData.csv')
        df_NSE = df[['Date','Close','Open','High','Low','Volume']]#.iloc[-730:,:]#.reset_index()   #work the predictions only for the column 'Close' in the rest of code
	#df_NSE = df[['Date','Close']]		#Use date columns for housekeeping and Close column is the target but older values are forked to populate lag columns which are used as regressor features
        df_NSE.dropna(axis=0,how='any', inplace=True) #drop rows with null values
        df_NSE['Date'] = pd.to_datetime(df_NSE['Date']) #change date values to standard date 
        return df_NSE['Date'].iloc[-1]

