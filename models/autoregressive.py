#Methods of Auto-Regressive Model for featue extraction and building test matrix for test using different algorithms for structural health monitoring.
import numpy as np
from scipy import io
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AR, ARResults
from sklearn.preprocessing import normalize

def load_data(path):
    data = io.loadmat(path)
    return data['dataset']
	
def plot_signal(data,channel_no,test_no):
    plt.plot(data[:,channel_no,test_no])
    plt.xlabel('Time Step')
    plt.ylabel('Accelaration')
    
def fit_AR(test):
    model = AR(test)
    model_fit = model.fit()
    return model_fit
def plot_ar_parameters(params):
    plt.plot(params)
    plt.xlabel('Parameter Number')
    plt.ylabel('Parameter Value')

def get_model_order(data):
    model_fit = fit_AR(data[:,1,1])
    return model_fit.k_ar+1

def get_params(fitted_model):
    return fitted_model.params

def build_matrix(data,given_state_test_number,start_channel_number,end_channel_number):
    X = [] #test_matrix
    for i in range(given_state_test_number):
        order = get_model_order(data)
        avg_feature_over_channels = np.zeros(order)
        for j in range(start_channel_number,end_channel_number):
            model_fit = fit_AR(data[:,j,i])
            params = get_params(model_fit)
            avg_feature_over_channels+=params
        a = avg_feature_over_channels.shape[0]
        avg_feature_over_channels/=(end_channel_number-start_channel_number)
        avg_feature_over_channels.reshape(a,1)
        X = normalize(X,axis=1)
        X.append(avg_feature_over_channels)
    return np.array(X)

def build_test_matrix(data,start,end,start_channel,end_channel):
	T = []
	for i in range(start,end):
		order = get_model_order(data)
		avg_feature_over_channels = np.zeros(order)
		for j in range(start_channel,end_channel):
			model_fit = fit_AR(data[:,j,i])
			params = get_params(model_fit)
			avg_feature_over_channels+=params
		avg_feature_over_channels/=(end_channel-start_channel)
		a = avg_feature_over_channels.shape[0]
		avg_feature_over_channels.reshape(a,1)
		T = normalize(T,axis=1)
		T.append(avg_feature_over_channels)
	return np.array(T)


