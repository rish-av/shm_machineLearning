#Methods of Auto-Regressive Model for featue extraction and building test matrix for test using different algorithms for structural health monitoring.

import numpy as np
import scipy.io as sci
from matplotlib import pyplot as plt
from statsmodels.tsa.ar_model import AR, ARResults

def load_data(path):
    data = sci.io.loadmat(path)
    return data['dataset']
	
def plot_signal(data,channel_no,test_no):
    plt.plot(data[:,channel_no,test_no])
    plt.xlabel('Accelaration')
    plt.ylabel('Time Step')
    
def fit_AR(test):
    model = AR(test)
    model_fit = model.fit()
    return model.fit
def plot_ar_parameters(params):
    plt.plot(params)
    plt.xlabel('')
    plt.ylabel
def comparative_plot_different_states(param1,param2):
    x = [i for i in range(len(param1))]
    plt.subplot(x,param1,'-o')
    plt.subplot(x,param2,'-o')
    plt.xlabel('Feature Number')
    plt.ylabel('Auto Regressive Feature Value')
    plt.show()
def get_model_order(data):
    model_fit = fit_AR(data[:,1,1])
    return len(model_fit.params)

def get_params(fitted_model):
    return fitted_model.params

def build_test_matrix(data,given_state_test_number,start_channel_number,end_channel_number):
    X = [] #test_matrix
    for i in range(given_state_test_number):
        order = get_model_order(data)
        avg_feature_over_channels = np.zeros(order)
        for j in range(start_channel_number,end_channel_number):
            model_fit = fit_AR(data[:,j,i])
            params = get_params(model_fit)
            avg_feature_over_channels+=params
        stdev = np.std(avg_feature_over_channels)
        mean = np.mean(avg_feature_over_channels)
        avg_feature_over_channels = (avg_feature_over_channels - mean)/(stdev)
        a = avg_feature_over_channels.shape[0]
        avg_feature_over_channels.reshape(a,1)
        X.append(avg_feature_over_channels)
    return np.array(X)