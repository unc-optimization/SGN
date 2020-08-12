"""!@package import_data

Useful function to read different dataset.

"""

# import library
import numpy as np
from sklearn.utils import resample
from sklearn.datasets import load_svmlight_file,dump_svmlight_file
import pandas as pd
from joblib import Memory

from pathlib import Path
from csv import reader
import sys
import os
import sklearn
import scipy

from .func_utils import *

# Important: change these paths according to your setup.
data_path = '../data/'

# check if dataset path exists
if not os.path.exists( data_path ):
	sys.exit( "\033[91m {}\033[00m" .format( "Error: Dataset not found!!!" ))

mem = Memory( data_path + "mycache" )

@mem.cache
def import_data( data_name):
	"""! Import dataset

	Depending on the name of dataset, this function will return the normalized dataset.

	Parameters
	---------- 
	@param data_name : name of the dataset
	    
	Returns
	---------- 
	@retval X_train : input data
	@retval Y_train : input label
	"""

	if not os.path.exists( data_path + data_name + '_normalized'):
		X_train, Y_train = load_svmlight_file( data_path + data_name )

		# normalize data
		print("Normalizing data...")
		sklearn.preprocessing.normalize(X_train, 'l2', axis=1, copy=False)

		dump_svmlight_file(X_train, Y_train, data_path + data_name + '_normalized')
	else:
		X_train, Y_train = load_svmlight_file( data_path + data_name + '_normalized' )
	return X_train, Y_train

def generate_data(num_func=1, n=1000, d=200, seed=42):

	Func_list = [	func_val_bin_class_loss_1,
					func_val_bin_class_loss_2,
					func_val_bin_class_loss_3,
					func_val_bin_class_loss_4,]

	Func_Diff_list = [	func_diff_eval_bin_class_loss_1,
						func_diff_eval_bin_class_loss_2,
						func_diff_eval_bin_class_loss_3,
						func_diff_eval_bin_class_loss_4,]

	Grad_list = [	grad_eval_bin_class_loss_1,
					grad_eval_bin_class_loss_2,
					grad_eval_bin_class_loss_3,
					grad_eval_bin_class_loss_4,]

	Grad_Diff_list = [	grad_diff_eval_bin_class_loss_1,
						grad_diff_eval_bin_class_loss_2,
						grad_diff_eval_bin_class_loss_3,
						grad_diff_eval_bin_class_loss_4,]

	num_available_func = len(Func_list)

	if num_func > num_available_func:
		num_func = num_available_func

	Non_Linear_Func_list = []
	Non_Linear_Func_Diff_list = []
	Non_Linear_Grad_list = []
	Non_Linear_Grad_Diff_list = []

	for i in range(num_func):
		Non_Linear_Func_list.append(Func_list[i])
		Non_Linear_Func_Diff_list.append(Func_Diff_list[i])
		Non_Linear_Grad_list.append(Grad_list[i])
		Non_Linear_Grad_Diff_list.append(Grad_Diff_list[i])

	# check if synthetic data exists, if not then generate data
	if not os.path.exists( os.path.join(data_path,'X_synth_'+str(n) +'.npz') ):

		print('Create Synthetic Data')

		A_mat = scipy.sparse.random(n,d, density = 0.1, format = 'csr')
		print("Normalizing data...")
		sklearn.preprocessing.normalize(A_mat, 'l2', axis=1, copy=False)

		scipy.sparse.save_npz(os.path.join(data_path,'X_synth_'+str(n) +'.npz'), A_mat)

		# intialize a label vector
		np.random.seed(seed)
		y_in = 2*np.random.binomial(1,0.45,size=n)-1
		np.save(os.path.join(data_path,'Y_synth_'+str(n) +'.npy'), y_in)

	# if synthetic data exists, load data
	else:
		print('Load Synthetic Data')

		A_mat = scipy.sparse.load_npz(os.path.join(data_path,'X_synth_'+str(n) +'.npz'))
		y_in = np.load(os.path.join(data_path,'Y_synth_'+str(n) +'.npy') )	

	bias = np.zeros( n )

	Non_Linear_Func = {
		'Func': Non_Linear_Func_list,
		'FuncDiff': Non_Linear_Func_Diff_list,
		'Grad': Non_Linear_Grad_list,
		'GradDiff': Non_Linear_Grad_Diff_list,
	}

	Non_Linear_Data = {
		'matrix': A_mat,
		'label': y_in,
		'bias': bias,
	}

	return Non_Linear_Func, Non_Linear_Data

def intialize_func(num_func, X, Y):

	Func_list = [	func_val_bin_class_loss_1,
					func_val_bin_class_loss_2,
					func_val_bin_class_loss_3,
					func_val_bin_class_loss_4,]

	Func_Diff_list = [	func_diff_eval_bin_class_loss_1,
						func_diff_eval_bin_class_loss_2,
						func_diff_eval_bin_class_loss_3,
						func_diff_eval_bin_class_loss_4,]

	Grad_list = [	grad_eval_bin_class_loss_1,
					grad_eval_bin_class_loss_2,
					grad_eval_bin_class_loss_3,
					grad_eval_bin_class_loss_4,]

	Grad_Diff_list = [	grad_diff_eval_bin_class_loss_1,
						grad_diff_eval_bin_class_loss_2,
						grad_diff_eval_bin_class_loss_3,
						grad_diff_eval_bin_class_loss_4,]

	num_available_func = len(Func_list)

	Non_Linear_Func_list = []
	Non_Linear_Func_Diff_list = []
	Non_Linear_Grad_list = []
	Non_Linear_Grad_Diff_list = []

	for i in range(num_func):
		Non_Linear_Func_list.append(Func_list[i%num_available_func])
		Non_Linear_Func_Diff_list.append(Func_Diff_list[i%num_available_func])
		Non_Linear_Grad_list.append(Grad_list[i%num_available_func])
		Non_Linear_Grad_Diff_list.append(Grad_Diff_list[i%num_available_func])

	Non_Linear_Func = {
		'Func': Non_Linear_Func_list,
		'FuncDiff': Non_Linear_Func_Diff_list,
		'Grad': Non_Linear_Grad_list,
		'GradDiff': Non_Linear_Grad_Diff_list,
	}

	n_ = X.shape[0]

	bias = np.zeros( n_ )

	Non_Linear_Data = {
		'matrix': X,
		'label': Y,
		'bias': bias,
	}

	return Non_Linear_Func, Non_Linear_Data

def save_history(log_dir,history):
	# check if directory exists, if not, create directory then save
	if not os.path.exists( log_dir ):
	    os.makedirs( log_dir )

	df = pd.DataFrame(history)
	df.to_csv(os.path.join(log_dir,"history.csv"), index=False)