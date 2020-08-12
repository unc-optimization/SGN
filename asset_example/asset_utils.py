"""!@package asset_utils

Supporting functions for the asset allocation example.

"""

import numpy as np
import scipy
import random
import math

# projection onto standard simplex
def proj_simplex(x ,lbd = 1):
	sort_x = np.sort(x)[::-1]

	len_x = len(x)

	stop = False

	tmp_sum = 0
	for i in range(len_x-1):
		tmp_sum = tmp_sum + sort_x[i]

		tmax = (tmp_sum - 1) / (i+1)
		if tmax >= sort_x[i + 1]:
			stop = True
			break

	if stop == False:
		tmax = (tmp_sum + sort_x[-1] - 1)/len_x

	return np.maximum(x - tmax,0)

def prox_l1_norm( w, lamb=1 ):
	"""! Compute the proximal operator of the \f$\ell_1\f$ - norm

	\f$ prox_{\lambda \|.\|_1} = {arg\min_x}\left\{\|.\|_1^2 + \frac{1}{2\lambda}\|x - w\|^2\right\} \f$
	
	Parameters
	---------- 
	@param w : input vector
	@param lamb : penalty paramemeter
	    
	Returns
	---------- 
	@retval : perform soft - thresholding on input vector
	"""
	return np.sign( w ) * np.maximum( np.abs( w ) - lamb, 0 )


# asset allocation example
def asset_alloc_obj(x, lbd=1):
	lbd=0.01

	return -x[0] + lbd*x[0]**2 - lbd*x[1]

def prox_asset_alloc(x, rho):
	lbd=0.01

	res = np.zeros_like(x)
	res[0] = (1 + x[0]/rho)/(2*lbd + 1.0/rho)
	res[1] = x[1] + lbd*rho

	return res

def f_func(A, w, tau, beta=0.1, epsilon=1e-3, bs=1, index=None):

	if bs is None and index is None:
		print('Error, must input batch size or index set')
		return

	if bs is None:
		bs = len(index)

	n = A.shape[0]
	d = A.shape[1]

	if bs == 1:
		if index is None:
			index = np.random.randint( 0, n )
		Ai = A[index,:]

	elif bs < n:
		if index is None:
			index = random.sample( range( n ), bs )
		Ai = A[index,:]
	else:
		Ai = A

	Awt = Ai.dot(w) + tau

	return tau + 0.5*(1/beta)*np.mean(np.sqrt(Awt**2 + epsilon**2) - Awt - epsilon)

def f_func_diff(A, w1, tau1, w2, tau2, beta=0.1, epsilon=1e-3, bs=1, index=None):

	if bs is None and index is None:
		print('Error, must input batch size or index set')
		return

	if bs is None:
		bs = len(index)

	n = A.shape[0]
	d = A.shape[1]

	if bs == 1:
		if index is None:
			index = np.random.randint( 0, n )
		Ai = A[index,:]

	elif bs < n:
		if index is None:
			index = random.sample( range( n ), bs )
		Ai = A[index,:]
	else:
		Ai = A

	Awt1 = Ai.dot(w1) + tau1
	Awt2 = Ai.dot(w2) + tau2

	return tau2 - tau1 + 0.5*(1/beta)*np.mean(np.sqrt((Awt2)**2 + epsilon**2) - np.sqrt((Awt1)**2 + epsilon**2) - Awt2 + Awt1)
	
def f_jac(A, w, tau, beta=0.1, epsilon=1e-3, bs=1, index=None):

	if bs is None and index is None:
		raise ValueError('Error, must input batch size or index set')

	if bs is None:
		bs = len(index)

	n = A.shape[0]
	d = A.shape[1]

	res = np.zeros(d+1)

	if bs == 1:
		if index is None:
			index = np.random.randint( 0, n )
		Ai = A[index,:]

	elif bs < n:
		if index is None:
			index = random.sample( range( n ), bs )
		Ai = A[index,:]
	else:
		Ai = A

	Awt = Ai.dot(w) + tau

	tmp_ratio = (Awt/np.sqrt(epsilon**2 + Awt**2))

	res[:-1] = 0.5*beta* np.mean(-Ai + Ai.T.dot(tmp_ratio), axis = 0)

	res[-1] = 1 + 0.5*beta*np.mean(-1 + tmp_ratio)

	return res

def f_jac_diff(A, w1, tau1, w2, tau2, beta=0.1, epsilon=1e-3, bs=1, index=None):

	if bs is None and index is None:
		raise ValueError('Error, must input batch size or index set')

	if bs is None:
		bs = len(index)

	n = A.shape[0]
	d = A.shape[1]

	res = np.zeros(d+1)

	if bs == 1:
		if index is None:
			index = np.random.randint( 0, n )
		Ai = A[index,:]

	elif bs < n:
		if index is None:
			index = random.sample( range( n ), bs )
		Ai = A[index,:]
	else:
		Ai = A

	Awt1 = Ai.dot(w1) + tau1
	Awt2 = Ai.dot(w2) + tau2

	tmp_ratio1 = (Awt1/np.sqrt(epsilon**2 + Awt1**2))
	tmp_ratio2 = (Awt2/np.sqrt(epsilon**2 + Awt2**2))

	res[:-1] = 0.5*beta* (np.mean(Ai.T.dot(tmp_ratio2), axis = 0) - np.mean(Ai.T.dot(tmp_ratio1), axis = 0))

	res[-1] = 0.5*beta*(np.mean(tmp_ratio2) - np.mean(tmp_ratio2))

	return res


