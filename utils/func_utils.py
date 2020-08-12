"""!@package func_utils

Multiple nonconvex loss functions.

"""

import numpy as np
import scipy
import random
import math

import time

## constant indicating total available memory when calculating full gradient
total_mem_full = 3.0e10

## constant indicating total available memory when calculating batch gradient
total_mem_batch = 2.0e10


def prox_l1_norm( w, lamb=1 ):
	"""! Compute the proximal operator of the \f$\ell_1\f$ - norm

	\f$ prox_{\lambda \|.\|_1} = {arg\min_x}\left\{\|.\|_1^2 + \frac{1}{2\lambda}\|x - w\|^2\right\} \f$
	
	Parameters
	---------- 
	@param w : input vector
	@param lamb : penalty paramemeter
	    
	Returns
	---------- 
	@retval : output vector
	"""
	return np.sign( w ) * np.maximum( np.abs( w ) - lamb, 0 )

def prox_l2_norm( w, lamb=1 ):
	"""! Compute the proximal operator of the \f$\ell_2\f$ - norm

	Parameters
	---------- 
	@param w : input vector
	@param lamb : penalty paramemeter
	    
	Returns
	---------- 
	@retval : output vector
	"""
	norm_w = np.linalg.norm(w, ord=2)
	return np.maximum(1 - lamb/norm_w, 0) * w

def prox_linf_norm( w, lamb=1 ):
	"""! Compute the proximal operator of the \f$\ell_2\f$ - norm

	\f$ prox_{\lambda \|.\|_2} = {arg\min_x}\left\{\|.\|_1^2 + \frac{1}{2\lambda}\|x - w\|^2\right\} \f$
	
	Parameters
	---------- 
	@param w : input vector
	@param lamb : penalty paramemeter
	    
	Returns
	---------- 
	@retval : output vector
	"""
	return w - lamb * proj_l1_ball(w / lamb)

def proj_l2_ball( w, lamb=1 ):
	"""! Compute the projection onto \f$\ell_2\f$-ball
	
	Parameters
	---------- 
	@param w : input vector
	@param lamb : penalty paramemeter
	    
	Returns
	---------- 
	@retval : output vector
	"""
	norm_w = np.linalg.norm(w, ord=2)
	if norm_w > lamb:
		return lamb * w / norm_w
	else:
		return w

def proj_l1_ball( w, lamb=1 ):
	"""! Compute the projection onto \f$\ell_1\f$-ball
	
	Parameters
	---------- 
	@param w : input vector
	@param lamb : penalty paramemeter
	    
	Returns
	---------- 
	@retval : output vector
	"""
	norm_w = np.linalg.norm(w, ord=1)

	if norm_w <= 1:
		return w
	else:
		# find lamb by bisection
		sort_w = np.sort(w)[::-1]
		tmp_sum = 0
		index = -1
		for i in range(len(sort_w)):
			tmp_sum += sort_w[i]
			if sort_w[i] <= (1.0/(i+1)) * (tmp_sum - 1):
				break
			else:
				index += 1

			index = np.max(index,0)

		lamb = np.max((1.0/(index+1))*(tmp_sum - 1),0)

		return prox_l1_norm(w,lamb)

def proj_linf_ball( w, lamb=1 ):
	"""! Compute the projection onto \f$\ell_{\infty}\f$-ball
	
	Parameters
	---------- 
	@param w : input vector
	@param lamb : penalty paramemeter
	    
	Returns
	---------- 
	@retval : perform projection onto $\ell_{\infty}$-ball
	"""
	norm_w = np.linalg.norm(w, ord=np.inf)

	if norm_w > lamb:
		return lamb * w / norm_w
	else:
		return w

def ind_l2_ball( w, lamb=1 ):
	"""! Compute the indication function of the \f$\ell_{2}\f$-ball
	
	Parameters
	---------- 
	@param w : input vector
	@param lamb : penalty paramemeter
	    
	Returns
	---------- 
	@retval : whether the input is in $\ell_{2}$-ball
	"""
	# norm_w = np.linalg.norm(w, ord=2)
	# if norm_w > lamb:
	# 	return np.inf
	# else:
	# 	return 0.0
	return 0.0

def ind_l1_ball( w, lamb=1 ):
	"""! Compute the indication function of the \f$\ell_1\f$-ball
	
	Parameters
	---------- 
	@param w : input vector
	@param lamb : penalty paramemeter
	    
	Returns
	---------- 
	@retval : whether the input is in $\ell_1$-ball
	"""
	# norm_w = np.linalg.norm(w, ord=1)
	# if norm_w > lamb:
	# 	return np.inf
	# else:
	# 	return 0.0
	return 0.0

def ind_linf_ball( w, lamb=1 ):
	"""! Compute the indication function of the \f$\ell_{\infty}\f$-ball
	
	Parameters
	---------- 
	@param w : input vector
	@param lamb : penalty paramemeter
	    
	Returns
	---------- 
	@retval : whether the input is in $\ell_{\infty}$-ball
	"""
	# norm_w = np.linalg.norm(w, ord=np.inf)
	# if norm_w > lamb:
	# 	return np.inf
	# else:
	# 	return 0.0
	return 0.0

def func_val_l1_norm( w, lamb=1 ):
	"""! Compute \f$\ell_1\f$ - norm of a vector

	Parameters
	---------- 
	@param w : input vector

	Returns
	---------- 
	@retval : \f$ \|w\|_1 \f$
	"""
	return lamb * np.linalg.norm( w,ord = 1 )

def func_val_l2_norm( w, lamb=1 ):
	"""! Compute \f$\ell_1\f$ - norm of a vector

	Parameters
	---------- 
	@param w : input vector

	Returns
	---------- 
	@retval : \f$ \|w\|_1 \f$
	"""
	return lamb * np.linalg.norm( w,ord = 2 )

def func_val_linf_norm( w, lamb=1 ):
	"""! Compute \f$\ell_1\f$ - norm of a vector

	Parameters
	---------- 
	@param w : input vector

	Returns
	---------- 
	@retval : \f$ \|w\|_1 \f$
	"""
	return lamb * np.linalg.norm( w, ord=np.inf )

def func_val_huber( w, lamb=1.0, dt=1.0 ):
	"""! Compute huber loss of a vector

	Parameters
	---------- 
	@param w : input vector

	Returns
	---------- 
	@retval : huber loss
	"""
	abs_w = np.abs(w)

	return np.sum((abs_w <= dt)*0.5*w*w + (abs_w > dt) * dt*(abs_w - 0.5*dt))

def grad_eval_huber( w, lamb=1.0, dt=1.0 ):
	"""! Compute gradient of huber loss of a vector

	Parameters
	---------- 
	@param w : input vector

	Returns
	---------- 
	@retval : grad huber loss
	"""
	abs_w = np.abs(w)

	return (abs_w <= dt)*w + (abs_w > dt) * dt*np.sign(w)

def prox_huber(w, lamb=1.0, dt=1.0):
	"""! Compute proximal operator of huber loss

	Parameters
	---------- 
	@param w : input vector
	@param lamd : penalty param

	Returns
	---------- 
	@retval : proximal operator of huber loss
	"""
	abs_w = np.abs(w)
	
	return (abs_w <= dt)*(1.0/(1.0 + lamb))*w + (abs_w > dt) * prox_l1_norm(w,dt*lamb)

def func_val_huber_conj( w, lamb=1.0, dt=1.0 ):
	"""! Compute conjugate of huber loss

	Parameters
	---------- 
	@param w : input vector
	@param lamd : penalty param
	@param dt : delta

	Returns
	---------- 
	@retval : conjugate of huber function
	"""
	abs_w = np.abs(w)

	return np.sum((abs_w <= dt)*0.5*w*w + (abs_w > dt)*0.5*dt*dt )

###################################################################

def func_val_bin_class_loss_1( n, d, b, X, Y, bias, w, lamb = None, XYw_bias = None, nnzX = None,  index=None):
	"""! Compute the objective value of loss function 1

	\f$\ell_1( Y( Xw + b )) := 1 - \tanh( \omega Y( Xw + b )) \f$

	for a given \f$ \omega > 0\f$. Here \f$ \omega = 1\f$.

	Parameters
	---------- 
	@param n : sample size
	@param d : number of features
	@param b : mini - batch size

		b = 1: single stochastic gradient

		1 < b < n: mini - batch stochastic gradient

		b = n: full gradient
	@param X : input data
	@param Y : input label
	@param bias : input bias
	@param w : input vector
	@param lamb: penalty parameters
	@param XYw_bias : precomputed Y(Xw + b) if available
	@param nnzX : average number of non - zero elements for each sample
	@param index : index set for mini-batch calculation

	Returns
	---------- 
	@retval : \f$\ell_1( Y( Xw + b ))\f$
	"""

	omega = 1.0

	if b == 1:
		# get a random sample
		if index is None:
			index = np.random.randint( 0, n )

		Xi = X[index,:]
		expt = np.exp( 2.0 * omega * Y[i] * ( Xi.dot( w ) + bias[i] ))
		return ( 1.0/float( b )) * np.sum( 2.0 / ( expt + 1.0 ))
	# mini-batch
	elif b < n:
		# get a random batch of size b
		if index is None:
			index = random.sample( range( n ), b )

		# calculate number of batches
		if nnzX is None:
			nnzX = d
		batch_size = np.maximum( int( total_mem_full // nnzX ), 1 )
		num_batches = math.ceil( b / batch_size )
		batch_loss = 0.0

		for j in range( num_batches ): 
			# calculate start/end indices for each batch
			startIdx = batch_size * j
			endIdx = np.minimum( batch_size * ( j + 1 ), b - 1 )

			batch_X = X[index[startIdx:endIdx],:]
			batch_Y = Y[index[startIdx:endIdx]]
			batch_bias = bias[index[startIdx:endIdx]]

			expt = np.exp( 2.0 * omega * batch_Y * ( batch_X.dot( w ) + batch_bias ))

			batch_loss +=  np.sum( 2.0 / ( expt + 1.0 ))

		return batch_loss / float( b )
	# full 
	else:
		if XYw_bias is not None:
			expt = np.exp( 2.0 * omega * XYw_bias )
			return ( 1.0/float( n )) * np.sum( 2.0 / ( expt + 1.0 ))
		else:
			# calculate number of batches
			if nnzX is None:
				nnzX = d
			batch_size = np.maximum( int( total_mem_full // nnzX ), 1 )
			num_batches = math.ceil( n / batch_size )
			full_loss = 0.0

			for j in range( num_batches ): 
				# calculate start/end indices for each batch
				startIdx = batch_size * j
				endIdx = np.minimum( batch_size * ( j + 1 ), n - 1 )

				batch_X = X[startIdx:endIdx,:]
				batch_Y = Y[startIdx:endIdx]
				batch_bias = bias[startIdx:endIdx]

				expt = np.exp( 2.0 * omega * batch_Y * ( batch_X.dot( w ) + batch_bias ))

				full_loss +=  np.sum( 2.0 / ( expt + 1.0 ))

			return full_loss / float( n )

def func_diff_eval_bin_class_loss_1( n, d, b, X, Y, bias, w1, w2, lamb = None, XYw_bias = None, nnzX = None, index=None ):
	"""! Compute the objective value of loss function 1,

	\f$\ell_1( Y( Xw2 + b )) - \f$\ell_1( Y( Xw1 + b )) \f$

	for a given \f$ \omega > 0\f$. Here \f$ \omega = 1\f$.

	Parameters
	---------- 
	@param n : sample size
	@param d : number of features
	@param b : mini - batch size

		b = 1: single stochastic gradient

		1 < b < n: mini - batch stochastic gradient

		b = n: full gradient
	@param X : input data
	@param Y : input label
	@param bias : input bias
	@param w1 : 1st input vector
	@param w1 : 2nd input vector
	@param lamb : penalty parameters
	@param XYw_bias : precomputed Y(Xw + b) if available
	@param nnzX : average number of non - zero elements for each sample
	@param index : index set for mini-batch calculation

	Returns
	---------- 
	@retval : \f$\ell_1( Y( Xw + b ))\f$
	"""

	omega = 1.0

	if b == 1:
		# get a random sample
		i = np.random.randint( 0, n )

		Xi = X[i,:]
		expt1 = np.exp( 2.0 * omega * Y[i] * ( Xi.dot( w1 ) + bias[i] ))
		expt2 = np.exp( 2.0 * omega * Y[i] * ( Xi.dot( w2 ) + bias[i] ))

		return ( 1.0/float( b )) * np.sum( 2.0 / ( expt2 + 1.0 ) - 2.0 / ( expt1 + 1.0 ))
	# mini-batch
	elif b < n:
		# get a random batch of size b
		index = random.sample( range( n ), b )

		# calculate number of batches
		if nnzX is None:
			nnzX = d
		batch_size = np.maximum( int( total_mem_full // nnzX ), 1 )
		num_batches = math.ceil( b / batch_size )
		batch_loss_diff = 0.0

		for j in range( num_batches ): 
			# calculate start/end indices for each batch
			startIdx = batch_size * j
			endIdx = np.minimum( batch_size * ( j + 1 ), b - 1 )

			batch_X = X[index[startIdx:endIdx],:]
			batch_Y = Y[index[startIdx:endIdx]]
			batch_bias = bias[index[startIdx:endIdx]]

			expt1 = np.exp( 2.0 * omega * batch_Y * ( batch_X.dot( w1 ) + batch_bias ))
			expt2 = np.exp( 2.0 * omega * batch_Y * ( batch_X.dot( w2 ) + batch_bias ))

			batch_loss_diff +=  np.sum( 2.0 / ( expt2 + 1.0 ) - 2.0 / ( expt1 + 1.0 ))

		return batch_loss_diff / float( b )
	
	# full	 
	else:
		if XYw_bias is not None:
			expt = np.exp( 2.0 * omega * XYw_bias )
			return ( 1.0/float( b )) * np.sum( 2.0 / ( expt + 1.0 ))
		else:
			# calculate number of batches
			if nnzX is None:
				nnzX = d
			batch_size = np.maximum( int( total_mem_full // nnzX ), 1 )
			num_batches = math.ceil( n / batch_size )
			full_loss_diff = 0.0

			for j in range( num_batches ): 
				# calculate start/end indices for each batch
				startIdx = batch_size * j
				endIdx = np.minimum( batch_size * ( j + 1 ), n - 1 )

				batch_X = X[startIdx:endIdx,:]
				batch_Y = Y[startIdx:endIdx]
				batch_bias = bias[startIdx:endIdx]

				expt1 = np.exp( 2.0 * omega * batch_Y * ( batch_X.dot( w1 ) + batch_bias ))
				expt2 = np.exp( 2.0 * omega * batch_Y * ( batch_X.dot( w2 ) + batch_bias ))

				full_loss_diff +=  np.sum( 2.0 / ( expt2 + 1.0 )- 2.0 / ( expt1 + 1.0 ) )

			return full_loss_diff / float( n )

def grad_eval_bin_class_loss_1( n, d, b, X, Y, bias, w, lamb = None, nnzX = None, index=None ):
	"""! Compute the ( full/stochastic ) gradient of loss function 1.

	where \f$\ell_1( Y( Xw + b )) := 1 - \tanh( \omega Y( Xw + b )) \f$

	for a given \f$ \omega > 0\f$. Here \f$ \omega = 0\f$.

	Parameters
	---------- 
	@param n : sample size
	@param d : number of features
	@param b : mini - batch size

		b = 1: single stochastic gradient

		1 < b < n: mini - batch stochastic gradient

		b = n: full gradient
	@param X : input data
	@param Y : input label
	@param bias : input bias
	@param w : input vector
	@param lamb: penalty parameters
	@param nnzX : average number of non - zero elements for each sample
	@param index : index set for mini-batch calculation

	Returns
	---------- 
	@retval : computed full/stochastic gradient

	@retval XYw_bias: The precomputed \f$ Y( Xw + bias )\f$
	"""
	if nnzX is None:
		nnzX = d

	omega = 1.0
	# single sample
	if b == 1:
		# get a random sample
		i = np.random.randint( 0, n )

		Xi = X[i,:]
		expt = np.exp( 2.0 * omega * Y[i] * ( Xi.dot( w ) + bias[i] ))

		return - 4.0 * omega * ( (expt/( expt + 1.0 ))/( expt + 1.0 )) * Y[i] * Xi
	# mini-batch
	elif b < n:
		# get a random batch of size b
		index = random.sample( range( n ), b )

		# calculate number of batches
		if nnzX == 0:
			nnzX = d
		batch_size = np.maximum( int( total_mem_full // nnzX ), 1 )
		num_batches = math.ceil( b / batch_size )
		batch_grad = np.zeros( d )

		for j in range( num_batches ): 
			# calculate start/end indices for each batch
			startIdx = batch_size * j
			endIdx = np.minimum( batch_size * ( j + 1 ), b - 1 )

			batch_X = X[index[startIdx:endIdx],:]
			batch_Y = Y[index[startIdx:endIdx]]
			batch_bias = bias[index[startIdx:endIdx]]

			expt = np.exp( 2.0 * omega * batch_Y * ( batch_X.dot( w ) + batch_bias ))

			batch_grad -= 4.0 * omega * batch_X.transpose().dot( batch_Y * ( (expt/( expt + 1.0 ))/( expt + 1.0 )) ) 

		return batch_grad / float( b )
	# full
	else:
		# calculate number of batches
		if nnzX == 0:
			nnzX = d
		batch_size = np.maximum( int( total_mem_full // nnzX ), 1 )
		num_batches = math.ceil( b / batch_size )
		full_grad = np.zeros( d )
		XYw_bias = np.zeros( n )

		for j in range( num_batches ): 
			# calculate start/end indices for each batch
			startIdx = batch_size * j
			endIdx = np.minimum( batch_size * ( j + 1 ), n - 1 )

			batch_X = X[startIdx:endIdx,:]
			batch_Y = Y[startIdx:endIdx]
			batch_bias = bias[startIdx:endIdx]

			batch_XYw_bias = batch_Y * ( batch_X.dot( w ) + batch_bias )

			XYw_bias[startIdx:endIdx] = batch_XYw_bias

			expt = np.exp( 2.0 * omega * batch_XYw_bias )

			full_grad -= 4.0 * omega * batch_X.transpose().dot( batch_Y * ( (expt/( expt + 1.0 ))/( expt + 1.0 )) ) 

		return full_grad / float( n ), XYw_bias

def grad_diff_eval_bin_class_loss_1( n, d, b, X, Y, bias, w1, w2, lamb = None, nnzX = None, index=None ):
	"""! Compute the ( full/stochastic ) gradient difference of loss function 1

	\f$\displaystyle\frac{1}{b}\left( \sum_{i \in \mathcal{B}_t}( \nabla f_i( w_2 ) - \nabla f_i( w_1 )) \right ) \f$

	Parameters
	---------- 
	@param n : sample size
	@param d : number of features
	@param b : mini - batch size

		b = 1: single stochastic gradient

		1 < b < n: mini - batch stochastic gradient

		b = n: full gradient
	@param X : input data
	@param Y : input label
	@param bias : input bias
	@param w1 : input vector
	@param w2 : input vector
	@param nnzX : average number of non - zero elements for each sample

	Returns
	---------- 
	@retval  : computed full/stochastic gradient
	"""
	if nnzX is None:
		nnzX = d

	omega = 1.0
	# single sample
	if b == 1:
		# get a random sample
		i = np.random.randint( 0,n )

		Xi = X[i, :]
		expt1 = np.exp( 2.0 * omega * Y[i] * ( Xi.dot( w1 ) + bias[i] ))
		expt2 = np.exp( 2.0 * omega * Y[i] * ( Xi.dot( w2 ) + bias[i] ))

		diff_expt = (expt2 / ( expt2 + 1.0 )) / ( expt2 + 1.0 ) - expt1 / ( expt1 + 1.0 ) / ( expt1 + 1.0 )
		
		return - ( 4.0 * omega * diff_expt * Y[i] ) * Xi
	# batch
	elif b < n:
		# get a random batch of size b
		index = random.sample( range( n ), b )

		# calculate number of batches
		if nnzX == 0:
			nnzX = d
		batch_size = np.maximum( int( total_mem_full // nnzX ), 1 )
		num_batches = math.ceil( b / batch_size )
		batch_grad_diff = np.zeros( d )

		for j in range( num_batches ): 
			# calculate start/end indices for each batch
			startIdx = batch_size * j
			endIdx = np.minimum( batch_size * ( j + 1 ), b - 1 )

			batch_X = X[index[startIdx:endIdx],:]
			batch_Y = Y[index[startIdx:endIdx]]
			batch_bias = bias[index[startIdx:endIdx]]

			expt1 = np.exp( 2.0 * omega * batch_Y * ( batch_X.dot( w1 ) + batch_bias ))
			expt2 = np.exp( 2.0 * omega * batch_Y * ( batch_X.dot( w2 ) + batch_bias ))

			diff_expt = (expt2/( expt2 + 1.0 ))/( expt2 + 1.0 ) - (expt1/( expt1 + 1.0 ))/( expt1 + 1.0 )

			batch_grad_diff -= 4.0 * omega * batch_X.transpose().dot( batch_Y * diff_expt )

		return batch_grad_diff / float( b )
	# full
	else:
		# calculate number of batches
		if nnzX == 0:
			nnzX = d
		batch_size = np.maximum( int( total_mem_full // nnzX ), 1 )
		num_batches = math.ceil( b / batch_size )
		full_grad_diff = np.zeros( d )

		for j in range( num_batches ): 
			# calculate start/end indices for each batch
			startIdx = batch_size * j
			endIdx = np.minimum( batch_size * ( j + 1 ), n - 1 )

			batch_X = X[startIdx:endIdx]
			batch_Y = Y[startIdx:endIdx]
			batch_bias = bias[startIdx:endIdx]

			expt1 = np.exp( 2.0 * omega * batch_Y * ( batch_X.dot( w1 ) + batch_bias ))
			expt2 = np.exp( 2.0 * omega * batch_Y * ( batch_X.dot( w2 ) + batch_bias ))

			diff_expt = (expt2/( expt2 + 1.0 ))/( expt2 + 1.0 ) - (expt1/( expt1 + 1.0 ))/( expt1 + 1.0 )

			full_grad_diff -= 4.0 * omega * batch_X.transpose().dot( batch_Y * diff_expt )

		return full_grad_diff / float( n )

######################################################################

def func_val_bin_class_loss_2( n, d, b, X, Y, bias, w, lamb = None, XYw_bias = None, nnzX = None ):
	"""! Compute the objective value of loss function 2

	\f$\ell_2( Y( Xw + b )) := \left( 1 - \frac{1}{1 + \exp[ -Y( Xw + b )]}\right )^2 \f$

	for a given \f$ \omega > 0\f$.

	Parameters
	---------- 
	@param n : sample size
	@param d : number of features
	@param b : mini - batch size

		b = 1: single stochastic gradient

		1 < b < n: mini - batch stochastic gradient

		b = n: full gradient
	@param X : input data
	@param Y : input label
	@param bias : input bias
	@param w : input vector
	@param lamb: penalty parameters
	@param XYw_bias : precomputed Y(Xw + b) if available
	@param nnzX : average number of non - zero elements for each sample
	@param index : index set for mini-batch calculation

	Returns
	---------- 
	@retval  : \f$\ell_2( Y( Xw + b ))\f$
	"""

	if b == 1:
		# get a random sample
		i = np.random.randint( 0, n )

		Xi = X[i,:]
		expt = np.exp( Y[i] * ( Xi.dot( w ) + bias[i] ))
		return np.sum ( (1.0 / ( expt + 1.0 )) / ( expt + 1.0 ) )
	# batch
	elif b < n:
		# get a random batch of size b
		index = random.sample( range( n ), b )

		# calculate number of batches
		if nnzX is None:
			nnzX = d
		batch_size = np.maximum( int( total_mem_full // nnzX ), 1 )
		num_batches = math.ceil( b / batch_size )
		batch_loss = 0.0

		for j in range( num_batches ): 
			# calculate start/end indices for each batch
			startIdx = batch_size * j
			endIdx = np.minimum( batch_size * ( j + 1 ), b - 1 )

			batch_X = X[index[startIdx:endIdx],:]
			batch_Y = Y[index[startIdx:endIdx]]
			batch_bias = bias[index[startIdx:endIdx]]

			expt = np.exp( batch_Y * ( batch_X.dot( w ) + batch_bias ))

			batch_loss +=  np.sum ( (1.0 / ( expt + 1.0 )) / ( expt + 1.0 ) )

		return batch_loss / float( b )
		# batch_X = X[index,:]
		# batch_Y = Y[index]
		# batch_bias = bias[index]

		# expt = np.exp( batch_Y * ( batch_X.dot( w ) + batch_bias ))

		# return ( 1.0/float( b )) * np.sum ( 1.0 / ( expt + 1.0 ) / ( expt + 1.0 ) )
	else:
		if XYw_bias is not None:
			expt = np.exp( XYw_bias )
			return ( 1.0/float( n )) * np.sum ( (1.0 / ( expt + 1.0 )) / ( expt + 1.0 ) )
		else:
			# calculate number of batches
			if nnzX is None:
				nnzX = d
			batch_size = np.maximum( int( total_mem_full // nnzX ), 1 )
			num_batches = math.ceil( n / batch_size )
			full_loss = 0.0

			for j in range( num_batches ): 
				# calculate start/end indices for each batch
				startIdx = batch_size * j
				endIdx = np.minimum( batch_size * ( j + 1 ), n - 1 )

				batch_X = X[startIdx:endIdx,:]
				batch_Y = Y[startIdx:endIdx]
				batch_bias = bias[startIdx:endIdx]

				expt = np.exp( batch_Y * ( batch_X.dot( w ) + batch_bias ))

				full_loss +=  np.sum ( (1.0 / ( expt + 1.0 )) / ( expt + 1.0 ) )

			return full_loss / float( n )

def func_diff_eval_bin_class_loss_2( n, d, b, X, Y, bias, w1, w2, lamb = None, XYw_bias = None, nnzX = None ):
	"""! Compute the objective value of loss function 2

	\f$\ell_2( Y( Xw + b )) := \left( 1 - \frac{1}{1 + \exp[ -Y( Xw + b )]}\right )^2 \f$

	for a given \f$ \omega > 0\f$.

	Parameters
	---------- 
	@param n : sample size
	@param d : number of features
	@param b : mini - batch size

		b = 1: single stochastic gradient

		1 < b < n: mini - batch stochastic gradient

		b = n: full gradient
	@param X : input data
	@param Y : input label
	@param bias : input bias
	@param w1 : 1st input vector
	@param w2 : 2nd input vector
	@param lamb: penalty parameters
	@param XYw_bias : precomputed Y(Xw + b) if available
	@param nnzX : average number of non - zero elements for each sample
	@param index : index set for mini-batch calculation

	Returns
	---------- 
	@retval  : \f$\ell_2( Y( Xw + b ))\f$
	"""

	if b == 1:
		# get a random sample
		i = np.random.randint( 0, n )

		Xi = X[i,:]
		expt1 = np.exp( Y[i] * ( Xi.dot( w1 ) + bias[i] ))
		expt2 = np.exp( Y[i] * ( Xi.dot( w2 ) + bias[i] ))

		return np.sum ( (1.0 / ( expt2 + 1.0 )) / ( expt2 + 1.0 ) - (1.0 / ( expt1 + 1.0 )) / ( expt1 + 1.0 ) )
	# batch
	elif b < n:
		# get a random batch of size b
		index = random.sample( range( n ), b )

		# calculate number of batches
		if nnzX is None:
			nnzX = d
		batch_size = np.maximum( int( total_mem_full // nnzX ), 1 )
		num_batches = math.ceil( b / batch_size )
		batch_loss = 0.0

		for j in range( num_batches ): 
			# calculate start/end indices for each batch
			startIdx = batch_size * j
			endIdx = np.minimum( batch_size * ( j + 1 ), b - 1 )

			batch_X = X[index[startIdx:endIdx],:]
			batch_Y = Y[index[startIdx:endIdx]]
			batch_bias = bias[index[startIdx:endIdx]]

			expt1 = np.exp( batch_Y * ( batch_X.dot( w1 ) + batch_bias ))
			expt2 = np.exp( batch_Y * ( batch_X.dot( w2 ) + batch_bias ))

			batch_loss +=  np.sum ( (1.0 / ( expt2 + 1.0 )) / ( expt2 + 1.0 ) - (1.0 / ( expt1 + 1.0 )) / ( expt1 + 1.0 ) )

		return batch_loss / float( b )
		# batch_X = X[index,:]
		# batch_Y = Y[index]
		# batch_bias = bias[index]

		# expt = np.exp( batch_Y * ( batch_X.dot( w ) + batch_bias ))

		# return ( 1.0/float( b )) * np.sum ( 1.0 / ( expt + 1.0 ) / ( expt + 1.0 ) )
	else:
		if XYw_bias is not None:
			expt = np.exp( XYw_bias )
			return ( 1.0/float( n )) * np.sum ( (1.0 / ( expt + 1.0 )) / ( expt + 1.0 ) )
		else:
			# calculate number of batches
			if nnzX is None:
				nnzX = d
			batch_size = np.maximum( int( total_mem_full // nnzX ), 1 )
			num_batches = math.ceil( n / batch_size )
			full_loss = 0.0

			for j in range( num_batches ): 
				# calculate start/end indices for each batch
				startIdx = batch_size * j
				endIdx = np.minimum( batch_size * ( j + 1 ), n - 1 )

				batch_X = X[startIdx:endIdx,:]
				batch_Y = Y[startIdx:endIdx]
				batch_bias = bias[startIdx:endIdx]

				expt1 = np.exp( batch_Y * ( batch_X.dot( w1 ) + batch_bias ))
				expt2 = np.exp( batch_Y * ( batch_X.dot( w2 ) + batch_bias ))

				full_loss +=  np.sum ( (1.0 / ( expt2 + 1.0 )) / ( expt2 + 1.0 ) - (1.0 / ( expt1 + 1.0 )) / ( expt1 + 1.0 ) )

			return full_loss / float( n )

def grad_eval_bin_class_loss_2( n, d, b, X, Y, bias, w, lamb = None, nnzX = None ):
	"""! Compute the ( full/stochastic ) gradient of loss function 2.

	\f$\ell_2( Y( Xw + b )) := \left( 1 - \frac{1}{1 + \exp[ -Y( Xw + b )]}\right )^2 \f$

	Parameters
	---------- 
	@param n : sample size
	@param d : number of features
	@param b : mini - batch size

		b = 1: single stochastic gradient

		1 < b < n: mini - batch stochastic gradient

		b = n: full gradient
	@param X : input data
	@param Y : input label
	@param bias : input bias
	@param w : input vector
	@param lamb: penalty parameters
	@param nnzX : average number of non - zero elements for each sample
	@param index : index set for mini-batch calculation

	Returns
	---------- 
	@retval  : computed full/stochastic gradient

	@retval XYw_bias: The precomputed \f$ Y( Xw + bias )\f$
	"""
	# single sample
	if nnzX is None:
		nnzX = d
	
	if b == 1:
		# get a random sample
		i = np.random.randint( 0, n )
		
		Xi = X[i, :]
		expt = np.exp( Y[i] * ( Xi.dot( w ) + bias[i] ))
		
		return ( - 2.0 * ( ((expt/( 1.0 + expt ))/( 1.0 + expt ))/( 1.0 + expt )) * Y[i] ) * Xi
	# batch
	elif b < n:
		# get a random batch of size b
		index = random.sample( range( n ), b )

		# calculate number of batches
		if nnzX == 0:
			nnzX = d
		batch_size = np.maximum( int( total_mem_full // nnzX ), 1 )
		num_batches = math.ceil( b / batch_size )
		batch_grad = np.zeros( d )

		for j in range( num_batches ): 
			# calculate start/end indices for each batch
			startIdx = batch_size * j
			endIdx = np.minimum( batch_size * ( j + 1 ), b - 1 )

			batch_X = X[index[startIdx:endIdx],:]
			batch_Y = Y[index[startIdx:endIdx]]
			batch_bias = bias[index[startIdx:endIdx]]

			expt = np.exp( batch_Y * ( batch_X.dot( w ) + batch_bias ))

			batch_grad -= 2.0 * batch_X.transpose().dot( batch_Y \
											 * ( ((expt/( 1.0 + expt ))/( 1.0 + expt ))/( 1.0 + expt )) )
        
		return batch_grad / float( b )
	# full
	else:
		# calculate number of batches
		if nnzX == 0:
			nnzX = d
		batch_size = np.maximum( int( total_mem_full // nnzX ), 1 )
		num_batches = math.ceil( b / batch_size )
		full_grad = np.zeros( d )
		XYw_bias = np.zeros( n )

		for j in range( num_batches ): 
			# calculate start/end indices for each batch
			startIdx = batch_size * j
			endIdx = np.minimum( batch_size * ( j + 1 ), n - 1 )

			batch_X = X[startIdx:endIdx]
			batch_Y = Y[startIdx:endIdx]
			batch_bias = bias[startIdx:endIdx]

			batch_XYw_bias = batch_Y * ( batch_X.dot( w ) + batch_bias )

			XYw_bias[startIdx:endIdx] = batch_XYw_bias

			expt = np.exp( batch_XYw_bias )

			full_grad -= 2.0 * batch_X.transpose().dot( batch_Y * ( ((expt/( expt + 1 ))/( expt + 1 ))/( expt + 1 )) )

		return full_grad / float( n ), XYw_bias

def grad_diff_eval_bin_class_loss_2( n, d, b, X, Y, bias, w1, w2, lamb = None, nnzX = None ):
	"""! Compute the ( full/stochastic ) gradient difference of loss function 2

	\f$\displaystyle\frac{1}{b}\left( \sum_{i \in \mathcal{B}_t}( \nabla f_i( w_2 ) - \nabla f_i( w_1 )) \right ) \f$

	Parameters
	---------- 
	@param n : sample size
	@param d : number of features
	@param b : mini - batch size

		b = 1: single stochastic gradient

		1 < b < n: mini - batch stochastic gradient

		b = n: full gradient
	@param X : input data
	@param Y : input label
	@param bias : input bias
	@param w1 : 1st input vector
	@param w2 : 2nd input vector
	@param lamb: penalty parameters
	@param nnzX : average number of non - zero elements for each sample
	@param index : index set for mini-batch calculation

	Returns
	---------- 
	@retval  : computed full/stochastic gradient
	"""
	# single sample
	if nnzX is None:
		nnzX = d
	
	if b == 1:
		# get a random sample
		i = np.random.randint( 0, n )
		
		Xi = X[i,:]
		expt1 = np.exp( Y[i] * ( Xi.dot( w1 ) + bias[i] ))
		expt2 = np.exp( Y[i] * ( Xi.dot( w2 ) + bias[i] ))

		diff_expt = ( ((expt2/( 1.0 + expt2 ))/( 1.0 + expt2 ))/( 1.0 + expt2 )) - ( ((expt1/( 1.0 + expt1 ))/( 1.0 + expt1 ))/( 1.0 + expt1 ))
		
		return - 2.0 * diff_expt * Y[i] * Xi
	# batch
	elif b < n:
		# get a random batch of size b
		index = random.sample( range( n ), b )

		# calculate number of batches
		if nnzX == 0:
			nnzX = d
		batch_size = np.maximum( int( total_mem_full // nnzX ), 1 )
		num_batches = math.ceil( b / batch_size )
		batch_grad_diff = np.zeros( d )

		for j in range( num_batches ): 
			# calculate start/end indices for each batch
			startIdx = batch_size * j
			endIdx = np.minimum( batch_size * ( j + 1 ), b - 1 )

			batch_X = X[index[startIdx:endIdx],:]
			batch_Y = Y[index[startIdx:endIdx]]
			batch_bias = bias[index[startIdx:endIdx]]

			expt1 = np.exp( batch_Y * ( batch_X.dot( w1 ) + batch_bias ))
			expt2 = np.exp( batch_Y * ( batch_X.dot( w2 ) + batch_bias ))

			diff_expt = ((expt2/( 1.0 + expt2 ))/( 1.0 + expt2 ))/( 1.0 + expt2 ) - ((expt1/( 1.0 + expt1 ))/( 1.0 + expt1 ))/( 1.0 + expt1 )
		
			batch_grad_diff -= 2.0 * batch_X.transpose().dot( batch_Y * diff_expt )

		return batch_grad_diff / float( b )
	# full
	else:
		# calculate number of batches
		if nnzX == 0:
			nnzX = d
		batch_size = np.maximum( int( total_mem_full // nnzX ), 1 )
		num_batches = math.ceil( b / batch_size )
		full_grad_diff = np.zeros( d )

		for j in range( num_batches ): 
			# calculate start/end indices for each batch
			startIdx = batch_size * j
			endIdx = np.minimum( batch_size * ( j + 1 ), n - 1 )

			batch_X = X[startIdx:endIdx]
			batch_Y = Y[startIdx:endIdx]
			batch_bias = bias[startIdx:endIdx]

			expt1 = np.exp( batch_Y * ( batch_X.dot( w1 ) + batch_bias ))
			expt2 = np.exp( batch_Y * ( batch_X.dot( w2 ) + batch_bias ))

			diff_expt = ((expt2/( 1.0 + expt2 ))/( 1.0 + expt2 ))/( 1.0 + expt2 ) - ((expt1/( 1.0 + expt1 ))/( 1.0 + expt1 ))/( 1.0 + expt1 )

			full_grad_diff -= 2.0 * batch_X.transpose().dot( batch_Y * diff_expt )

		return full_grad_diff / float( n )

##################################################################

def func_val_bin_class_loss_3( n, d, b, X, Y, bias, w, lamb = None, XYw_bias = None, nnzX = None ):
	"""! Compute the objective value of loss function 3

	\f$ \ell_3( Y( Xw + b )) := \ln( 1 + \exp( -Y( Xw + b )) ) - \ln( 1 + \exp( -Y( Xw + b ) - \alpha ))\f$

	for a given \f$ \alpha > 0\f$.

	Parameters
	---------- 
	@param n : sample size
	@param d : number of features
	@param b : mini - batch size

		b = 1: single stochastic gradient

		1 < b < n: mini - batch stochastic gradient

		b = n: full gradient
	@param X : input data
	@param Y : input label
	@param bias : input bias
	@param w : input vector
	@param lamb: penalty parameters
	@param XYw_bias : precomputed Y(Xw + b) if available
	@param nnzX : average number of non - zero elements for each sample
	@param index : index set for mini-batch calculation

	Returns
	---------- 
	@retval : \f$\ell_3( Y( Xw + b ))\f$
	"""

	alpha = 1.0
	exp_a = np.exp( - alpha )
	
	if b == 1:
		# get a random sample
		i = np.random.randint( 0, n )

		Xi = X[i,:]
		expt = np.exp( -Y[i] * ( Xi.dot( w ) + bias[i] ))

		return  np.sum( np.log( 1.0 + expt ) - np.log( 1.0 + exp_a * expt ) )
	# batch
	elif b < n:
		# get a random batch of size b
		index = random.sample( range( n ), b )

		# calculate number of batches
		if nnzX is None:
			nnzX = d
		batch_size = np.maximum( int( total_mem_full // nnzX ), 1 )
		num_batches = math.ceil( b / batch_size )
		batch_loss = 0.0

		for j in range( num_batches ): 
			# calculate start/end indices for each batch
			startIdx = batch_size * j
			endIdx = np.minimum( batch_size * ( j + 1 ), b - 1 )

			batch_X = X[index[startIdx:endIdx],:]
			batch_Y = Y[index[startIdx:endIdx]]
			batch_bias = bias[index[startIdx:endIdx]]

			expt = np.exp( -batch_Y * ( batch_X.dot( w ) + batch_bias ))

			batch_loss +=  np.sum( np.log( 1.0 + expt ) - np.log( 1.0 + exp_a * expt ) )

		return batch_loss / float( b )

	else:
		if XYw_bias is not None:
			expt = np.exp( - XYw_bias )
			return ( 1.0 / float( n )) * np.sum(( np.log( 1.0 + expt ) - np.log( 1.0 + exp_a * expt )) )
		else:
			# calculate number of batches
			if nnzX is None:
				nnzX = d
			batch_size = np.maximum( int( total_mem_full // nnzX ), 1 )
			num_batches = math.ceil( n / batch_size )
			full_loss = 0.0

			for j in range( num_batches ): 
				# calculate start/end indices for each batch
				startIdx = batch_size * j
				endIdx = np.minimum( batch_size * ( j + 1 ), n - 1 )

				batch_X = X[startIdx:endIdx,:]
				batch_Y = Y[startIdx:endIdx]
				batch_bias = bias[startIdx:endIdx]

				expt = np.exp( -batch_Y * ( batch_X.dot( w ) + batch_bias ))

				full_loss +=  np.sum( np.log( 1.0 + expt ) - np.log( 1.0 + exp_a * expt ) )

			return full_loss / float( n )

def func_diff_eval_bin_class_loss_3( n, d, b, X, Y, bias, w1, w2, lamb = None, XYw_bias = None, nnzX = None ):
	"""! Compute the objective value of loss function 3

	\f$ \ell_3( Y( Xw2 + b )) - \ell_3( Y( Xw1 + b )) \f$

	Parameters
	---------- 
	@param n : sample size
	@param d : number of features
	@param b : mini - batch size

		b = 1: single stochastic gradient

		1 < b < n: mini - batch stochastic gradient

		b = n: full gradient
	@param X : input data
	@param Y : input label
	@param bias : input bias
	@param w1 : 1st input vector
	@param w2 : 2nd input vector
	@param lamb: penalty parameters
	@param XYw_bias : precomputed Y(Xw + b) if available
	@param nnzX : average number of non - zero elements for each sample
	@param index : index set for mini-batch calculation

	Returns
	---------- 
	"""

	alpha = 1.0
	exp_a = np.exp( - alpha )
	
	if b == 1:
		# get a random sample
		i = np.random.randint( 0, n )

		Xi = X[i,:]
		expt1 = np.exp( -Y[i] * ( Xi.dot( w1 ) + bias[i] ))
		expt2 = np.exp( -Y[i] * ( Xi.dot( w2 ) + bias[i] ))

		return  np.sum( (np.log( 1.0 + expt2 ) - np.log( 1.0 + exp_a * expt2 )) - (np.log( 1.0 + expt1 ) - np.log( 1.0 + exp_a * expt1 ) ) )
	# batch
	elif b < n:
		# get a random batch of size b
		index = random.sample( range( n ), b )

		# calculate number of batches
		if nnzX is None:
			nnzX = d
		batch_size = np.maximum( int( total_mem_full // nnzX ), 1 )
		num_batches = math.ceil( b / batch_size )
		batch_loss_diff = 0.0

		for j in range( num_batches ): 
			# calculate start/end indices for each batch
			startIdx = batch_size * j
			endIdx = np.minimum( batch_size * ( j + 1 ), b - 1 )

			batch_X = X[index[startIdx:endIdx],:]
			batch_Y = Y[index[startIdx:endIdx]]
			batch_bias = bias[index[startIdx:endIdx]]

			expt1 = np.exp( -batch_Y * ( batch_X.dot( w1 ) + batch_bias ))
			expt2 = np.exp( -batch_Y * ( batch_X.dot( w2 ) + batch_bias ))

			batch_loss_diff +=  np.sum( (np.log( 1.0 + expt2 ) - np.log( 1.0 + exp_a * expt2 )) - (np.log( 1.0 + expt1 ) - np.log( 1.0 + exp_a * expt1 )) )

		return batch_loss_diff / float( b )

	else:
		if XYw_bias is not None:
			expt = np.exp( - XYw_bias )
			return ( 1.0 / float( n )) * np.sum(( np.log( 1.0 + expt ) - np.log( 1.0 + exp_a * expt )) )
		else:
			# calculate number of batches
			if nnzX is None:
				nnzX = d
			batch_size = np.maximum( int( total_mem_full // nnzX ), 1 )
			num_batches = math.ceil( n / batch_size )
			full_loss_diff = 0.0

			for j in range( num_batches ): 
				# calculate start/end indices for each batch
				startIdx = batch_size * j
				endIdx = np.minimum( batch_size * ( j + 1 ), n - 1 )

				batch_X = X[startIdx:endIdx,:]
				batch_Y = Y[startIdx:endIdx]
				batch_bias = bias[startIdx:endIdx]

				expt1 = np.exp( -batch_Y * ( batch_X.dot( w1 ) + batch_bias ))
				expt2 = np.exp( -batch_Y * ( batch_X.dot( w2 ) + batch_bias ))

				full_loss_diff +=  np.sum( (np.log( 1.0 + expt2 ) - np.log( 1.0 + exp_a * expt2 )) - (np.log( 1.0 + expt1 ) - np.log( 1.0 + exp_a * expt1 )) )

			return full_loss_diff / float( n )

def grad_eval_bin_class_loss_3( n, d, b, X, Y, bias, w, lamb = None, nnzX = None ):
	"""! Compute the ( full/stochastic ) gradient of loss function 3.

	where \f$ \ell_3( Y( Xw + b )) := \ln( 1 + \exp( -Y( Xw + b )) ) - \ln( 1 + \exp( -Y( Xw + b ) - \omega ))\f$

	for a given \f$ \omega > 0\f$.

	Parameters
	---------- 
	@param n : sample size
	@param d : number of features
	@param b : mini - batch size

		b = 1: single stochastic gradient

		1 < b < n: mini - batch stochastic gradient

		b = n: full gradient
	@param X : input data
	@param Y : input label
	@param bias : input bias
	@param w : input vector
	@param lamb: penalty parameters
	@param nnzX : average number of non - zero elements for each sample
	@param index : index set for mini-batch calculation

	Returns
	---------- 
	@retval  : computed full/stochastic gradient

	@retval XYw_bias: The precomputed \f$ Y( Xw + bias )\f$
	"""
	if nnzX is None:
		nnzX = d
	
	alpha = 1
	exp_a = np.exp( alpha )
	# single sample
	if b == 1:
		# get a random sample
		i = np.random.randint( 0, n )

		Xi = X[i, :]
		expt = np.exp( Y[i] * ( Xi.dot( w ) + bias[i] ))

		return (( 1 / ( expt * exp_a + 1.0 ) - 1 / ( expt + 1.0 )) * Y[i] ) * Xi
	# batch
	elif b < n:
		# get a random batch of size b
		index = random.sample( range( n ), b )

		# calculate number of batches
		if nnzX == 0:
			nnzX = d
		batch_size = np.maximum( int( total_mem_full // nnzX ), 1 )
		num_batches = math.ceil( b / batch_size )
		batch_grad = np.zeros( d )

		for j in range( num_batches ): 
			# calculate start/end indices for each batch
			startIdx = batch_size * j
			endIdx = np.minimum( batch_size * ( j + 1 ), b - 1 )

			batch_X = X[index[startIdx:endIdx],:]
			batch_Y = Y[index[startIdx:endIdx]]
			batch_bias = bias[index[startIdx:endIdx]]

			expt = np.exp( batch_Y * ( batch_X.dot( w ) + batch_bias ))

			batch_grad += batch_X.transpose().dot( batch_Y * ( 1 / ( expt * exp_a + 1.0 ) - 1 / ( expt + 1.0 )) )

		return batch_grad / float( b )
	# full
	else:
		# calculate number of batches
		if nnzX == 0:
			nnzX = d
		batch_size = np.maximum( int( total_mem_full // nnzX ), 1 )
		num_batches = math.ceil( b / batch_size )
		full_grad = np.zeros( d )
		XYw_bias = np.zeros( n )

		for j in range( num_batches ): 
			# calculate start/end indices for each batch
			startIdx = batch_size * j
			endIdx = np.minimum( batch_size * ( j + 1 ), n - 1 )

			batch_X = X[startIdx:endIdx,:]
			batch_Y = Y[startIdx:endIdx]
			batch_bias = bias[startIdx:endIdx]

			batch_XYw_bias = batch_Y * ( batch_X.dot( w ) + batch_bias )

			XYw_bias[startIdx:endIdx] = batch_XYw_bias

			expt = np.exp( batch_XYw_bias )

			full_grad += batch_X.transpose().dot( batch_Y * ( 1.0/ ( expt * exp_a + 1.0 ) - 1.0/ ( expt + 1.0 )) )

		return full_grad / float( n ), XYw_bias
		
def grad_diff_eval_bin_class_loss_3( n, d, b, X, Y, bias, w1, w2, lamb = None, nnzX = None ):
	"""! Compute the ( full/stochastic ) gradient difference of loss function 3

	\f$\displaystyle\frac{1}{b}\left( \sum_{i \in \mathcal{B}_t}( \nabla f_i( w_2 ) - \nabla f_i( w_1 )) \right ) \f$

	Parameters
	---------- 
	@param n : sample size
	@param d : number of features
	@param b : mini - batch size

		b = 1: single stochastic gradient

		1 < b < n: mini - batch stochastic gradient

		b = n: full gradient
	@param X : input data
	@param Y : input label
	@param bias : input bias
	@param w1 : 1st input vector
	@param w2 : 2nd input vector
	@param lamb: penalty parameters
	@param nnzX : average number of non - zero elements for each sample
	@param index : index set for mini-batch calculation

	Returns
	---------- 
	@retval  : computed full/stochastic gradient
	"""
	if nnzX is None:
		nnzX = d
	
	alpha = 1
	exp_a = np.exp( alpha )
	# single sample
	if b == 1:
		# get a random sample
		i = np.random.randint( 0, n )

		Xi = X[i,:]
		expt1 = np.exp( Y[i] * ( Xi.dot( w1 ) + bias[i] ))
		expt2 = np.exp( Y[i] * ( Xi.dot( w2 ) + bias[i] ))

		diff_expt = ( 1.0/( expt2 * exp_a + 1.0 ) - 1.0/( expt2 + 1.0 )) - ( 1.0/( expt1 * exp_a + 1.0 ) - 1.0/( expt1 + 1.0 ))
		
		return diff_expt * Y[i] * Xi
	# batch
	elif b < n:
		# get a random batch of size b
		index = random.sample( range( n ), b )

		# calculate number of batches
		if nnzX == 0:
			nnzX = d
		batch_size = int( total_mem_batch // nnzX )
		num_batches = math.ceil( b / batch_size )
		batch_grad_diff = np.zeros( d )

		for j in range( num_batches ): 
			# calculate start/end indices for each batch
			startIdx = batch_size * j
			endIdx = np.minimum( batch_size * ( j + 1 ), b - 1 )

			batch_X = X[index[startIdx:endIdx],:]
			batch_Y = Y[index[startIdx:endIdx]]
			batch_bias = bias[index[startIdx:endIdx]]

			expt1 = np.exp( batch_Y * ( batch_X.dot( w1 ) + batch_bias ))
			expt2 = np.exp( batch_Y * ( batch_X.dot( w2 ) + batch_bias ))

			diff_expt = ( 1/( expt2 * exp_a + 1.0 ) - 1/( expt2 + 1.0 )) - ( 1/( expt1 * exp_a + 1.0 ) - 1/( expt1 + 1.0 ))

			batch_grad_diff += batch_X.transpose().dot( batch_Y * diff_expt )

		return batch_grad_diff / float( b )
	# full
	else:
		# calculate number of batches
		if nnzX == 0:
			nnzX = d
		batch_size = int( total_mem_full // nnzX )
		num_batches = math.ceil( b / batch_size )
		full_grad_diff = np.zeros( d )

		for j in range( num_batches ): 
			# calculate start/end indices for each batch
			startIdx = batch_size * j
			endIdx = np.minimum( batch_size * ( j + 1 ), n - 1 )

			batch_X = X[startIdx:endIdx]
			batch_Y = Y[startIdx:endIdx]
			batch_bias = bias[startIdx:endIdx]

			expt1 = np.exp( batch_Y * ( batch_X.dot( w1 ) + batch_bias ))
			expt2 = np.exp( batch_Y * ( batch_X.dot( w2 ) + batch_bias ))

			diff_expt = ( 1/( expt2 * exp_a + 1.0 ) - 1/( expt2 + 1.0 )) - ( 1/( expt1 * exp_a + 1.0 ) - 1/( expt1 + 1.0 ))

			full_grad_diff += batch_X.transpose().dot( batch_Y * diff_expt )

		return full_grad_diff / float( n )

##################################################################

def func_val_bin_class_loss_4( n, d, b, X, Y, bias, w, lamb = None, XYw_bias = None, nnzX = None ):
	"""! Compute the objective value of loss function 4

	\f$ \ell_4^{(i)}( Y_i( X_i^Tw + b )) := \begin{cases}	
			0,~&\text{if }Y_i( X_i^Tw + b ) > 1\\
			\ln(1 + (Y_i( X_i^Tw + b ) - 1)^2),&\text{otherwise}
	 \end{cases}\f$

	for a given \f$ \omega > 0\f$.

	Parameters
	---------- 
	@param n : sample size
	@param d : number of features
	@param b : mini - batch size

		b = 1: single stochastic gradient

		1 < b < n: mini - batch stochastic gradient

		b = n: full gradient
	@param X : input data
	@param Y : input label
	@param bias : input bias
	@param w : input vector
	@param lamb: penalty parameters
	@param XYw_bias : precomputed Y(Xw + b) if available
	@param nnzX : average number of non - zero elements for each sample
	@param index : index set for mini-batch calculation

	Returns
	---------- 
	@retval : \f$\ell_4( Y( Xw + b )) = [\ell_4^{(0)},\ell_4^{(1)},\dots,\ell_4^{(n-1)}]^T\f$
	"""

	if b == 1:
		# get a random sample
		i = np.random.randint( 0, n )

		Xi = X[i,:]
		XYw_bias = Y[i] * ( Xi.dot( w ) + bias[i] )

		return  (XYw_bias <= 1) * ( np.log( 1 + np.square( XYw_bias - 1 ) ))
	# batch
	elif b < n:
		# get a random batch of size b
		index = random.sample( range( n ), b )

		# calculate number of batches
		if nnzX is None:
			nnzX = d
		batch_size = np.maximum( int( total_mem_full // nnzX ), 1 )
		num_batches = math.ceil( b / batch_size )
		batch_loss = 0.0

		for j in range( num_batches ): 
			# calculate start/end indices for each batch
			startIdx = batch_size * j
			endIdx = np.minimum( batch_size * ( j + 1 ), b - 1 )

			batch_X = X[index[startIdx:endIdx],:]
			batch_Y = Y[index[startIdx:endIdx]]
			batch_bias = bias[index[startIdx:endIdx]]

			batch_XYw_bias = batch_Y * ( batch_X.dot( w ) + batch_bias )

			batch_loss +=  np.sum((batch_XYw_bias <= 1) * ( np.log( 1 + np.square( batch_XYw_bias - 1 ) )))

		return batch_loss / float( b )

	else:
		if XYw_bias is not None:
			return ( 1.0 / float( n )) * (  np.sum((XYw_bias <= 1) * ( np.log( 1 + np.square( XYw_bias - 1 ) ))))
		else:
			# calculate number of batches
			if nnzX is None:
				nnzX = d
			batch_size = np.maximum( int( total_mem_full // nnzX ), 1 )
			num_batches = math.ceil( n / batch_size )
			full_loss = 0.0

			for j in range( num_batches ): 
				# calculate start/end indices for each batch
				startIdx = batch_size * j
				endIdx = np.minimum( batch_size * ( j + 1 ), n - 1 )

				batch_X = X[startIdx:endIdx,:]
				batch_Y = Y[startIdx:endIdx]
				batch_bias = bias[startIdx:endIdx]

				batch_XYw_bias = batch_Y * ( batch_X.dot( w ) + batch_bias )

				full_loss +=  np.sum((batch_XYw_bias <= 1) * ( np.log( 1 + np.square( batch_XYw_bias - 1 ) )))

			return full_loss / float( n )

def func_diff_eval_bin_class_loss_4( n, d, b, X, Y, bias, w1, w2, lamb = None, XYw_bias = None, nnzX = None ):
	"""! Compute the objective value of loss function 3

	\f$ \ell_4( Y( Xw2 + b )) - \ell_4( Y( Xw1 + b )) \f$

	Parameters
	---------- 
	@param n : sample size
	@param d : number of features
	@param b : mini - batch size

		b = 1: single stochastic gradient

		1 < b < n: mini - batch stochastic gradient

		b = n: full gradient
	@param X : input data
	@param Y : input label
	@param bias : input bias
	@param w1 : 1st input vector
	@param w2 : 2nd input vector
	@param lamb: penalty parameters
	@param XYw_bias : precomputed Y(Xw + b) if available
	@param nnzX : average number of non - zero elements for each sample
	@param index : index set for mini-batch calculation

	Returns
	---------- 
	"""

	if b == 1:
		# get a random sample
		i = np.random.randint( 0, n )

		Xi = X[i,:]
		XYw_bias1 = Y[i] * ( Xi.dot( w1 ) + bias[i] )
		XYw_bias2 = Y[i] * ( Xi.dot( w2 ) + bias[i] )

		return  (XYw_bias2 <= 1) * ( np.log( 1 + np.square( XYw_bias2 - 1 ) )) \
				- (XYw_bias1 <= 1) * ( np.log( 1 + np.square( XYw_bias1 - 1 ) ))
	# batch
	elif b < n:
		# get a random batch of size b
		index = random.sample( range( n ), b )

		# calculate number of batches
		if nnzX is None:
			nnzX = d
		batch_size = np.maximum( int( total_mem_full // nnzX ), 1 )
		num_batches = math.ceil( b / batch_size )
		batch_loss_diff = 0.0

		for j in range( num_batches ): 
			# calculate start/end indices for each batch
			startIdx = batch_size * j
			endIdx = np.minimum( batch_size * ( j + 1 ), b - 1 )

			batch_X = X[index[startIdx:endIdx],:]
			batch_Y = Y[index[startIdx:endIdx]]
			batch_bias = bias[index[startIdx:endIdx]]

			batch_XYw_bias1 = batch_Y * ( batch_X.dot( w1 ) + batch_bias )
			batch_XYw_bias2 = batch_Y * ( batch_X.dot( w2 ) + batch_bias )

			batch_loss_diff +=  np.sum((batch_XYw_bias2 <= 1) * ( np.log( 1 + np.square( batch_XYw_bias2 - 1 ) )))\
								- np.sum((batch_XYw_bias1 <= 1) * ( np.log( 1 + np.square( batch_XYw_bias1 - 1 ) )))

		return batch_loss_diff / float( b )

	else:
		# calculate number of batches
		if nnzX is None:
			nnzX = d
		batch_size = np.maximum( int( total_mem_full // nnzX ), 1 )
		num_batches = math.ceil( n / batch_size )
		full_loss_diff = 0.0

		for j in range( num_batches ): 
			# calculate start/end indices for each batch
			startIdx = batch_size * j
			endIdx = np.minimum( batch_size * ( j + 1 ), n - 1 )

			batch_X = X[startIdx:endIdx,:]
			batch_Y = Y[startIdx:endIdx]
			batch_bias = bias[startIdx:endIdx]

			batch_XYw_bias1 = batch_Y * ( batch_X.dot( w1 ) + batch_bias )
			batch_XYw_bias2 = batch_Y * ( batch_X.dot( w2 ) + batch_bias )

			full_loss_diff +=  np.sum((batch_XYw_bias2 <= 1) * ( np.log( 1 + np.square( batch_XYw_bias2 - 1 ) )))\
								- np.sum((batch_XYw_bias1 <= 1) * ( np.log( 1 + np.square( batch_XYw_bias1 - 1 ) )))

		return full_loss_diff / float( n )

def grad_eval_bin_class_loss_4( n, d, b, X, Y, bias, w, lamb = None, nnzX = None ):
	"""! Compute the ( full/stochastic ) gradient of loss function 4.

	where \f$ \ell_4^{(i)}( Y_i( X_i^Tw + b )) := \begin{cases}	
			0,~&\text{if }Y_i( X_i^Tw + b ) > 1\\
			\ln(1 + (Y_i( X_i^Tw + b ) - 1)^2),&\text{otherwise}
	 \end{cases}\f$

	Parameters
	---------- 
	@param n : sample size
	@param d : number of features
	@param b : mini - batch size

		b = 1: single stochastic gradient

		1 < b < n: mini - batch stochastic gradient

		b = n: full gradient
	@param X : input data
	@param Y : input label
	@param bias : input bias
	@param w : input vector
	@param lamb: penalty parameters
	@param nnzX : average number of non - zero elements for each sample
	@param index : index set for mini-batch calculation

	Returns
	---------- 
	@retval  : computed full/stochastic gradient

	@retval XYw_bias: The precomputed \f$ Y( Xw + bias )\f$
	"""
	if nnzX is None:
		nnzX = d
	
	# single sample
	if b == 1:
		# get a random sample
		i = np.random.randint( 0, n )

		Xi = X[i, :]
		tempV = Y[i] * ( Xi.dot( w ) + bias[i] ) - 1

		return ( 2 * tempV * Y[i] / ( 1 + np.square( tempV ) )) * Xi
	# batch
	elif b < n:
		# get a random batch of size b
		index = random.sample( range( n ), b )

		# calculate number of batches
		if nnzX == 0:
			nnzX = d
		batch_size = np.maximum( int( total_mem_full // nnzX ), 1 )
		num_batches = math.ceil( b / batch_size )
		batch_grad = np.zeros( d )

		for j in range( num_batches ): 
			# calculate start/end indices for each batch
			startIdx = batch_size * j
			endIdx = np.minimum( batch_size * ( j + 1 ), b - 1 )

			batch_X = X[index[startIdx:endIdx],:]
			batch_Y = Y[index[startIdx:endIdx]]
			batch_bias = bias[index[startIdx:endIdx]]

			tempVect = batch_Y * ( batch_X.dot( w ) + batch_bias ) - 1

			batch_grad += batch_X.transpose().dot( batch_Y * 2 * tempVect / ( 1.0 + np.square( tempVect ) ))

		return batch_grad / float( b )
	# full
	else:
		# calculate number of batches
		if nnzX == 0:
			nnzX = d
		batch_size = np.maximum( int( total_mem_full // nnzX ), 1 )
		num_batches = math.ceil( b / batch_size )
		full_grad = np.zeros( d )
		XYw_bias = np.zeros( n )

		for j in range( num_batches ): 
			# calculate start/end indices for each batch
			startIdx = batch_size * j
			endIdx = np.minimum( batch_size * ( j + 1 ), n - 1 )

			batch_X = X[startIdx:endIdx,:]
			batch_Y = Y[startIdx:endIdx]
			batch_bias = bias[startIdx:endIdx]

			batch_XYw_bias = batch_Y * ( batch_X.dot( w ) + batch_bias )

			tempVect = batch_XYw_bias - 1

			XYw_bias[startIdx:endIdx] = batch_XYw_bias

			full_grad += batch_X.transpose().dot( batch_Y * 2 * tempVect / ( 1.0 + np.square( tempVect ) ))

		return full_grad / float( n ), XYw_bias
		
def grad_diff_eval_bin_class_loss_4( n, d, b, X, Y, bias, w1, w2, lamb = None, nnzX = None ):
	"""! Compute the ( full/stochastic ) gradient difference of loss function 3

	\f$\displaystyle\frac{1}{b}\left( \sum_{i \in \mathcal{B}_t}( \nabla f_i( w_2 ) - \nabla f_i( w_1 )) \right ) \f$

	Parameters
	---------- 
	@param n : sample size
	@param d : number of features
	@param b : mini - batch size

		b = 1: single stochastic gradient

		1 < b < n: mini - batch stochastic gradient

		b = n: full gradient
	@param X : input data
	@param Y : input label
	@param bias : input bias
	@param w1 : 1st input vector
	@param w2 : 2nd input vector
	@param lamb: penalty parameters
	@param nnzX : average number of non - zero elements for each sample
	@param index : index set for mini-batch calculation

	Returns
	---------- 
	@retval  : computed full/stochastic gradient
	"""
	if nnzX is None:
		nnzX = d
	
	alpha = 1
	exp_a = np.exp( alpha )
	# single sample
	if b == 1:
		# get a random sample
		i = np.random.randint( 0, n )

		Xi = X[i,:]

		tempV1 = Y[i] * ( Xi.dot( w1 ) + bias[i] ) - 1
		tempV2 = Y[i] * ( Xi.dot( w2 ) + bias[i] ) - 1

		diff = 2 * ( tempV2 / ( 1.0 + np.square(tempV2) ) - tempV1 / ( 1.0 + np.square(tempV1) ))
		
		return diff * Y[i] * Xi
	# batch
	elif b < n:
		# get a random batch of size b
		index = random.sample( range( n ), b )

		# calculate number of batches
		if nnzX == 0:
			nnzX = d
		batch_size = int( total_mem_batch // nnzX )
		num_batches = math.ceil( b / batch_size )
		batch_grad_diff = np.zeros( d )

		for j in range( num_batches ): 
			# calculate start/end indices for each batch
			startIdx = batch_size * j
			endIdx = np.minimum( batch_size * ( j + 1 ), b - 1 )

			batch_X = X[index[startIdx:endIdx],:]
			batch_Y = Y[index[startIdx:endIdx]]
			batch_bias = bias[index[startIdx:endIdx]]

			tempVect1 = batch_Y * ( batch_X.dot( w1 )  + batch_bias ) - 1
			tempVect2 = batch_Y * ( batch_X.dot( w2 )  + batch_bias ) - 1

			diff = 2 * ( tempVect2 / ( 1.0 + np.square(tempVect2) ) - tempVect1 / ( 1.0 + np.square(tempVect1) ))

			batch_grad_diff += batch_X.transpose().dot( batch_Y * diff )

		return batch_grad_diff / float( b )
	# full
	else:
		# calculate number of batches
		if nnzX == 0:
			nnzX = d
		batch_size = int( total_mem_full // nnzX )
		num_batches = math.ceil( b / batch_size )
		full_grad_diff = np.zeros( d )

		for j in range( num_batches ): 
			# calculate start/end indices for each batch
			startIdx = batch_size * j
			endIdx = np.minimum( batch_size * ( j + 1 ), n - 1 )

			batch_X = X[startIdx:endIdx]
			batch_Y = Y[startIdx:endIdx]
			batch_bias = bias[startIdx:endIdx]

			tempVect1 = batch_Y * ( batch_X.dot( w1 )  + batch_bias ) - 1
			tempVect2 = batch_Y * ( batch_X.dot( w2 )  + batch_bias ) - 1

			diff = 2 * ( tempVect2 / ( 1.0 + np.square(tempVect2) ) - tempVect1 / ( 1.0 + np.square(tempVect1) ))

			full_grad_diff += batch_X.transpose().dot( batch_Y * diff )

		return full_grad_diff / float( n )

