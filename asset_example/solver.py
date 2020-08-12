"""!@package sub_pd

Primal-Dual solver for the Asset Allocation example.

"""

import numpy as np

def pd_solver(jac_est, func_est, x0, prox_1, prox_2, prob_params, debug = False):

	# solve
	# min F(Ax) + G(x)
	#
	#

	if jac_est.ndim > 1:
		d_ = jac_est.shape[1]
		m_ = jac_est.shape[0]
	else:
		d_ = len(jac_est)
		m_ = 1

	M_const = prob_params.get('M_const', 1.0)
	max_iter = prob_params.get('max_sub_iter', 100)

	norm_J = np.linalg.norm(jac_est)

	# strongly convex parameter
	mu_g = M_const
	sigma_cur = 5 / norm_J
	tau_cur = 1.0/(sigma_cur*norm_J**2)
	x_cur = x0
	x_bar = x_cur
	y_cur = np.zeros(m_)

	for i in range(max_iter):
		# y_{k+1}
		y_next = prox_1(y_cur + sigma_cur*jac_est.dot(x_bar), sigma_cur)

		# x_{k+1}
		if m_ == 1:
			x_next = prox_2(x_cur - tau_cur*jac_est*y_next, tau_cur)
		else:
			x_next = prox_2(x_cur - tau_cur*jac_est.T.dot(y_next), tau_cur)

		# theta_k
		theta_k = 1.0/np.sqrt(1 + 2*tau_cur*mu_g)
		tau_next = theta_k * tau_cur
		sigma_next = sigma_cur / theta_k

		# x_bar_next
		x_bar = x_next + theta_k*(x_next - x_cur)

		# Compute the feasibility.
		abs_pfeas = np.linalg.norm(y_next - y_cur, ord=2);
		rel_pfeas = abs_pfeas/np.maximum(np.linalg.norm(y_cur, ord=2), 1)

		# Compute the solution change.
		abs_schg = np.linalg.norm(x_next - x_cur, ord=2);
		rel_schg = abs_schg/np.maximum(1, np.linalg.norm(x_cur, ord=2))

		if rel_schg <= prob_params['RelTolX'] and rel_pfeas <= prob_params['RelTolFeas'] and i > 1:
			if debug:
				print('Convergence achieved')
				print('The serarch direction norm and the feasibility gap is below the desired threshold')
			x_cur        = x_next
			y_cur        = y_next
			break

		if debug:
			if i % 10 == 0:
				print("SubProblem, Iter: {0:5d}, Rel Sol Change: {1:3.2e}, Rel Feas.: {1:3.2e}".format(i,rel_schg,rel_pfeas))

		# update
		x_cur = x_next
		tau_cur = tau_next
		sigma_cur = sigma_next
		y_cur = y_next

	return x_cur