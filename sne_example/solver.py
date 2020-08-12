"""!@package sub_admm

ADMM solver for the SNE example.

"""

import numpy as np
from utils.func_utils import *
import time

def solve_sub_problem(jac_est, func_est, obj_func, x_til, prob_params, debug=False, solver='adpg'):
    if solver == 'adpg':
        def prox_phi_conj(x, lbd=1.0):
            return x - lbd * obj_func['Prox'](x/lbd, 1.0/lbd)
        
        return sub_adpgpd_solver(jac_est, func_est, x_til, prox_phi_conj, prob_params, debug = False)

    elif solver == 'pd':
        def prox_phi_hat(x, lbd=1.0):
            return obj_func['Prox'](x + func_est, lbd) - func_est
        def prox_phi_hat_conj(x, lbd=1.0):
            return x - lbd * prox_phi_hat(x/lbd, 1.0/lbd)

        M_const = prob_params.get('M_const', 1.0)
        def prox_psi(x, lbd=1.0):
            return x / (1 + lbd*M_const)
        
        return sub_pd_solver(x_til, prox_phi_hat_conj, prox_psi, jac_est, prob_params, debug = False)

    elif solver == 'admm':
        # form P and q
        P_mat = jac_est
        q_vect = func_est - jac_est.dot(x_til)
        return sub_admm(P_mat, q_vect, obj_func, x_til, prob_params, debug)

def sub_admm(P_mat, q_vect, obj_func, x_til, prob_params, debug=False):

    d_ = x_til.shape[0]
    p_ = P_mat.shape[0]
    
    M_const = prob_params.get('M_const', 1.0)
    rho = prob_params.get('rho', 1.0)
    max_sub_iter = prob_params.get('max_sub_iter', 100)
    epsilon = prob_params.get('epsilon', 1e-6)

    temp_mat= rho*np.matmul(P_mat.T,P_mat) + M_const*np.eye(d_,d_)

    temp_mat_inv = np.linalg.inv(temp_mat)

    x_cur = np.zeros(d_)
    r_cur = np.zeros(p_)
    w_cur = np.zeros(p_)

    for k in range(max_sub_iter):

        # solve for r_{k+1}
        r_next = obj_func['Prox'](P_mat@x_cur + q_vect + w_cur, 1.0/rho)

        # solve for x_{k+1}
        temp_vect = M_const * x_til + rho* P_mat.T@(r_next - q_vect - w_cur)

        # x_next = np.linalg.solve(temp_mat,temp_vect)
        x_next = temp_mat_inv.dot(temp_vect)

        # update w_k
        w_next = w_cur + P_mat@x_next - r_next + q_vect

        feas_norm = np.linalg.norm(P_mat@x_next - r_next + q_vect)
        sub_obj = 0.5*M_const*np.linalg.norm(r_next) + 0.5*rho*(np.linalg.norm(x_next - x_cur))**2

        x_cur = x_next
        r_cur = r_next
        w_cur = w_next

        if debug:
            if k % 5 == 0:
                print("Time: {:f}, Iter: {:5d}, Obj Val: {:3.2e}, Feasibility: {:3.2e}".format(time.time()-start_time,k,sub_obj,feas_norm))

        if feas_norm <= epsilon:
            break

    return x_cur

def sub_adpgpd_solver(jac_est, func_est, x_til, prox_phi_conj, prob_params, debug=False):

    d_ = x_til.shape[0]
    p_ = func_est.shape[0]

    jac_est_t = jac_est.T

    M_const = prob_params.get('M_const', 1.0)
    max_iter = prob_params.get('max_sub_iter', 100)

    L_const = np.linalg.norm(jac_est.dot(jac_est_t)) / M_const

    L_const_inv = 1.0/L_const
    M_const_inv = 1.0/M_const

    tau_cur = 1

    u_cur = u_hat = np.zeros(p_)

    if debug:
        start_time = time.time()

    for k in range(max_iter):
        u_next = prox_phi_conj( u_hat - L_const_inv*(M_const_inv*jac_est.dot(jac_est_t.dot(u_hat)) - func_est), L_const_inv)

        tau_next = 0.5*(1 + np.sqrt(1 + 4*tau_cur*tau_cur))

        u_hat = u_next + ((tau_cur-1)/tau_next) * (u_next - u_cur)

        # Compute the solution change.
        abs_schg = np.linalg.norm(u_next - u_cur, ord=2);
        rel_schg = abs_schg/np.maximum(1, np.linalg.norm(u_cur, ord=2))

        if rel_schg <= prob_params['RelTolSoln'] :
            if debug:
                print('Convergence achieved')
                print('The serarch direction norm and the feasibility gap is below the desired threshold')
            u_cur = u_next
            break

        if debug:
            print("Time: {:f}, SubProblem, Iter: {:5d}, Rel Sol Change: {:3.2e}".format(time.time()-start_time,k,rel_schg))

        # update
        u_cur = u_next
        tau_cur = tau_next

    return x_til - M_const_inv*jac_est.T.dot(u_cur)

def sub_pd_solver(x_til, prox_phi_conj, prox_psi, jac_est, prob_params, debug=False):

    d_ = x_til.shape[0]
    p_ = jac_est.shape[0]

    jac_est_t = jac_est.T

    M_const = prob_params.get('M_const', 1.0)
    max_iter = prob_params.get('max_sub_iter', 100)

    L_const = np.linalg.norm(jac_est.dot(jac_est_t))

    sigma_cur = 1
    tau_cur = 1.0/(sigma_cur* L_const)

    u_cur = u_hat = np.zeros(p_)
    d_cur = d_bar = np.zeros(d_)

    if debug:
        start_time = time.time()

    for k in range(max_iter):
        u_next = prox_phi_conj(u_cur + sigma_cur * jac_est.dot(d_bar), sigma_cur)

        d_next = prox_psi(d_cur - tau_cur* jac_est_t.dot(u_next))

        theta = 1.0/np.sqrt(1 + 2*M_const*tau_cur)

        tau_next = theta*tau_cur

        sigma_next = sigma_cur / theta

        d_bar = d_next + theta*(d_next - d_cur)

        # Compute the solution change.
        abs_schg = np.linalg.norm(d_bar - d_cur, ord=2);
        rel_schg = abs_schg/np.maximum(1, np.linalg.norm(d_cur, ord=2))

        if rel_schg <= prob_params['RelTolSoln'] :
            if debug:
                print('Convergence achieved')
                print('The serarch direction norm and the feasibility gap is below the desired threshold')
            d_cur = d_next
            break

        if debug:
            print("Time: {:f}, SubProblem, Iter: {:5d}, Rel Sol Change: {:3.2e}".format(time.time()-start_time,k,rel_schg))

        # update
        u_cur = u_next
        d_cur = d_next
        sigma_cur = sigma_next
        tau_cur = tau_next

    return x_til + d_cur
