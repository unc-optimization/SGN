"""!@package method_sgn2

SGN2 algorithm for the Asset Allocation example.

"""

import numpy as np
from .asset_utils import *
from .solver import *
import time

def asset_sgn2(x0, in_data, prob_params, alg_params, print_interval=2.0):

    rho = prob_params['rho']
    M_const = prob_params['M_const']
    beta = prob_params['beta']

    A_mat = in_data['A']
    mu_vect = in_data['mu']

    n_ = A_mat.shape[0]
    d_ = A_mat.shape[1]

    s_b_1 = int(alg_params['jac_snap_batch'])
    s_b_2 = int(alg_params['func_snap_batch'])

    b_1 = alg_params['jac_batch']
    b_2 = alg_params['func_batch']

    # initialize history
    history = {
        'Time'      :   [],
        'Epochs'    :   [],     
        'Samples'   :   [],
        'Obj'       :   [],
        'GradMap'   :   [],
    }

    # print information
    print('\n Asset Allocation Example with SGN2 algorithm\n')
    print('{message:{fill}{align}{width}}'.format(message='',fill='=',align='^',width=90))
    print(
            '{message:{fill}{align}{width}}'.format(message='Time',fill=' ',align='^',width=12,),'|',
            '{message:{fill}{align}{width}}'.format(message='# Epochs',fill=' ',align='^',width=10,),'|',
            '{message:{fill}{align}{width}}'.format(message='# Samples',fill=' ',align='^',width=13,),'|',
            '{message:{fill}{align}{width}}'.format(message='Obj. Val.',fill=' ',align='^',width=13,),'|',
            '{message:{fill}{align}{width}}'.format(message='||G_M(x)||',fill=' ',align='^',width=13,),'|',
            '{message:{fill}{align}{width}}'.format(message='Rel. Sol. Ch.',fill=' ',align='^',width=13,),
        )
    print('{message:{fill}{align}{width}}'.format(message='',fill='-',align='^',width=90))

    def phi_func(x):
        return 0.5*rho*(x + np.abs(x))

    def prox_phi(x, gamma):
        return prox_l1_norm(x - 0.5*gamma*rho, 0.5*gamma*rho)

    t_range = [0,1]

    # for stats computation
    n_samples = 0
    n_epochs = 0
    last_print = 0
    ep_inc = (b_1 + b_2) / n_

    # assign initial point
    x_cur = x0

    # calculate jacobian and function estimators
    jac_est = np.zeros((d_+1))
    func_est = np.zeros(1)

    # save start time
    start = time.time()

    # loop until done
    while n_epochs < alg_params['max_epochs']:
        # calculate jacobian and function estimators
        jac_est = f_jac(A_mat, x_cur[:-1], x_cur[-1], beta=beta, bs=s_b_1)
        func_est = f_func(A_mat,  x_cur[:-1], x_cur[-1], beta=beta, bs=s_b_2)

        a_vect = func_est - jac_est.dot(x_cur)

        def prox_psi(x, gamma):
            return prox_phi(x + a_vect, gamma) - a_vect

        def prox_psi_conj(x , gamma):
            return x - gamma* prox_psi (x/gamma, 1/gamma)

        def prox_psi_conj(x , gamma):
            return x - gamma* prox_psi (x/gamma, 1/gamma)

        def prox_g(x, gamma):
            res = np.zeros_like(x)
            tmp = (M_const*gamma*x_cur + x + gamma*mu_vect)/(1 + gamma*M_const)
            res[:-1] = proj_simplex( tmp[:-1] , gamma/(M_const*gamma + 1))
            t_range = [0,1]
            res[-1] = np.minimum(np.maximum(x[-1], t_range[0]),t_range[1])
            return res


        # solve subproblem
        x_prev = x_cur
        x_next = pd_solver(jac_est, func_est, x_cur, prox_psi_conj, prox_g, prob_params, debug = False)

        # check for printouts
        if n_epochs == 0 or n_epochs - last_print >= print_interval:
            # calculate jacobian and function estimators
            if s_b_1 < n_:
                jac_est = f_jac(A_mat, x_cur[:-1], x_cur[-1], beta=beta, bs=n_)
            if s_b_2 < n_:
                func_est = f_func(A_mat,  x_cur[:-1], x_cur[-1], beta=beta, bs=n_)

            # calculate objective value
            obj_val = -mu_vect.dot(x_cur) + phi_func(func_est)

            # calculate gradient mapping norm
            if s_b_1 < n_ or s_b_2 < n_:
                a_vect = func_est - jac_est.dot(x_cur)

                def prox_psi(x, gamma):
                    return prox_phi(x + a_vect, gamma) - a_vect

                def prox_psi_conj(x , gamma):
                    return x - gamma* prox_psi (x/gamma, 1/gamma)

            x_tmp = pd_solver(jac_est, func_est, x_cur, prox_psi_conj, prox_g, prob_params, debug = False)
            grad_map_norm = M_const* np.linalg.norm(x_cur - x_tmp, ord=2)

            # calculate relative solution change
            if n_epochs == 0:
                # no solution change yet
                rel_schg = np.inf
                
            else:
                # Compute the solution change
                abs_schg = np.linalg.norm(x_next - x_cur,ord=2);
                rel_schg = abs_schg/np.maximum(1, np.linalg.norm(x_next,ord=2))
            
            # print debug information
            cur_time = time.time() - start
            print(
                '{:^12.2e}'.format(cur_time),'|',
                '{:^10.1f}'.format(n_epochs),'|',
                '{:^13.3e}'.format(n_samples),'|',
                '{:^13.3e}'.format(obj_val),'|',
                '{:^13.3e}'.format(grad_map_norm),'|',
                '{:^13.3e}'.format(rel_schg),
            )

            # add to history
            history['Time'].append(cur_time)
            history['Epochs'].append(n_epochs)
            history['Samples'].append(n_samples)
            history['Obj'].append(obj_val)
            history['GradMap'].append(grad_map_norm)
            last_print = n_epochs

        # update number of samples
        n_samples += (s_b_1 + s_b_2)
        n_epochs +=  (s_b_1 + s_b_2) / n_

        # update
        x_cur = x_next

        # start inner loop
        for itr in range(alg_params['max_inner']):
            # calculate estimator of function and jacobian
            jac_est = jac_est + f_jac_diff(A_mat, x_prev[:-1], x_prev[-1], x_cur[:-1], x_cur[-1], beta=beta, bs=b_1)
            func_est = func_est + f_func_diff(A_mat,  x_prev[:-1], x_prev[-1], x_cur[:-1], x_cur[-1], beta=beta, bs=b_2)

            a_vect = func_est - jac_est.dot(x_cur)

            def prox_psi(x, gamma):
                return prox_phi(x + a_vect, gamma) - a_vect

            def prox_psi_conj(x , gamma):
                return x - gamma* prox_psi (x/gamma, 1/gamma)

            def prox_g(x, gamma):
                res = np.zeros_like(x)
                tmp = (M_const*gamma*x_cur + x + gamma*mu_vect)/(1 + gamma*M_const)
                res[:-1] = proj_simplex( tmp[:-1] , gamma/(M_const*gamma + 1))
                t_range = [0,1]
                res[-1] = np.minimum(np.maximum(x[-1], t_range[0]),t_range[1])
                return res

            # solve subproblem
            x_prev = x_cur
            x_next = pd_solver(jac_est, func_est, x_cur, prox_psi_conj, prox_g, prob_params, debug = False)

            if n_epochs - last_print >= print_interval:
                # calculate jacobian and function estimators
                jac_est = f_jac(A_mat, x_cur[:-1], x_cur[-1], beta=beta, bs=n_)
                func_est = f_func(A_mat,  x_cur[:-1], x_cur[-1], beta=beta, bs=n_)

                # calculate objective value
                obj_val = -mu_vect.dot(x_cur) + phi_func(func_est)

                # calculate gradient mapping norm
                a_vect = func_est - jac_est.dot(x_cur)

                def prox_psi(x, gamma):
                    return prox_phi(x + a_vect, gamma) - a_vect

                def prox_psi_conj(x , gamma):
                    return x - gamma* prox_psi (x/gamma, 1/gamma)

                x_tmp = pd_solver(jac_est, func_est, x_cur, prox_psi_conj, prox_g, prob_params, debug = False)
                grad_map_norm = M_const* np.linalg.norm(x_cur - x_tmp, ord=2)

                # calculate relative solution change
                if n_epochs == 0:
                    # no solution change yet
                    rel_schg = np.inf
                    
                else:
                    # Compute the solution change
                    abs_schg = np.linalg.norm(x_next - x_cur,ord=2);
                    rel_schg = abs_schg/np.maximum(1, np.linalg.norm(x_next,ord=2))
                
                # print debug information
                cur_time = time.time() - start
                print(
                    '{:^12.2e}'.format(cur_time),'|',
                    '{:^10.1f}'.format(n_epochs),'|',
                    '{:^13.3e}'.format(n_samples),'|',
                    '{:^13.3e}'.format(obj_val),'|',
                    '{:^13.3e}'.format(grad_map_norm),'|',
                    '{:^13.3e}'.format(rel_schg),
                )

                # add to history
                history['Time'].append(cur_time)
                history['Epochs'].append(n_epochs)
                history['Samples'].append(n_samples)
                history['Obj'].append(obj_val)
                history['GradMap'].append(grad_map_norm)
                last_print = n_epochs


            # update number of samples/epochs
            n_samples += 2*(b_1 + b_2)
            n_epochs += 2*ep_inc

            # update
            x_cur = x_next

            # check if done
            if n_epochs >= alg_params['max_epochs']:
                # calculate jacobian and function estimators
                jac_est = f_jac(A_mat, x_cur[:-1], x_cur[-1], beta=beta, bs=n_)
                func_est = f_func(A_mat,  x_cur[:-1], x_cur[-1], beta=beta, bs=n_)

                # calculate objective value
                obj_val = -mu_vect.dot(x_cur) + phi_func(func_est)

                # calculate gradient mapping norm
                a_vect = func_est - jac_est.dot(x_cur)

                def prox_psi(x, gamma):
                    return prox_phi(x + a_vect, gamma) - a_vect

                def prox_psi_conj(x , gamma):
                    return x - gamma* prox_psi (x/gamma, 1/gamma)

                x_tmp = pd_solver(jac_est, func_est, x_cur, prox_psi_conj, prox_g, prob_params, debug = False)
                grad_map_norm = M_const* np.linalg.norm(x_cur - x_tmp, ord=2)

                # Compute the solution change.
                abs_schg = np.linalg.norm(x_tmp - x_cur,ord=2)
                rel_schg = abs_schg/np.maximum(1, np.linalg.norm(x_tmp,ord=2))
                
                # print debug information
                cur_time = time.time() - start
                print(
                    '{:^12.2e}'.format(cur_time),'|',
                    '{:^10.1f}'.format(n_epochs),'|',
                    '{:^13.3e}'.format(n_samples),'|',
                    '{:^13.3e}'.format(obj_val),'|',
                    '{:^13.3e}'.format(grad_map_norm),'|',
                    '{:^13.3e}'.format(rel_schg),
                )
                history['Time'].append(cur_time)
                history['Epochs'].append(n_epochs)
                history['Samples'].append(n_samples)
                history['Obj'].append(obj_val)
                history['GradMap'].append(grad_map_norm)

                break

    print('{message:{fill}{align}{width}}'.format(message='',fill='=',align='^',width=90))
        
    return history

