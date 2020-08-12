"""!@package method_civr

CIVR algorithm for the SNE example.

"""

import numpy as np
import time

def sne_nspider(x0, obj_func, non_lin_func, non_lin_dat, prob_params, alg_params, print_interval=2.0):

    n_ = non_lin_dat['matrix'].shape[0]
    d_ = non_lin_dat['matrix'].shape[1]

    s_b_1 = int(alg_params['jac_snap_batch'])
    s_b_2 = int(alg_params['func_snap_batch'])

    b_1 = int(alg_params['jac_batch'])
    b_2 = int(alg_params['func_batch'])

    num_func = len(non_lin_func)
    lr = alg_params['lr']
    epsilon = alg_params.get('epsilon',1e-3)

    # initialize history
    history = {
        'Time'      :   [],
        'Epochs'    :   [],     
        'Samples'   :   [],
        'Obj'       :   [],
        'GradMap'   :   [],
    }

    # print information
    print( '\nN-SPIDER Algorithm\n' )
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

    # for stats computation
    n_samples = 0
    n_epochs = 0
    last_print = 0
    ep_inc = (b_1 + b_2) / n_

    # assign initial point
    x_cur = x0

    # calculate jacobian and function estimators
    jac_est = np.zeros((num_func,d_))
    func_est = np.zeros(num_func)

    # save start time
    start = time.time()

    # loop until done
    while n_epochs < alg_params['max_epochs']:
        # calculate jacobian and function estimators
        for i in range(num_func):
            if s_b_1 == n_:
                jac_est[i,:], _ = non_lin_func['Grad'][i]( n_, d_, s_b_1, non_lin_dat['matrix'], non_lin_dat['label'], non_lin_dat['bias'], x_cur )
            else:
                jac_est[i,:] = non_lin_func['Grad'][i]( n_, d_, s_b_1, non_lin_dat['matrix'], non_lin_dat['label'], non_lin_dat['bias'], x_cur )
            func_est[i] = non_lin_func['Func'][i](n_, d_, s_b_2, non_lin_dat['matrix'], non_lin_dat['label'], non_lin_dat['bias'], x_cur)

        # update
        x_prev = x_cur
        v_t = jac_est.T.dot(obj_func['Grad'](func_est))
        norm_v = np.linalg.norm(v_t)
        lr = min(alg_params['lr'], epsilon/norm_v)
        x_next = x_cur - lr*v_t

        # print info
        if n_epochs == 0 or n_epochs - last_print >= print_interval:
            # calculate jacobian and function estimators
            for i in range(num_func):
                if s_b_1 < n_:
                    jac_est[i,:], _ = non_lin_func['Grad'][i]( n_, d_, n_, non_lin_dat['matrix'], non_lin_dat['label'], non_lin_dat['bias'], x_cur )
                if s_b_2 < n_:
                    func_est[i] = non_lin_func['Func'][i](n_, d_, n_, non_lin_dat['matrix'], non_lin_dat['label'], non_lin_dat['bias'], x_cur )

            # calculate objective value
            obj_val = obj_func['Func'](func_est)

            # calculate gradient mapping norm
            grad_map = jac_est.T.dot(obj_func['Grad'](func_est))
            grad_map_norm = np.linalg.norm(grad_map, ord=2)

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
        n_samples += 2*n_
        n_epochs += 2

        # update
        x_next = x_cur

        # start inner loop
        for itr in range(alg_params['max_inner']):
            # calculate estimator of jacobian
            for i in range(num_func):
                jac_est[i,:] = jac_est[i,:] + non_lin_func['GradDiff'][i](n_, d_, b_1, non_lin_dat['matrix'], non_lin_dat['label'], non_lin_dat['bias'], x_prev, x_cur )
                func_est[i] = func_est[i] + non_lin_func['FuncDiff'][i](n_, d_, b_2, non_lin_dat['matrix'], non_lin_dat['label'], non_lin_dat['bias'], x_prev, x_cur)

            # solve subproblem
            x_prev = x_cur
            v_t = jac_est.T.dot(obj_func['Grad'](func_est))
            norm_v = np.linalg.norm(v_t)
            lr = min(alg_params['lr'], epsilon/norm_v)
            x_next = x_cur - lr*v_t

            if n_epochs - last_print >= print_interval:
                # calculate jacobian and function estimators
                for i in range(num_func):
                    jac_est[i,:], _ = non_lin_func['Grad'][i]( n_, d_, n_, non_lin_dat['matrix'], non_lin_dat['label'], non_lin_dat['bias'], x_cur )
                    func_est[i] = non_lin_func['Func'][i](n_, d_, n_, non_lin_dat['matrix'], non_lin_dat['label'], non_lin_dat['bias'], x_cur )

                # calculate objective value
                obj_val = obj_func['Func'](func_est)

                # calculate gradient mapping norm
                grad_map = jac_est.T.dot(obj_func['Grad'](func_est))
                grad_map_norm = np.linalg.norm(grad_map, ord=2)

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
                for i in range(num_func):
                    jac_est[i,:], _ = non_lin_func['Grad'][i]( n_, d_, n_, non_lin_dat['matrix'], non_lin_dat['label'], non_lin_dat['bias'], x_cur )
                    func_est[i] = non_lin_func['Func'][i](n_, d_, n_, non_lin_dat['matrix'], non_lin_dat['label'], non_lin_dat['bias'], x_cur )

                # calculate objective value
                obj_val = obj_func['Func'](func_est)

                # calculate grad_map
                v_t = jac_est.T.dot(obj_func['Grad'](func_est))

                # calculate gradient mapping norm
                grad_map_norm = np.linalg.norm(v_t, ord=2)

                # Compute the solution change.
                abs_schg = np.linalg.norm(x_next - x_cur,ord=2)
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
                history['Time'].append(cur_time)
                history['Epochs'].append(n_epochs)
                history['Samples'].append(n_samples)
                history['Obj'].append(obj_val)
                history['GradMap'].append(grad_map_norm)

                break

    print('{message:{fill}{align}{width}}'.format(message='',fill='=',align='^',width=90))

    return history
