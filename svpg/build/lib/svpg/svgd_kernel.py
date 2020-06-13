
''' Stein Varitional Policy Gradient '''

import numpy as np


def stein_update_policies(weights_list, gradient_list,
                          num_agents=8,
                          adaptive_kernel=False,
                          include_kernel=True,
                          stein_learning_rate=0.001,  #1e-2   #1e-3
                          search_space=np.linspace(0.1, 2.0, num=20),
                          temp=50,
                          stein_optimization_method='adam', # or 'adagrad'
                          stein_m=None,
                          stein_v=None,
                          stein_epsilon=1e-8,
                          stein_beta1=0.9,
                          stein_beta2=0.999,
                          stein_t=0
                          ):
    '''
    Inputs:
    (1) Numpy array: Calculated gradient_list from all agents for current episode (list of 1D concatenated array)
    (2) Numpy array: Extracted weights from all agents for current episode
    (3) Other parameters for stein updates
    Output:
    (1) Numpy array: Updated weights
    '''

    ## Siyan modified gradient on 09.30.2019 with no negative sign
    # gradient = np.array(gradient_list)
    gradient = -np.array(gradient_list)
    params = np.array(weights_list)
    # print('NEW PARAMS Shape: {}'.format(params.shape))
    # params = np.array([self.policy_list[i].get_param_values() for i in range(self.num_of_agents)])  # Check this Numpy array for the policy for all agents

    ## get distance matrix: squared Euclidean distance
    distance_matrix = np.sum(np.square(params[None, :, :] - params[:, None, :]), axis=-1)   # Check this 'distance_matrix'

    # get median
    distance_vector = distance_matrix.flatten()
    distance_vector.sort()
    median = 0.5 * (distance_vector[int(len(distance_vector) / 2)] + distance_vector[int(len(distance_vector) / 2) - 1])
    h = median / (2 * np.log(num_agents + 1))

    # we did not use adaptive_kernel
    if adaptive_kernel:
        L_min = None
        alpha_best = None
        for alpha in search_space:
            kernel_alpha = np.exp(distance_matrix * (-alpha / h))
            mean_kernel = np.sum(kernel_alpha, axis=1)
            L = np.mean(np.square(mean_kernel - 2.0 * np.ones_like(mean_kernel)))
            print("Current Loss {:} and Alpha : {:}".format(L, alpha))
            if L_min is None:
                L_min = L
                alpha_best = alpha
            elif L_min > L:
                L_min = L
                alpha_best = alpha
        # logger.record_tabular('Best Alpha', alpha_best)
        print('Best Alpha', alpha_best)
        h = h / alpha_best

    kernel = np.exp(distance_matrix[:, :] * (-1.0 / h))
    kernel_gradient = kernel[:, :, None] * (2.0 / h) * (params[None, :, :] - params[:, None, :])
    if include_kernel:
        weights = (1.0 / temp) * kernel[:, :, None] * gradient[:, None, :] + kernel_gradient[:, :, :]
    else:
        weights = kernel[:, :, None] * gradient[:, None, :]

    weights = -np.mean(weights[:, :, :], axis=0)

    # adam update
    if stein_optimization_method == 'adam':
        if stein_m is None:
            stein_m = np.zeros_like(params)
        if stein_v is None:
            stein_v = np.zeros_like(params)
        stein_t += 1.0
        stein_m = stein_beta1 * stein_m + (1.0 - stein_beta1) * weights
        stein_v = stein_beta2 * stein_v + (1.0 - stein_beta2) * np.square(weights)
        m_hat = stein_m / (1.0 - stein_beta1 ** stein_t)
        v_hat = stein_v / (1.0 - stein_beta2 ** stein_t)
        params = params - stein_learning_rate * (m_hat / (np.sqrt(v_hat) + stein_epsilon))
    elif stein_optimization_method == 'adagrad':
        if stein_m is None:
            stein_m = np.zeros_like(params)
        stein_m = stein_m + np.square(weights)
        params = params - stein_learning_rate * (weights / (
            np.sqrt(stein_m + stein_epsilon)))  # Update policies with learning rates and weights

    print('Median', median)
    print('KGradient_Max', np.max(kernel_gradient.flatten()))
    print('Kernal_Max', np.max(kernel.flatten()))
    print('PolicyGradient_Max', np.max(gradient.flatten()))

    return params  ## updated weights (num_agents, num_weights)