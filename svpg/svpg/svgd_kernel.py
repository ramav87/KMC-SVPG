''' Stein Varitional Policy Gradient '''

import numpy as np

def stein_update_policies(weights_list, gradient_list,
                          num_agents=4,
                          temp=0.8,
                          ):
    '''
    Inputs:
    (1) Numpy array: Extracted weights from all agents for current batch
    (2) Numpy array: Calculated gradients from all agents for current batch (list of 1D concatenated array)
    (3) temp: Stein update temperature
    Output:
    (1) Numpy array: Stein update gradient. This should then be applied with a choice of optimizer.
    '''


    gradient = -1.0*np.array(gradient_list) #the gradients are positive lob probability.
                                            ## But since we are using tf optimizers, they work on w = w-(dt*grad). Which means for maximizatioin
                                            #of objective we want to take the negative
    params = np.array(weights_list)

    distance_matrix = np.sum(np.square(params[None, :, :] - params[:, None, :]), axis=-1)
    # get median
    distance_vector = distance_matrix.flatten()
    distance_vector.sort()
    median = 0.5 * (
            distance_vector[int(len(distance_vector) / 2)] + distance_vector[int(len(distance_vector) / 2) - 1])
    h = median / (2 * np.log(num_agents + 1))
    kernel = np.exp(distance_matrix[:, :] * (-1.0 / h))
    kernel_gradient = kernel[:, :, None] * (2.0 / h) * (params[None, :, :] - params[:, None, :])

    weights = (1.0 / temp) * kernel[:, :, None] * gradient[:, None, :] + kernel_gradient[:, :, :]
    weights = np.mean(weights[:, :, :], axis=0)
    gradients = np.copy(weights)

    return gradients #Let's just return the gradients and let individually chosen optimizers in the actor handle the update.
