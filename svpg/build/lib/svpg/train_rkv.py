import pandas as pd
from pandas import HDFStore
import tensorflow as tf
import numpy as np
import random
import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import os
import itertools
import gym
from .vpg import CL_CartPole, CL_SimplePolicyGradient, CL_NeuralNet
from .svgd_kernel import stein_update_policies

import json


def train_single(number_of_agents = 2, number_of_episodes = 100, 
            number_of_timesteps = 20, num_hidden_nodes_1 = 128, num_hidden_nodes_2 = 128):

    ''' Generate Single instance of the environment '''
    INS_Env = CL_CartPole()
    
    print('--- Successfully intialized the KmcEnv')
    

    state, reward,_,_ = INS_Env.env.reset()  # from original KMC code --- initial reset
    
    print('Successfully reset the Environment ')

    ''' Shape, size of the state '''
    state_shape = np.array(INS_Env.env.state).shape
    state_dim = state_shape# * state_shape[1]  # Input dimension for flattened 2D image
    print('--- The input dimension is {}'.format(state_dim))
    dim_actions = 2  # CartPole has two
    learning_episode_arr = np.zeros((INS_Env.episodes, 1))

    ''' Generate Single SimplePolicyGradient instance '''
    # Instantiate model, define some variables
    RL = CL_SimplePolicyGradient(
        inputs_dim=state_dim,
        dim_actions=dim_actions,
        lr=0.01,
        reward_decay=0.95,
        output_graph=False,
    )
    ''' Generate NeuralNet instance -- add dictionary for NN structures, hyperparameters as arguments later '''
    NeuralNet = CL_NeuralNet()   ## NN take x, tf_actions as input, return actions_combination_prob, negative_log_prob

    # TODO: Define an optimizer (distributed)
    # create single optimizer instance with uniform learning rate self.lr
    learning_rate = 0.01
    INS_optimizer = tf.optimizers.SGD(learning_rate)
    # INS_optimizer = tf.optimizers.Adam(learning_rate)

    # TODO: calculate gradients, allgather, apply kernel, apply gradient from list

    # (1) The gradients calculations, probability calculations, weights/bias extractions, loss calculations will be conducted by function inside RL instance:
    # training_calculate_gradients(self, neural_net, tf_states, tf_actions, tf_rewards)
    # return actions_combination_prob, negative_log_prob, trainable_variables, gradients, loss

    # (2) allgather, apply kernel, apply gradient from list --> see functions at the bottom

    # TODO: Collect training operations, interactions with environment,

    # The individual training operations, environment interactions will happen within this 'train_single' function


    print('Successfully initialized the SinplePolicyGradient instance ')

    print('Printing {state}: '.format(state=state))
    print('Printing {reward}: '.format(reward=reward))

    done = False

    time_step_list = []
    # reward_list = []
    action_list = []
    rms_list = []
    single_episode_dict = {}

    # Train
    for iter in range(100):
        # print(iter)
        reward_list = []
        done = False
        state = INS_Env.env.reset()
        state = np.array(state)
        ''' (1) Reset the environment and obtain the observations !!! reset every epoch not every step'''
        # state, reward = INS_KmcEnv.env.reset()
        observation = np.reshape(state, (1, np.array(state_dim)[0]))  # flatten to shape (1, state_dim)
        #print(observation)
        #print(observation.shape)
        time_step = 0
        while not done:

            # ''' (1) Reset the environment and obtain the observations !!! reset every epoch not every step'''
            # # state, reward = INS_KmcEnv.env.reset()
            # observation = np.reshape(state, (1, state_dim))  # flatten to shape (1, state_dim)
            # print('--- Printing reshaped observation')
            # print(observation)

            ''' (2) Choose actions based on the observations by using the RL prediected probability '''
            # print(observation)
            # action_idx = RL.choose_actions(observation)  # action idx, need to be used for picking actual action from action space (2d arr)
            # print('--- Printing action_idx choosen')
            # print(action_idx)
            # action = actions_combinations[action_idx]
            # # print('--- Action chosen: {}'.format(action))
            ''' TF 2.0 native code '''
           
            
            output_weights,_ = NeuralNet(observation,tf_actions=None,  bool_actions_prob=True, bool_negative_log_prob=False)
            #print(output_weights)
            #action_idx = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.numpy().ravel())
            action = np.argmax(output_weights) #actions_combinations[action_idx]
            action_idx = action
            
            ''' (3) Obtain the new observations and rewards based on the actions just chosen using the environment '''
            new_observation, reward, done,_ = INS_Env.env.step(action)
            
            new_observation = np.array(new_observation)
            #print(new_observation)
            # print('******** printing observation')
            #print(new_observation.shape)
            # print('******** printing new_observation')
            # print(np.reshape(new_observation, (1, state_dim)).shape)

            ''' (4) Store the intermediate informations (observations, action_idx, reward) - will be used for NN learning '''
            RL.store_intermediate_results(observation, action_idx, reward)

            ''' (5) Assign new_observation to observation '''
            observation = np.reshape(new_observation, (1, np.array(state_dim)[0]))

            ''' (6) Calculate the roughness, append results to lists '''
            #rms_list.append(calc_roughness(observation))
            time_step_list.append(time_step)
            reward_list.append(reward)


            ''' (5) Calculate the running reward and conduct any evaluations '''
            # if done:
            #     epoch_reward_sum = sum(RL.epoch_reward)
            #

            ''' (6) Runing NN training step, conduct the computations based pre-defined graph and update the probabilities '''

            time_step += 1

            #print('Timestep: {}, reward: {:.4}, RMS: {:.3}, action: {}'.format(time_step, reward,
            #                                                                    calc_roughness(observation),
            #                                                                    action))
            if time_step == number_of_timesteps:
                done = True
                print('Maxsteps: {} reached !!!!'.format(number_of_timesteps))

                ''' Calculate running reward '''
                epoch_rewards_sum = sum(reward_list)
                if 'running_reward' not in globals():
                    running_reward = epoch_rewards_sum
                else:
                    running_reward = running_reward * 0.99 + epoch_rewards_sum * 0.01
                # if running_reward > some_kind_of_threadshold:
                #                 #     do something

                print('Episode: {}, running_reward: {}'.format(iter, running_reward))
                ''' After one epoch --> update RL-NN to update policy '''
                # vt = RL.learn()


                ''' misc from original learn() function --- Prepare the inputs of the NeuralNet and gradients, loss calculations '''
                epoch_actions_idx_arr = np.array(RL.epoch_actions_idx).astype(int)  # (n, ) 1D numpy array for tf_actions input
                epoch_observation = np.vstack(RL.epoch_observations)                # for tf_states input
                discounted_epoch_reward_norm = RL.discount_normalize_rewards(RL.epoch_rewards, RL.gamma) # for loss calculation in 'calculate_gradients' function


                ''' added params, gradients calculation function CALL '''
                actions_combination_prob, negative_log_prob, trainable_variables, weights, gradients, loss = \
                    RL.training_calculate_gradients(NeuralNet, epoch_observation, epoch_actions_idx_arr, discounted_epoch_reward_norm)

                ''' added optimizer function CALL '''
                # INS_optimizer.minimize(loss, trainable_variables)
                INS_optimizer.apply_gradients(zip(gradients, NeuralNet.trainable_variables))   # Vanilla VPG update

                ''' SVPG updates (initial ideas)
                1. Send weights and gradients outside the "train_single" function for Multi-Agent SVPG updates

                i.e. return weights, gradients

                2. Gather all "weights" and "gradients" together and do SVPG updates

                3. Re-assign weights outside the train_single function by accessing the list of NeuralNet instances (agents)

                '''

        # return weights, gradients



def gather_parameters_gradients(list_of_weights_np, list_of_gradients_tensor):
    ''' gather the parameters and gradients from each individual "train_single" function call for every single epoch
    Input: List of "weights" from all agents for one epoch   (single weights is a list of Numpy arrays of weights/biases)
           List of "gradients" from all agents for on epoch (single gradienst is a list of Tensors of gradients)

    Output:  2D Numpy array stacked from flattened “trainable_variables” Tensors
             2D Numpy array stacked from flattended "gradients" Tensors
             List of the shapes for the weights/gradients
             List of sizes for the weights/gradients
    '''
    shape_list = [w.shape for w in list_of_weights_np[0]]
    size_list = [w.size for w in list_of_weights_np[0]]
    single_agent_weights_concat_1D_arr = np.array([])
    single_agent_gradients_concat_1D_arr = np.array([])

    num_agents = len(list_of_weights_np)
    num_weights_section = len(list_of_weights_np[0])

    for i in range(num_agents):
        # Convert single list of gradients Tensors to list of numpy arrays
        single_gradients_np = [g.numpy() for g in list_of_gradients_tensor[i]]
        for j in range(num_weights_section):
            ## weights
            single_agent_weights_concat_1D_arr = np.concatenate((single_agent_weights_concat_1D_arr, list_of_weights_np[i][j].flatten()))
            ## gradients
            single_agent_gradients_concat_1D_arr = np.concatenate((single_agent_gradients_concat_1D_arr, single_gradients_np[j].flatten()))
        if i == 0:
            weights_2d_np = single_agent_weights_concat_1D_arr
            gradients_2d_np = single_agent_gradients_concat_1D_arr
        else:
            weights_2d_np = np.vstack((weights_2d_np, single_agent_weights_concat_1D_arr))
            gradients_2d_np = np.vstack((gradients_2d_np, single_agent_gradients_concat_1D_arr))

    return weights_2d_np, gradients_2d_np, shape_list, size_list


def SVPG_Updates(weights_2d_np, gradients_2d_np):
    ''' Do SVPG update step for one epoch based on the stacked parameters and gradients
    Input: 2D Numpy array stacked from flattened “weights” Numpy arrays
           2D Numpy array stacked from flattended "gradients" Tensors

    Output: UPDATED 2D Numpy array for "weights"
    '''
    return stein_update_policies(weights_2d_np, gradients_2d_np)



def reassign_weights(updated_weights_2d_np, NeuralNet_instance_list, shape_list, size_list):
    ''' Re-assign back the updated parameters to each individual NeuralNet instance
    Input: Updated 2D Numpy array for "weights"
           List of NeuralNet Instance
           Shape list and size list
    Operation: (1) Extract all rows of 2D array in a loop to 1D vector (or if there's any vectorization methods to avoid explicit for-loop)
               (2) Unpack the 1D vector to single "weights" (list of Numpy arrays) by using the function in "util.py"
               (3) Re-assign the updated "weights" for each agent instance by using the method NeuralNet.set_weights(weights)
    '''
    num_agents = len(NeuralNet_instance_list)
    num_sections = len(shape_list)
    for i in range(num_agents):
        single_agent_packed_list_of_weights = []
        for j in range(num_sections):

            updated_single_agent_1d_weights = updated_weights_2d_np[i, :]
            sliced_1d_arr = updated_single_agent_1d_weights[0:size_list[j]]
            reshaped_arr = sliced_1d_arr.reshape(shape_list[j])
            single_agent_packed_list_of_weights.append(reshaped_arr)     # list of updated weights (numpy array) for one agent

        # Assign to the specific agent
        NeuralNet_instance_list[i].set_weights(single_agent_packed_list_of_weights)


                # plt.plot(vt)  # plot the episode vt
                # plt.xlabel('episode steps')
                # plt.ylabel('normalized state-action value')
                # plt.show()

    # print(RL.epoch_observations)
    # print(RL.epoch_actions)
    # print(RL.epoch_rewards)



def train_multiple():
    pass


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_path = sys.argv[-1] if os.path.exists(sys.argv[-1]) else sys.exit()
        wdir = sys.argv[-2]
        train_single(wdir=wdir, input_path=sys.argv[-1])