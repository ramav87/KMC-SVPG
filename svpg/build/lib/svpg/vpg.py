'''
Siyan Liu
Multiple Agents VPG for average cumulative returns
Serial version
-- Two hidden layers
-- Version 2
--- Modify the code to be TF 2.0 compatible
--- Try to create scope of the graph to avoid the 'bloated graph' and memory explosion
--- Create new loss function system by harness the tf.nn.sigmoid_cross_entropy_with_logits which can be used for classification problems
    that the labels are not mutually exclusive (The original Softmax is good for classification problems where labels are mutually exclusive)
--- Try to add weight (say 0.01) for the summation of rewards of each epoch when adding to the overall rewards (say 0.99 for existing summation of rewards)

'''

import pandas as pd
from pandas import HDFStore
import tensorflow as tf
# tf.compat.v1.disable_eager_execution()
import numpy as np
import random
import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import os
import itertools
from .funcs.variable import *
from tensorflow.keras import Model, layers
import json
import gym


number_of_agents = 2
number_of_episodes = 1000
number_of_timesteps = 200

num_hidden_nodes_1 = 500
num_hidden_nodes_2 = 200


''' 10.29.2019 Siyan Liu --- try to seperate the environment (KMC) class and the NN class '''
''' PG Class'''
class CL_SimplePolicyGradient:
    def __init__(self,
                 inputs_dim=None,
                 lr=0.001,
                 dim_actions=None,
                 reward_decay=0.95,
                 output_graph=False,
                 ):
        self.inputs_dim = inputs_dim
        self.dim_actions = dim_actions
        self.lr = lr
        self.gamma = reward_decay

        # list for observations, actions based this observations, summation of the rewards for one epoch
        self.epoch_observations, self.epoch_actions_idx, self.epoch_rewards = [], [], []


    def store_intermediate_results(self, state, action_idx, reward):
        self.epoch_observations.append(state)
        self.epoch_actions_idx.append(action_idx)
        self.epoch_rewards.append(reward)

    def discount_normalize_rewards(self, epoch_rewards, gamma): # discount epoch rewards
        discounted_epoch_rewards = np.zeros_like(self.epoch_rewards)
        running_add = 0
        for t in reversed(range(0, len(epoch_rewards))):
            running_add = running_add * gamma + self.epoch_rewards[t]
            discounted_epoch_rewards[t] = running_add
        # normalize episode rewards
        discounted_epoch_rewards -= np.mean(discounted_epoch_rewards)
        discounted_epoch_rewards /= np.std(discounted_epoch_rewards)
        return discounted_epoch_rewards


    ''' 11.05.2019 added loss function for TF2 native code --- reward guided loss '''
    def loss_func(self, negative_log_prob, tf_rewards):
        loss = tf.math.reduce_mean(input_tensor=negative_log_prob * tf_rewards)
        return loss

    ''' 11.05.2019 added calculate gradients function for TF2 native code  '''
    def training_calculate_gradients(self, neural_net, tf_states, tf_actions, tf_rewards):
        # Wrap computation inside a GradientTape for automatic differentiation.
        with tf.GradientTape() as g:
            # Forward pass
            #This needs oversight. Check with original!!
            actions_combination_prob, negative_log_prob = neural_net(tf_states, tf_actions=tf_actions)   # is_training=True
            # Compute the loss
            loss = self.loss_func(negative_log_prob, tf_rewards)

        # trainable variables. (list of Tensors)
        trainable_variables = neural_net.trainable_variables

        # weights/biases (list of Numpy arrays)
        weights = neural_net.get_weights()


        # Compute gradients  (list of tensors with different shapes)
        gradients = g.gradient(loss, trainable_variables)
        #print('printing type of gradients')
        #print(type(gradients))
        #print(gradients)

        return actions_combination_prob, negative_log_prob, trainable_variables, weights, gradients, loss


''' Neural Network Class inherit from the tensorflow.keras.Model class --- set NN structure without placeholder '''
class CL_NeuralNet(Model):
    def __init__(self):
        num_hidden_nodes_1 = 128
        num_hidden_nodes_2 = 128
        dim_actions = 2
        super(CL_NeuralNet, self).__init__()
        
        # Fully-Connected layer 1
        self.fc_1 = layers.Dense(num_hidden_nodes_1, activation=tf.nn.tanh,
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3), name='fc_1')
        self.fc_2 = layers.Dense(num_hidden_nodes_2, activation=tf.nn.tanh,
                                 kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3), name='fc_2')
        self.output_action_layer = layers.Dense(dim_actions, activation=tf.nn.softmax,
                                                kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
                                                name='output_actions_layer')
    # define forward pass
    def call(self, states, tf_actions=None, bool_actions_prob=True, bool_negative_log_prob=True):
        states = self.fc_1(states)
        states = self.fc_2(states)
        states = self.output_action_layer(states)
        
        actions_combination_prob, negative_log_prob = None, None
        
        if bool_actions_prob:
            actions_combination_prob = tf.nn.softmax(states, name='actions_combination_prob')
        if bool_negative_log_prob:
            negative_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=states, labels=tf_actions)

        # if not is_training:
        #     # tf cross entropy expect logits without softmax, so only
        #     # apply softmax when not training.
        #     actions_combination_prob = tf.nn.softmax(x, name='actions_combination_prob')
        
        return actions_combination_prob, negative_log_prob

''' CartPole Environment Initialization Class'''
class CL_CartPole:
    def __init__(self, agent_idx=0, episodes = 200):
        self.agent_idx = agent_idx
        self.episodes = episodes                    # int
        ''' Instantiation of the <KmcEnv> class in kmc_env.py file '''
        env = gym.make('CartPole-v1')
        self.env = env

if __name__=='__main__':
    # throwing errors '__main__' is not a package
    INS_SimplePolicyGradient = CL_SimplePolicyGradient()
    INS_Env = CL_CartPole()
# TODO: add instantiation of CL_simple to test
# TODO: add instantiation of CL_KmcEnv to test



