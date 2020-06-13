import numpy as np
import sys
sys.path.append("..")
import os
from .funcs.variable import *
from tensorflow.keras import Model, layers
from tensorflow.keras.models import Sequential
import gym
from kmc_env.envs import KmcEnv #for kmc

tf.keras.backend.set_floatx('float64') #convert everything to float64. Avoids annoying tf warnings.

class CL_ConvNeuralNet_A2C(Model):
    # This defines two distinct models
    # One is an actor, another is the critic (value function estimation)
    # Both are neural networks that accept states as [image, numeric] inputs
    def __init__(self, input_dim=4, dim_actions=3, num_conv_filters_1=64,
                 num_conv_filters_2=64, num_hidden_nodes_1=128, num_hidden_nodes_2=128,
                 actor_lr = 0.003, critic_lr = 0.01):
        self.num_conv_filter_1 = num_conv_filters_1
        self.num_conv_filter_2 = num_conv_filters_2
        self.initializer =  tf.keras.initializers.he_uniform()
        self.num_hidden_nodes_1 = num_hidden_nodes_1
        self.num_hidden_nodes_2 = num_hidden_nodes_2
        self.input_dim = input_dim
        self.dim_actions = dim_actions
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr

        super(CL_ConvNeuralNet_A2C, self).__init__()
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor.optimizer = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.critic.optimizer = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)

    def build_actor(self):
        InputImage = layers.Input(shape=(16, 16, 1))
        InputNumeric = layers.Input(shape=(4,))

        cnet = layers.Conv2D(filters=self.num_conv_filter_1, kernel_size=(4, 4), strides=(2, 2),
                             activation=tf.nn.tanh,
                             kernel_initializer=self.initializer , name='conv1')(InputImage)
        cnet = layers.AveragePooling2D()(cnet)
        cnet = layers.Conv2D(self.num_conv_filter_2, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.tanh,
                             kernel_initializer=self.initializer , name='conv2')(cnet)
        cnet = layers.Flatten()(cnet)
        cnet = Model(inputs=InputImage, outputs=cnet)

        numeric = layers.Dense(self.num_hidden_nodes_1, activation=tf.nn.tanh,
                               kernel_initializer=self.initializer )(InputNumeric)

        numeric = layers.Dense(self.num_hidden_nodes_2, activation=tf.nn.tanh,
                               kernel_initializer=self.initializer )(numeric)

        numeric = Model(inputs=InputNumeric, outputs=numeric)

        combined = layers.concatenate([cnet.output, numeric.output])

        combined_network = layers.Dense(self.num_hidden_nodes_1, activation=tf.nn.tanh, kernel_initializer=self.initializer)(combined)

        combined_network = layers.Dense(self.num_hidden_nodes_2, activation=tf.nn.tanh, kernel_initializer=self.initializer)(combined_network)

        combined_network = layers.Dense(self.dim_actions * 2, activation='linear',
                                        kernel_initializer=self.initializer ,
                                        name='output_actions_layer')(combined_network)

        actor = Model(inputs=[cnet.input, numeric.input], outputs=combined_network)

        return actor

    def build_critic(self):
        # critic neural network
        InputImage = layers.Input(shape=(16, 16, 1))
        InputNumeric = layers.Input(shape=(4,))

        cnet = layers.Conv2D(filters=self.num_conv_filter_1, kernel_size=(4, 4), strides=(2, 2),
                             activation=tf.nn.tanh,
                             kernel_initializer=self.initializer , name='conv1')(InputImage)
        cnet = layers.AveragePooling2D()(cnet)
        cnet = layers.Conv2D(self.num_conv_filter_2, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.tanh,
                             kernel_initializer=self.initializer , name='conv2')(cnet)
        cnet = layers.Flatten()(cnet)
        cnet = Model(inputs=InputImage, outputs=cnet)

        numeric = layers.Dense(self.num_hidden_nodes_1, activation=tf.nn.tanh,
                               kernel_initializer=self.initializer )(InputNumeric)

        numeric = layers.Dense(self.num_hidden_nodes_2, activation=tf.nn.tanh,
                               kernel_initializer=self.initializer )(numeric)

        numeric = Model(inputs=InputNumeric, outputs=numeric)

        combined = layers.concatenate([cnet.output, numeric.output])

        combined_network = layers.Dense(self.num_hidden_nodes_1, activation=tf.nn.tanh, kernel_initializer=self.initializer)(combined)

        combined_network = layers.Dense(self.num_hidden_nodes_2, activation=tf.nn.tanh, kernel_initializer=self.initializer)(combined_network)

        combined_network = layers.Dense(1, activation='linear', kernel_initializer=self.initializer,
                                        name='value_estimate')(combined_network)

        critic = Model(inputs=[cnet.input, numeric.input], outputs=combined_network)

        return critic

    # define forward pass
    def call(self, states):
        actions_output = tf.reshape(self.actor(states), (-1, self.dim_actions, 2))
        value_estimate = self.critic(states)

        return actions_output, value_estimate

    def set_weights(self, weights):
        #SVGD will call this to set weights
        self.actor.set_weights(weights)

    def save_actor_critic(self, save_folder_name, world_rank, iteration):
        self.actor.save_weights(os.path.join(save_folder_name, 'Actor_agent_'+str(world_rank) + str(iteration) + '.h5'))
        self.critic.save_weights(os.path.join(save_folder_name, 'Critic_agent_' + str(world_rank) + str(iteration) + '.h5'))

class CL_NeuralNet_A2C(Model):
    #This defines an actor-critic model but with only fully connected layers (no convolutions)

    def __init__(self, num_hidden_nodes_1=24, num_hidden_nodes_2=1, dim_actions=2,actor_lr = 0.005, critic_lr = 0.01):
        self.num_hidden_nodes_1 = num_hidden_nodes_1
        self.num_hidden_nodes_2 = num_hidden_nodes_2
        self.dim_actions = dim_actions
        self.critic_lr = critic_lr
        self.actor_lr = actor_lr
        super(CL_NeuralNet_A2C, self).__init__()
        self.actor = self.build_actor()
        self.critic = self.build_critic()

    def build_actor(self):
        # Fully-Connected layer 1
        actor = Sequential()
        actor.add(layers.Dense(self.num_hidden_nodes_1, input_dim=4, activation=tf.nn.relu,
                                 kernel_initializer='he_uniform',
                               name='fc_1'))
        actor.add(layers.Dense(self.num_hidden_nodes_2, activation=tf.nn.relu,
                                 kernel_initializer='he_uniform', name='fc_2'))
        actor.add(layers.Dense(self.dim_actions, activation=tf.nn.softmax,
                                                kernel_initializer='he_uniform',
                                                name='output_actions_layer'))

        actor.compile(loss="categorical_crossentropy", optimizer=tf.optimizers.Adam(lr=self.actor_lr))

        return actor

    def build_critic(self):
        #critic neural network
        critic = Sequential()
        critic.add(layers.Dense(self.num_hidden_nodes_1, input_dim=4, activation=tf.nn.relu,
                                 kernel_initializer='he_uniform', name='fc_1_vs'))

        critic.add(layers.Dense(self.num_hidden_nodes_1, activation=tf.nn.relu,
                                    kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3), name='fc_2_vs'))

        critic.add(layers.Dense(1, activation='linear',
                                                kernel_initializer='he_uniform',
                                                name='output_value_layer'))
        critic.compile(loss = "mse", optimizer=tf.optimizers.Adam(lr=self.critic_lr))
        return critic

    # define forward pass
    def call(self, states, tf_actions=None, bool_actions_prob=True, bool_negative_log_prob=False):

        states_out = self.actor(states)
        value_estimate = self.critic(states)

        actions_combination_prob, negative_log_prob = None, None

        if bool_actions_prob:
            actions_combination_prob = tf.nn.softmax(states_out, name='actions_combination_prob')
        if bool_negative_log_prob:
            negative_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=states_out, labels=tf_actions)

        return actions_combination_prob, negative_log_prob, value_estimate

    def set_weights(self, weights):
        #SVGD will call this to set weights
        self.actor.set_weights(weights)

    def save_actor_critic(self, save_folder_name, world_rank):
        self.actor.save(os.path.join(save_folder_name, 'Actor_agent_'+str(world_rank)+'.h5'))
        self.critic.save(os.path.join(save_folder_name, 'Critic_agent_' + str(world_rank) + '.h5'))


''' KMC Environment Initialization Class'''
class CL_KMCEnv:
    def __init__(self, agent_idx=0, number_of_episodes = 250):
        self.agent_idx = agent_idx
        self.episodes = number_of_episodes
        wdir = r'../../kmc-openai-env/kmc-openai-env/kmc_env/envs/data'
        env = KmcEnv(target_roughness=0.80,
             reward_type='gaussian',reward_multiplier=50,reward_tolerance=0.07,
             rates_spread=0.1,rates_adjustment=5,folder_with_params=wdir)
        self.env = env
        self.dim_actions = 3
        self.save_path = r'svpg_results/KMC_lr1E2/'

        #KMCSIM creates a strange sort of state input: A surface projection image of size (16,16) and
        # a numeric vector length 3, that has the current simulation parameters (deposition rates, temperature)
        # this should fully define the behavior of the system moving forward.

        self.state_descriptors = [(16,16), (1,4)]