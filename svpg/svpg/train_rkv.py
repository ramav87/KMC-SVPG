
import os
import sys
sys.path.append("..")
import tensorflow as tf
import numpy as np
from itertools import chain
from .vpg import CL_ConvNeuralNet_A2C, CL_KMCEnv
from .svgd_kernel import stein_update_policies
from mpi4py import MPI
import horovod.tensorflow as hvd
tf.keras.backend.set_floatx('float32')
import tensorflow_probability as tfp

hvd.init()

#Communications setup
comm_world = MPI.COMM_WORLD
global world_rank, world_size, world_rank_hvd
world_rank = comm_world.Get_rank()
world_rank_hvd = hvd.local_rank()
world_size = comm_world.Get_size()

num_agents_per_gpu = 4

#device setup
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpu_devices[hvd.rank()//num_agents_per_gpu ], 'GPU')
tf.config.experimental.set_memory_growth(gpu_devices[hvd.rank()//num_agents_per_gpu ], True)

#print function
def print_rank(*args, **kwargs):
    if world_rank == 0:
        print(*args, **kwargs)

tf.keras.backend.set_floatx('float64') #convert everything to float64. Avoids annoying tf warnings.

def train_svpg(number_of_episodes = 400,  pure_a2c=False, run = 1, batch_size_ep = 20):
    '''
    SVPG or A2C training on the KMC environment
    Inputs: - number of episodes (int)
            - pure A2C: Boolean, specifies whether the trainign will be N independent agents,
                or via SVPG. Default is False (i.e., SVPG updates are the default)
            - run: run number, used for repeating the process. Runs are stored in different folders. Default is 1.
            - batch_size_ep: batch size for SVPG update, in terms of number of episodes. Defauly 20.

    '''

    print_rank("We are now at run = " + str(run))


    ''' Generate Single instance of the environment '''
    INS_Env = CL_KMCEnv(number_of_episodes=number_of_episodes)

    dim_actions = INS_Env.dim_actions #Continuous action space.
    gamma = 0.90

    ''' Generate NeuralNet instance -- A2C algorithm '''
    NeuralNet =  CL_ConvNeuralNet_A2C(num_conv_filters_1=32, num_conv_filters_2=32,
                num_hidden_nodes_1=128, num_hidden_nodes_2=128,
                 dim_actions=dim_actions, actor_lr=0.005, critic_lr=0.01)

    # Keep track of things
    running_reward_list = []
    all_epoch_rewards = []

    # lists of states,actions, targets
    state_image_history = []
    state_numerics_history = []
    actions_history = []
    target_history = []
    advantage_history = []
    print_rank('number of episodes is {}'.format(number_of_episodes))

    if pure_a2c==True: save_folder_name = os.path.join(INS_Env.save_path, 'A2C_Results_Run=' + str(run))
    elif pure_a2c==False: save_folder_name = os.path.join(INS_Env.save_path, 'A2C_SVPG_CNETUpdates_Results_Run=' + str(run))

    for iteration in range(number_of_episodes):
        print_rank('Episode: {}'.format(iteration))
        reward_list = []
        done = False
        state,_ = INS_Env.env.reset(mode = 'hard')
        reward_total = 0

        while not done:

            # Take an action
            #Reshape the state space if needed.
            if state[0].shape[0] == 16:
                state = [state[0][None, :, :, None], state[1]]

            actions_output, value_estimate = NeuralNet(state)

            # actions are given by mu, sigma (action_dims x 2) tensor

            # Sample the policy to get the action
            output_action = tfp.distributions.MultivariateNormalDiag(
                actions_output[0,:, 0], tf.nn.softplus(actions_output[0,:, 1])).sample(1)

            output_actions = np.squeeze(tf.clip_by_value(output_action, -10, 10).numpy())

            # Take the selected action in the environment
            next_state, reward, done = INS_Env.env.step(output_actions)

            next_state = [next_state[0][None, :, :, None], next_state[1]]

            reward_total += reward

            # Now predict the value of the next state.
            actions_output_next, value_estimate_next = NeuralNet(next_state)

            # Calculate the TD error:
            # target = r + gamma * V(next_state)

            if not done:
                # td_error = target - V(s)
                target = reward + gamma * value_estimate_next
                td_error = target - NeuralNet.critic(state)
                # advantage = reward - V(s)
                advantage = reward - value_estimate_next
                target = tf.constant(np.array(target).reshape(-1, 1), dtype=tf.double)
            else:
                target = reward
                td_error = target - NeuralNet.critic(state)
                advantage = tf.constant(np.array(reward).reshape(-1,1), dtype = tf.double)
                target = tf.constant(np.array(target).reshape(-1, 1), dtype=tf.double)

            # Store transitions

            state_image_history.append(state[0])
            state_numerics_history.append(state[1])
            actions_history.append(output_action) #Here we will have the (mu,sigma) and (actual actions sampled).
            target_history.append(target[None,:])
            advantage_history.append(advantage[None,:])
            reward_list.append(reward)

            state = next_state


        ''' Calculate running reward '''
        epoch_rewards_sum = sum(reward_list)
        all_epoch_rewards.append(epoch_rewards_sum)
        running_reward_list.append(epoch_rewards_sum)


        if iteration > 1 and iteration % 1 == 0:

            print_rank('Episode {} with Reward {}, Max Reward this agent so far: {}'.format(iteration,
                                                                                 epoch_rewards_sum,
                                                                                 np.max(all_epoch_rewards)))
            if not os.path.exists(save_folder_name):
                os.mkdir((save_folder_name))

            filename_rewards = 'Agent_' + str(world_rank_hvd) + '_reward_results'
            filename_actions = 'Agent_' + str(world_rank) + '_action_results'
            np.savetxt(os.path.join(save_folder_name, filename_rewards), np.array(all_epoch_rewards), delimiter='\t',
                       fmt='%.2f')
            np.save(os.path.join(save_folder_name, filename_actions), np.array(state_numerics_history))

            # every ten epochs save the model
            if iteration > 90 and iteration % 100 == 0: NeuralNet.save_actor_critic(save_folder_name, world_rank, iteration)

        #Batch size is 6 * number of episodes per batch
        batch_size = batch_size_ep*6

        if iteration >= batch_size_ep and iteration%batch_size_ep == 0:

            len_states = len(state_image_history)

            state_history_tf = [tf.squeeze(tf.stack(state_image_history[-min(len_states, batch_size):]), axis=1),
                                tf.squeeze(tf.stack(state_numerics_history[-min(len_states, batch_size):]), axis=1)]

            advantage_history_tf = tf.squeeze(tf.stack(advantage_history[-min(len_states, batch_size):]), axis=1)
            actions_history_tf = tf.squeeze(tf.stack(actions_history[-min(len_states, batch_size):]), axis=1)
            target_history_tf = tf.squeeze(tf.stack(target_history[-min(len_states, batch_size):]), axis=1)

            #If we are in A2C mode, then just update the actor and critic with losses calculated from the above rollouts:
            if pure_a2c == True:
                with tf.GradientTape() as tape:
                    actions_mean_tt = tf.reshape(NeuralNet.actor(state_history_tf), (-1, 3, 2))

                    lognorm_dist = tfp.distributions.MultivariateNormalDiag(
                        actions_mean_tt[:, :, 0],
                        tf.nn.softplus(actions_mean_tt[:, :, 1])).log_prob(actions_history_tf)

                    loss = -lognorm_dist * advantage_history_tf[:,0,0]  # calculate negative log probability times advantage (td_error)
                    gradients = tape.gradient(loss, NeuralNet.actor.trainable_variables)
                #Update the actor with chosen optimizer
                NeuralNet.actor.optimizer.apply_gradients(zip(gradients, NeuralNet.actor.trainable_variables))

                # for the critic, let's use this to update the value estimation
                with tf.GradientTape() as tape:
                    td_error = target_history_tf[:,0,0] - NeuralNet.critic(state_history_tf)[:,0] # TD_error
                    grads = tape.gradient(td_error ** 2, NeuralNet.critic.trainable_variables)

                NeuralNet.critic.optimizer.apply_gradients(zip(grads, NeuralNet.critic.trainable_variables))

            #But if we want to do SVPG, we need to calculate gradients but NOT apply them:
            elif pure_a2c==False:

                #Let's update the critic in the normal way

                with tf.GradientTape() as tape:
                    td_error = target_history_tf[:,0,0] - NeuralNet.critic(state_history_tf)[:,0] # TD_error
                    grads = tape.gradient(td_error ** 2, NeuralNet.critic.trainable_variables)

                NeuralNet.critic.optimizer.apply_gradients(zip(grads, NeuralNet.critic.trainable_variables))

                #Now we update the actor with SVPG
                print_rank("Performing SVGD Update")

                # do an svpg update. For this, first calculate policy objective to get policy gradient
                with tf.GradientTape() as tape:
                    actions_mean_tt = tf.reshape(NeuralNet.actor(state_history_tf), (-1, 3, 2))

                    lognorm_dist = tfp.distributions.MultivariateNormalDiag(
                        actions_mean_tt[:, :, 0],
                        tf.nn.softplus(actions_mean_tt[:, :, 1])).log_prob(actions_history_tf)

                    loss = lognorm_dist * advantage_history_tf[:,0,0]  # gradient of objective function
                    gradients = tape.gradient(loss, NeuralNet.actor.trainable_variables)

                # weights/biases (list of Numpy arrays)
                weights = NeuralNet.actor.trainable_variables

                #do an all gather
                stein_temp = 5.0
                all_agent_scores = hvd_gather_scores([epoch_rewards_sum])
                print_rank('Mean Reward from all agents so far: {}'.format(np.mean(all_agent_scores)))
                all_agent_weights, all_agent_grads, shape_list = hvd_gather_parameters_gradients( [weights], [gradients])
                all_agent_gradients = SVPG_Updates(all_agent_weights, all_agent_grads,
                                                                      num_agents=world_size,
                                                                      temp=stein_temp)


                this_agent_gradients = all_agent_gradients[world_rank]
                # need to reshape gradients back
                updated_gradients = []
                offset = 0

                for (i,itm) in enumerate(weights):
                    new_shape = itm.shape
                    chunk = np.prod(new_shape)

                    if i == len(weights):
                        new_grad = this_agent_gradients[offset+chunk:]
                    else:
                        new_grad = this_agent_gradients[offset:offset + chunk]
                    offset = chunk
                    new_grad = tf.reshape(new_grad, new_shape)
                    updated_gradients.append(tf.constant(new_grad))

                NeuralNet.actor.optimizer.apply_gradients(zip(updated_gradients, NeuralNet.actor.trainable_variables))

                if world_rank_hvd==0:
                    actions_mean_tt = tf.reshape(NeuralNet.actor(state_history_tf), (-1, 3, 2))
                    lognorm_dist = tfp.distributions.MultivariateNormalDiag(
                        actions_mean_tt[:, :, 0],
                        tf.nn.softplus(actions_mean_tt[:, :, 1])).log_prob(actions_history_tf)

                    loss_after = lognorm_dist * advantage_history_tf[:,0,0]  # gradient of objective function

                    print("Objective Actor before update: " + str(tf.reduce_mean(loss).numpy()))
                    print("Objective Actor AFTER SVPG update: " + str(tf.reduce_mean(loss_after).numpy()))

                    #Now let's examine the critic loss behavior

                    td_error_after = target_history_tf[:, 0, 0] - NeuralNet.critic(state_history_tf)[:, 0]  # TD_error
                    print("Loss Critic before update: " + str(tf.reduce_mean(td_error**2).numpy()))
                    print("Loss Critic AFTER SVPG update: " + str(tf.reduce_mean(td_error_after**2).numpy()))

    return

def hvd_gather_parameters_gradients(list_of_weights, list_of_gradients):
    ''' gather the parameters and gradients from each individual "train_svpg" function call
    Input: List of "weights" from all agents for one batch  (single weights is a list of tensor of weights/biases)
           List of "gradients" from all agents for one batch (single gradienst is a list of Tensors of gradients)

    Output:  2D tf tensor stacked from flattened “trainable_variables” Tensors
             2D tf tes stacked from flattended "gradients" Tensors
             List of the shapes for the weights/gradients
             List of sizes for the weights/gradients
    '''

    # assume 1 mpi process per agent

    # flatten
    weight_list = list(chain.from_iterable(list_of_weights))
    shape_list = [itm.shape for itm in weight_list]
    grad_list = list(chain.from_iterable(list_of_gradients))

    # Convert single list of gradients Tensors to list of numpy arrays
    flat_gradients = tf.concat([tf.reshape(g,[-1]) for g in grad_list],axis=0)
    flat_gradients = tf.expand_dims(flat_gradients, axis=0)

    flat_weights = tf.concat([tf.reshape(vec,[-1]) for vec in weight_list], axis=0)
    flat_weights = tf.expand_dims(flat_weights, axis=0)

    gather_weight_tensor = hvd.allgather(flat_weights)
    gather_gradient_tensor = hvd.allgather(flat_gradients)

    return gather_weight_tensor, gather_gradient_tensor, shape_list

def hvd_gather_scores(list_of_epoch_scores):
    score_list = list(list_of_epoch_scores)

    flat_scores = tf.concat([tf.reshape(vec,[-1]) for vec in score_list], axis=0)
    flat_scores = tf.expand_dims(flat_scores, axis=0)

    gather_score_tensor = hvd.allgather(flat_scores)

    return gather_score_tensor

def SVPG_Updates(weights_2d_np, gradients_2d_np, **kwargs):
    ''' Do SVPG update step for one epoch (episode = epoch here), based on the stacked parameters and gradients
    Input: 2D Numpy array stacked from flattened “weights” Numpy arrays
           2D Numpy array stacked from flattended "gradients" Tensors

    Output: UPDATED 2D Numpy array for "weights"
    '''
    return stein_update_policies(weights_2d_np, gradients_2d_np,  **kwargs)




