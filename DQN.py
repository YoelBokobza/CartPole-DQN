import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import gym
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import collections
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import box

from buffer import ReplayBuffer
from model import Network

import copy

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'not_done'))

env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
plt.ion()

# look for a gpu
if torch.cuda.is_available():
    device = torch.device(f"cuda:2")
else:
    device = torch.device('cpu')
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Parameters
network_params = {
    'state_dim': env.observation_space.shape[0],
    'action_dim': env.action_space.n,
    'hidden_dim': 64
}

training_params = {
    'batch_size': 256,
    'gamma': 0.95,
    'epsilon_start': 1.1,
    'epsilon_end': 0.05,
    'epsilon_decay': 0.95,
    'target_update': 'soft',     # use 'soft' or 'hard'
    'tau': 0.01,                 # relevant for soft update
    'target_update_period': 15,  # relevant for hard update
    'grad_clip': 0.1,
}

network_params = box.Box(network_params)
params = box.Box(training_params)

# ============================================================================
# Plotting function
def plot_graphs(all_scores, all_losses, all_errors, axes):
    axes[0].plot(range(len(all_scores)), all_scores, color='blue')
    axes[0].set_title('Score over episodes')
    axes[1].plot(range(len(all_losses)), all_losses, color='blue')
    axes[1].set_title('Loss over episodes')
    axes[2].plot(range(len(all_errors)), all_errors, color='blue')
    axes[2].set_title('Mean Q error over episodes')


# Training functions
def select_action(s,policy_net,epsilon):
    '''
    This function gets a state and returns an action.
    The function uses an epsilon-greedy policy.
    :param s: the current state of the environment
    :return: a tensor of size [1,1] (use 'return torch.tensor([[action]], device=device, dtype=torch.long)')
    '''
    #with epsilon probability
    if np.random.uniform() < epsilon:
        # This case we choose random action
        # Pick action randomly
        action = np.random.choice(network_params.action_dim)
        # return the action as [1,1] tensor
        return torch.tensor([[action]], device=device, dtype=torch.long)
    else:
        # this case we choose action according to the greedy policy
        with torch.no_grad():  #To reduce memory usage
            return policy_net(s).max(1)[1].view(1, 1)


def train_model(buffer,policy_net,target_policy_net,optimizer):
    # Pros tips: 1. There is no need for any loop here!!!!! Use matrices!
    #            2. Use the pseudo-code.

    if len(buffer) < params.batch_size:
        # not enough samples
        return 0, 0

    # sample mini-batch
    transitions = buffer.sample(params.batch_size)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    next_states_batch = torch.cat(batch.next_state)
    reward_batch = torch.cat(batch.reward)
    not_done_batch = batch.not_done

    # Compute curr_Q = Q(s, a) - the model computes Q(s), then we select the columns of the taken actions.
    # Pros tips: First pass all s_batch through the network
    #            and then choose the relevant action for each state using the method 'gather'
    curr_Q = policy_net(state_batch).gather(1, action_batch)




    # Compute expected_Q (target value) for all states.

    #           
    # calculate the values for all next states ( Q_(s', argmax_a(Q_(s')) )
    next_state_value_function = torch.max(target_policy_net(next_states_batch), dim=1).values.detach()
    # masking next state's value with 0, where not_done is False (i.e., done) and calculate the target value.
    expected_Q = reward_batch + torch.mul(next_state_value_function * torch.tensor(not_done_batch, device=device), params.gamma)
    #expand the dimension of expected_Q such that it will be fitted to curr_Q dimensions
    expected_Q = expected_Q.unsqueeze(1) 

    # Compute Huber loss. Smoother than MSE
    loss = F.smooth_l1_loss(curr_Q, expected_Q)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # clip gradients to help convergence
    nn.utils.clip_grad_norm_(policy_net.parameters(), params.grad_clip)
    optimizer.step()


    estimation_diff = torch.mean(curr_Q - expected_Q).item()

    return loss.item(), estimation_diff

# ============================================================================
def cartpole_play():

    FPS = 25
    visualize = 'True'

    env = gym.make('CartPole-v1')
    env = gym.wrappers.Monitor(env,'recording',force=True)
    net = Network(network_params, device).to(device)
    print('load best model ...')
    net.load_state_dict(torch.load(f'best_{params.target_update}.dat',map_location=torch.device('cpu')))

    print('make movie ...')
    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()

    while True:
        start_ts = time.time()
        if visualize:
            env.render()
        state_v = torch.tensor(np.array([state], copy=False)).float()
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)
        c[action] += 1
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        if visualize:
            delta = 1 / FPS - (time.time() - start_ts)
            if delta > 0:
                time.sleep(delta)
    print("Total reward: %.2f" % total_reward)
    print("Action counts:", c)
    env.close()

# ============================================================================
def run_exp(max_episodes, buffer_mode):
    # Build neural networks
    policy_net = Network(network_params, device).to(device)
    # Build a target network
    target_policy_net = Network(network_params, device).to(device)
    for target_param, local_param in zip(target_policy_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(local_param.data)
    target_policy_net.eval()
    # target_policy_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters())
    if buffer_mode is True:
        buffer = ReplayBuffer(100000)
    else:
        buffer = ReplayBuffer(1)
    epsilon = params.epsilon_start

    # Training loop
    max_score = 500
    task_score = 0
    # performances plots
    all_scores = []
    all_losses = []
    all_errors = []
    fig, axes = plt.subplots(3, 1)

    # train for max_episodes
    for i_episode in range(max_episodes):
        epsilon = max(epsilon * params.epsilon_decay, params.epsilon_end)
        ep_loss = []
        ep_error = []
        # Initialize the environment and state
        state = torch.tensor([env.reset()], device=device).float()
        done = False
        score = 0
        for t in count():
            # Select and perform an action
            action = select_action(state,policy_net,epsilon)
            next_state, reward, done, _ = env.step(action.item())
            score += reward

            next_state = torch.tensor([next_state], device=device).float()
            reward = torch.tensor([reward], device=device).float()
            # Store the transition in memory
            buffer.push(state, action, next_state, reward, not done)

            # Update state
            state = next_state

            # Perform one optimization step (on the policy network)
            loss, Q_estimation_error = train_model(buffer,policy_net,target_policy_net,optimizer)

            # save results
            ep_loss.append(loss)
            ep_error.append(Q_estimation_error)

            # soft target update
            if params.target_update == 'soft':
                for target_param, local_param in zip(target_policy_net.parameters(), policy_net.parameters()):
                    target_param.data.copy_(params.tau * local_param.data + (1 - params.tau) * target_param.data)

            if params.target_update == 'None':
                for target_param, local_param in zip(target_policy_net.parameters(), policy_net.parameters()):
                    target_param.data.copy_(local_param.data)
            # hard target update. Copying all weights and biases in DQN

            if done or t >= max_score:
                print("Episode: {} | Current target score {} | Score: {}".format(i_episode + 1, task_score, score))
                break

        if params.target_update == 'hard':
            # update every params.target_update_period episodes the target network
            if i_episode % params.target_update_period == 0:
                for target_param, local_param in zip(target_policy_net.parameters(),
                                                     policy_net.parameters()):
                    target_param.data.copy_(local_param.data)
        # plot results
        all_scores.append(score)
        all_losses.append(np.average(ep_loss))
        all_errors.append(np.average(ep_error))
        plot_graphs(all_scores, all_losses, all_errors, axes)
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Copy the weights from policy_net to target_net after every x episodes

        # update task score
        if min(all_scores[-5:]) > task_score:
            task_score = min(all_scores[-5:])
            torch.save(policy_net.state_dict(), f'best_{params.target_update}.dat')

    #plt.savefig(f'update_mode_{params.target_update}_gamma_{params.gamma}_target_update_period_{params.target_update_period}.png', dpi=500)
    #plt.show()
    return task_score, all_scores, all_losses, all_errors

def plot_mean_results(all_scores_vec,all_losses_vec,all_errors_vec,MAX_EPISODES,name):
    fig, axes = plt.subplots(3, 1)
    y = np.mean(all_scores_vec,axis=0)
    error = np.std(all_scores_vec, axis=0)
    axes[0].plot(range(MAX_EPISODES), np.mean(all_scores_vec,axis=0), color='blue')
    axes[0].fill_between(range(MAX_EPISODES), y - error, y + error, alpha=0.3, edgecolor='#1B2ACC', facecolor='#089FFF')
    axes[0].set_title('Score over episodes', size=10)

    y = np.mean(all_losses_vec,axis=0)
    error = np.std(all_losses_vec, axis=0)
    axes[1].plot(range(MAX_EPISODES), y, color='blue')
    axes[1].fill_between(range(MAX_EPISODES), y - error, y + error, alpha=0.3, edgecolor='#1B2ACC', facecolor='#089FFF')
    axes[1].set_title('Loss over episodes', size=10)

    y = np.mean(all_errors_vec,axis=0)
    error = np.std(all_errors_vec, axis=0)
    axes[2].plot(range(MAX_EPISODES), y, color='blue')
    axes[2].fill_between(range(MAX_EPISODES), y - error, y + error, alpha=0.3, edgecolor='#1B2ACC', facecolor='#089FFF')
    axes[2].set_title('Mean Q error over episodes', size=10)
    fig.tight_layout()
    plt.savefig(name,dpi=500)
    


# mode_options = ['soft']
# MONTE_CARLO_NUMBER = 1
# MAX_EPISODES = 200
# buffer_mode = [True]
# final_results = {}
# for mode in mode_options:
#     final_results[mode] = {}
#     params.target_update = mode
#     for curr_buffer_mode in buffer_mode:

#         if curr_buffer_mode is True:
#             params.batch_size = 256
#         else:
#             params.batch_size = 1

#         task_total = 0
#         all_scores_vec = np.zeros((MONTE_CARLO_NUMBER,MAX_EPISODES))
#         all_losses_vec = np.zeros((MONTE_CARLO_NUMBER,MAX_EPISODES))
#         all_errors_vec = np.zeros((MONTE_CARLO_NUMBER,MAX_EPISODES))
#         task_score_vec = np.zeros((MONTE_CARLO_NUMBER))
#         for iteration in range(MONTE_CARLO_NUMBER):
#             task_score, all_scores, all_losses, all_errors = run_exp(max_episodes=MAX_EPISODES, buffer_mode=curr_buffer_mode)
#             task_score_vec[iteration] = task_score
#             all_scores_vec[iteration] = np.array(all_scores)
#             all_losses_vec[iteration] = np.array(all_losses)
#             all_errors_vec[iteration] = np.array(all_errors)

#         exp_name = f'buffer_mode_{curr_buffer_mode}_mode_{mode}'
#         print(50 * '=')
#         print(f'Finished experiment {exp_name}')
#         print(50 * '=')
#         mean_result = np.mean(task_score_vec)
#         std_result = np.std(task_score_vec)
#         min_result = np.min(task_score_vec)
#         max_result = np.max(task_score_vec)
#         results_dict = {'Mean Score':mean_result,'name':exp_name, 'std':std_result, 'min':min_result,'max':max_result}

#         if curr_buffer_mode:
#             final_results[mode]['with_buffer'] = results_dict
#         else:
#             final_results[mode]['without_buffer'] = results_dict

#         plot_mean_results(all_scores_vec, all_losses_vec, all_errors_vec, MAX_EPISODES, name=exp_name)

# for key in final_results:
#     for nest_key in final_results[key]:
#         print('------------------------------------------------------------------------------')
#         print('Experiment')
#         print(final_results[key][nest_key]['name'])
#         for result_mode,result in final_results[key][nest_key].items():
#             print(f'{result_mode} - {result}')
#         print('------------------------------------------------------------------------------')
cartpole_play()
plt.ioff()
