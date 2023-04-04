import discreteaction_pendulum
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
from plotters import plotters

# DQN for balancing a pendulum:
# This code use the DQN algorithm to balance a pendulum defined in the 
# 'discreteaction_pendulum' environment. The optimal policy is computed
# through training a set of agents using a three layered neural network.
# Below, under 'Experiment options', you may choose which components of
# of the DQN algorithm you would like to use. There are also two seperate
# .py files that generate figures and perform an ablation study. 

# Experiment options
Target_Q = True # Turns on/off Target_Q
Replay = True
ablation = True # Turns on/off Experience 
case1 = True  # Turns on/off Case 1 - With replay, with target Q 
case2 = True # Turns on/off Case 2 - With replay, without target Q 
case3 = True # Turns on/off Case 3 - Without replay, with target Q
case4 = True # Turns on/off Case 4 - Without replay, without target Q

# Hyperparameters
episodes_max = 200
Target_Update_Frequency = 10
memory_length = 10000000
batch_length = 32
gamma = 0.95
alpha = 0.00025
epsilon_update = True # decrease epsilon over steps
epsilon = 1
epsilon_delta = 0.0001
epsilon_final = 0.1

# Experiment parameters
realizations = 20 # number of realizations for the ablation study

# Environment
env = discreteaction_pendulum.Pendulum()
num_states = env.num_states
num_actions = env.num_actions

# Initialize memory
if Replay == False: # Truns off Experience Replay
    memory_length = batch_length
memory = deque(maxlen=memory_length) # element removed from the end when new element is added



#####################################################################################
# Neural Network:
# This is a neural network with three layers. It contains
# two hidden layers that each have 64 units. Specifically,
# it uses a tanh activation function at both the hidden
# layers, with a linear activation function at the output layer.
class DQN_NN(nn.Module):
    def __init__(self, input, num_actions, hidden_dim=64):
        super(DQN_NN, self).__init__()
        self.layer1 = nn.Linear(input, hidden_dim) # num_obsx64
        self.layer2 = nn.Linear(hidden_dim, hidden_dim) # 64x64
        self.layer3 = nn.Linear(hidden_dim, num_actions) # 64xnum_actions

    def forward(self, s): # forward pass with three layers
        y1 = F.tanh(self.layer1(s))
        y2 = F.tanh(self.layer2(y1))
        y3 = self.layer3(y2)
        return y3
    
Q = DQN_NN(num_states, num_actions) # Q-network
target_Q = DQN_NN(num_states, num_actions) # Target Q-network

# net = DQN_NN(4, 31, 64)
# print(net)
# print(net.parameters())
# print(net.state_dict())
#####################################################################################



#####################################################################################
# Greedy action:
# This function computes the greedy action
# using the Q network output, i.e. by taking
# the argmax of the neural netwrok's otuput
def greedy_a(num_actions,s,Q,epsilon):
    if np.random.rand() < epsilon:
        a = random.randrange(num_actions)
    else:
        a = torch.argmax(Q(torch.from_numpy(s).float()))
    return a

# Stochastic Gradient Descent (SGD):
# This function performs a stochastic descent step,
# either on a batch, or a single tuple to update
# the Q-network weights.  
def SGD(s, a, r, s1, done,Q,Replay):
    optimizer = optim.RMSprop(Q.parameters(), lr=alpha)  # Initialize optimizer to update network weights
    if Replay == True:
        loss_function = 0 
        for i in range(batch_length):
            Q_values = Q(torch.FloatTensor([s[i]])).gather(1, torch.LongTensor([[a[i]]])) # batch_size x 1
            if done[i]:
                E_y = r[i]
            else:
                E_y = r[i] + gamma * torch.max(target_Q(torch.FloatTensor([s1[i]]))).item()
            loss_function += F.mse_loss(Q_values, torch.FloatTensor([E_y]))
    else:
        loss_function = 0 
        Q_values = Q(torch.FloatTensor([s])).gather(1, torch.LongTensor([[a]]))
        if done:
            E_y = r
        else:
            E_y = r + gamma * torch.max(target_Q(torch.FloatTensor([s1]))).item()
        loss_function += F.mse_loss(Q_values, torch.FloatTensor([E_y]))
    
    optimizer.zero_grad() # clear gradients
    loss_function.backward() # compute gradient of loss_function
    optimizer.step() # update the NN params
    return Q_values
#####################################################################################



#####################################################################################
# Experience Replay:
# This function checks if the memory is filled up
# Then it, takes a random sample batch from the memory,
# and it uses that batch to perfrom a stochastic gradient
# descent step. This is only done if Replay is 'True'
def exp_replay(memory,batch_length,Q,target_Q,Replay):
    if len(memory) <= batch_length:
        return
    batch = random.sample(memory,batch_length)
    s, a, r, s1, done = zip(*batch) # tuples of (s,a,r,s1,done)

    SGD(s, a, r, s1, done,Q,Replay) # Perform a Stochastic Gradient Descent to update NN weights
#####################################################################################



#####################################################################################
# Training Agent:
# This function trains a number of 'episodes_max' agents
# to generate an optimal policy, using the DQN algortihm
# The epsilon is decayed at every step here to imporve
# convergence. The neural netwrok weights are reset after
# Target_Update_Frequency condition is set, given Q-netwrok
# option is set to True. Otherwise, we update the network
# weights at every step. Besides policy, we also output
# an array of the total rewards of each trained agent. 
def train(episodes_max,batch_length,gamma,num_actions,Q,
          target_Q,memory,Replay,Target_Update_Frequency,
          epsilon,epsilon_final,epsilon_delta,epsilon_update):
    Gs = []
    episodes = []
    for episode in range(episodes_max):
        s = env.reset()
        done = False
        G = 0
        power = 0
        while not done:
            a = greedy_a(num_actions,s,Q,epsilon)
            s1, r, done = env.step(a)
            exp_replay(memory,batch_length,Q,target_Q,Replay)
            if Replay ==  True:
                memory.append((s, a, r, s1, done))
            else: # If Replay is False, perfrom a stochastic gradient descent for a tuple instead of a batch.
                SGD(s, a, r, s1, done,Q,Replay) # perform a Stochastic Gradient Descent to update NN weights    
            s = s1
            G += (gamma**power)*r
            power += 1
            if done:
                break

            if epsilon_update == True:
                if epsilon > epsilon_final:
                    epsilon -= epsilon_delta
            if Target_Q == False: # without Target Q
                target_Q.load_state_dict(Q.state_dict()) # update the weights of the Target Q-network
                    
        if Target_Q == True: # with Target Q
            if episode % Target_Update_Frequency == 0:
                target_Q.load_state_dict(Q.state_dict()) # update the weights of the Target Q-network
        
        Gs.append(G)
        episodes.append(episode)
        
        print(f"Episode {episode+1}")
        print(f"G = {G}")
        if episode % Target_Update_Frequency == 0:  # epsilon is decreased at a higher frequency 
            print(f"epsilon = {epsilon}") # but I am printing it here whenever target Q is updated

    # Define a policy that maps every state to the "zero torque" action
    policy = lambda s: torch.argmax(Q(torch.from_numpy(s).float())).item()
    return policy, Gs, episodes, epsilon
#####################################################################################



# Run DQN and plot results:
if ablation == False: # just plot the trained agent trajectories
    policy, Gs, episodes, epsilon = train(episodes_max,batch_length,gamma,num_actions,Q,
            target_Q,memory,Replay,Target_Update_Frequency,
            epsilon,epsilon_final,epsilon_delta,epsilon_update) # perform training
else: # carry out the ablation study
    Gs_1 = np.zeros((realizations,episodes_max)) # Initialize the rewards
    Gs_2 = np.zeros((realizations,episodes_max))
    Gs_3 = np.zeros((realizations,episodes_max))
    Gs_4 = np.zeros((realizations,episodes_max))
    for i in range(realizations):
        if case1 == True:
            # Case 1 - With replay, with target Q (i.e., the standard algorithm):
            print('Case 1 - With replay, with target Q:')
            Target_Q = True
            Replay = True
            if epsilon_update == True:
                epsilon = 1 # reset the epsilon
            memory = [] # reset the memory 
            Q = DQN_NN(num_states, num_actions) # reset the Q-network
            target_Q = DQN_NN(num_states, num_actions) # reset the Target Q-network
            policy1, Gs1, episodes1, epsilon1 = train(episodes_max,batch_length,gamma,num_actions,Q,
                target_Q,memory,Replay,Target_Update_Frequency,
                epsilon,epsilon_final,epsilon_delta,epsilon_update) # perform training
            Gs_1[i,:] = Gs1 # save the current rewards array as a column
        
        if case2 == True:
            # Case 2 - With replay, without target Q (i.e., the target network is reset after each step):
            print('Case 2 - With replay, without target Q:')
            Target_Q = False
            Replay = True
            if epsilon_update == True:
                epsilon = 1 # reset the epsilon
            memory = [] # reset the memory
            Q = DQN_NN(num_states, num_actions) # reset the Q-network
            target_Q = DQN_NN(num_states, num_actions) # reset the Target Q-network 
            policy2, Gs2, episodes2, epsilon2 = train(episodes_max,batch_length,gamma,num_actions,Q,
                target_Q,memory,Replay,Target_Update_Frequency,
                epsilon,epsilon_final,epsilon_delta,epsilon_update) # perform training
            Gs_2[i,:] = Gs2 # save the current rewards array as a column
        
        if case3 == True:
            # Case 3 - Without replay, with target Q (i.e., the size of the replay memory buffer is equal to the size of each minibatch):
            print('Case 3 - Without replay, with target Q:')
            Target_Q = True
            Replay = False
            if epsilon_update == True:
                epsilon = 1 # reset the epsilon
            memory = [] # reset the memory
            Q = DQN_NN(num_states, num_actions) # reset the Q-network
            target_Q = DQN_NN(num_states, num_actions) # reset the Target Q-network 
            memory_length = batch_length
            policy3, Gs3, episodes3, epsilon3 = train(episodes_max,batch_length,gamma,num_actions,Q,
                target_Q,memory,Replay,Target_Update_Frequency,
                epsilon,epsilon_final,epsilon_delta,epsilon_update) # perform training
            Gs_3[i,:] = Gs3 # save the current rewards array as a column

        if case4 == True:
            # Case 4 - Without replay, without target Q (i.e., the target network is reset after each step and the size of the replay memory buffer is equal to the size of each minibatch):
            print('Case 4 - Without replay, without target Q:')
            Target_Q = False
            Replay = False
            if epsilon_update == True:
                epsilon = 1 # reset the epsilon
            memory = [] # reset the memory
            Q = DQN_NN(num_states, num_actions) # reset the Q-network
            target_Q = DQN_NN(num_states, num_actions) # reset the Target Q-network 
            memory_length = batch_length
            policy4, Gs4, episodes4, epsilon4 = train(episodes_max,batch_length,gamma,num_actions,Q,
                target_Q,memory,Replay,Target_Update_Frequency,
                epsilon,epsilon_final,epsilon_delta,epsilon_update) # perform training
            Gs_4[i,:] = Gs4 # save the current rewards array as a column



if ablation == False:
    # See plotters.py for this function:
    plotters(env,episodes,Gs,Q,policy) # plots the results of the training
else:
        plt.figure(plt.gcf().number+1)
        if case1 == True:
            plt.plot(episodes1,np.mean(Gs_1,axis=0), color='r', label='Mean Return')
            plt.fill_between(episodes1,(np.mean(Gs_1,axis=0)-np.std(Gs1,axis=0)),(np.mean(Gs_1,axis=0)+np.std(Gs_1,axis=0)),color='r',alpha=.2)
        if case2 == True:    
            plt.plot(episodes2,np.mean(Gs_2,axis=0), color='g', label='Mean Return (With replay, without target Q)')
            plt.fill_between(episodes2,(np.mean(Gs_2,axis=0)-np.std(Gs2,axis=0)),(np.mean(Gs_2,axis=0)+np.std(Gs_2,axis=0)),color='g',alpha=.2)
        if case3 == True: 
            plt.plot(episodes3,np.mean(Gs_3,axis=0), color='b', label='Mean Return (Without replay, with target Q)')
            plt.fill_between(episodes3,(np.mean(Gs_3,axis=0)-np.std(Gs3,axis=0)),(np.mean(Gs_3,axis=0)+np.std(Gs_3,axis=0)),color='b',alpha=.2)
        if case4 == True: 
            plt.plot(episodes4,np.mean(Gs_4,axis=0), color='c', label='Mean Return (Without replay, without target Q)')
            plt.fill_between(episodes4,(np.mean(Gs_4,axis=0)-np.std(Gs4,axis=0)),(np.mean(Gs_4,axis=0)+np.std(Gs_4,axis=0)),color='c',alpha=.2)
        plt.legend()
        # plt.ylim(auto=True)
        plt.ylim(-5,20)
        plt.xlabel('Epsiodes')
        plt.ylabel('Reward Sum')
        plt.title('Ablation Study')
        plt.savefig('figures/ablation_study.png')

        if case1 == True:
            plt.figure(plt.gcf().number+1)
            plt.plot(episodes1,np.mean(Gs_1,axis=0), color='r', label='Mean Return')
            plt.fill_between(episodes1,(np.mean(Gs_1,axis=0)-np.std(Gs1,axis=0)),(np.mean(Gs_1,axis=0)+np.std(Gs_1,axis=0)),color='r',alpha=.2,label=r'Standard Deviation Bounds')
            plt.legend()
            plt.ylim(auto=True)
            plt.xlabel('Epsiodes')
            plt.ylabel('Reward Sum')
            plt.title('Mean Discounted Reward Sum (with replay, with target Q)')
            plt.savefig('figures/case1.png')

        if case2 == True:
            plt.figure(plt.gcf().number+1)
            plt.plot(episodes2,np.mean(Gs_2,axis=0), color='b', label='Mean Return')
            plt.fill_between(episodes2,(np.mean(Gs_2,axis=0)-np.std(Gs2,axis=0)),(np.mean(Gs_2,axis=0)+np.std(Gs_2,axis=0)),color='b',alpha=.2,label=r'Standard Deviation Bounds')
            plt.legend()
            plt.ylim(auto=True)
            plt.xlabel('Epsiodes')
            plt.ylabel('Reward Sum')
            plt.title('Mean Discounted Reward Sum (with replay, without target Q)')
            plt.savefig('figures/case2.png')
        
        if case3 == True:
            plt.figure(plt.gcf().number+1)
            plt.plot(episodes3,np.mean(Gs_3,axis=0), color='g', label='Mean Return')
            plt.fill_between(episodes3,(np.mean(Gs_3,axis=0)-np.std(Gs3,axis=0)),(np.mean(Gs_3,axis=0)+np.std(Gs_3,axis=0)),color='g',alpha=.2,label=r'Standard Deviation Bounds')
            plt.legend()
            plt.ylim(auto=True)
            plt.xlabel('Epsiodes')
            plt.ylabel('Reward Sum')
            plt.title('Mean Discounted Reward Sum (without replay, with target Q)')
            plt.savefig('figures/case3.png')
            
        if case3 == True:
            plt.figure(plt.gcf().number+1)
            plt.plot(episodes4,np.mean(Gs_4,axis=0), color='c', label='Mean Return')
            plt.fill_between(episodes4,(np.mean(Gs_4,axis=0)-np.std(Gs4,axis=0)),(np.mean(Gs_4,axis=0)+np.std(Gs_4,axis=0)),color='c',alpha=.2,label=r'Standard Deviation Bounds')
            plt.legend()
            plt.ylim(auto=True)
            plt.xlabel('Epsiodes')
            plt.ylabel('Reward Sum')
            plt.title('Mean Discounted Reward Sum (without replay, without target Q)')
            plt.savefig('figures/case4.png')