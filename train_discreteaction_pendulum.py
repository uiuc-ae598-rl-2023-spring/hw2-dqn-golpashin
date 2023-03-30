import discreteaction_pendulum
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

# Experiment options
Target_Q = True # Truns on/off Target_Q
Replay = True # Truns on/off Experience Replay
epsilon_update = True # decrease epsilon over steps

# Hyperparameters
episodes_max = 200
Target_Update_Frequency = 10
memory_length = 10000000
batch_length = 32
gamma = 0.95
alpha = 0.00025
epsilon = 1
epsilon_delta = 0.0001
epsilon_final = 0.1

if Replay == False: # Truns off Experience Replay
    memory_length = batch_length

# Environment
env = discreteaction_pendulum.Pendulum()
num_states = env.num_states
num_actions = env.num_actions

# Define the Neural Network
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

# net = DQN_NN(4, 31, 64)
# print(net)
# print(net.parameters())
# print(net.state_dict())

# DQN algorithm
# Initialize objects
memory = deque(maxlen=memory_length) # element removed from the end when new element is added
Q = DQN_NN(num_states, num_actions) # Q-network
target_Q = DQN_NN(num_states, num_actions) # Target Q-network
optimizer = optim.RMSprop(Q.parameters(), lr=alpha) 

def greedy_a(num_actions,s,Q,epsilon):
    if np.random.rand() < epsilon:
        a = random.randrange(num_actions)
    else:
        a = torch.argmax(Q(torch.from_numpy(s).float()))
    return a

def exp_replay(memory,batch_length,Q,target_Q):
    if len(memory) <= batch_length:
        return
    batch = random.sample(memory,batch_length)
    s, a, r, s1, done = zip(*batch) # tuples of (s,a,r,s1,done)
    
    # Stochastic Gradient Descent
    loss_function = 0 
    for i in range(batch_length):
        Q_values = Q(torch.FloatTensor([s[i]])).gather(1, torch.LongTensor([[a[i]]])) # batch_size x 1
        if done[i]:
            E_y = r[i]
        else:
            E_y = r[i] + gamma * torch.max(target_Q(torch.FloatTensor([s1[i]]))).item()
        loss_function += F.mse_loss(Q_values, torch.FloatTensor([E_y]))
    
    optimizer.zero_grad() # clear gradients
    loss_function.backward() # compute gradient of loss_function
    optimizer.step() # update the NN params
    return Q_values

# Train the agents
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
        exp_replay(memory,batch_length,Q,target_Q)
        memory.append((s, a, r, s1, done))
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


# Plot the learning curve
plt.figure(plt.gcf().number+1)
plt.plot(episodes,Gs)
plt.xlabel('episodes')
plt.ylabel('G')
plt.savefig('figures/learning_curve.png')   

# Define a policy that maps every state to the "zero torque" action
policy = lambda s: torch.argmax(Q(torch.from_numpy(s).float())).item()

# Simulate an episode and save the result as an animated gif
env.video(policy, filename='figures/discreteaction_pendulum.gif')

# Simulate a single agent now
# Initialize simulation 
s = env.reset()

# Create dict to store data from simulation
data = {
    't': [0],
    's': [s],
    'a': [],
    'r': [],
}

# Simulate until episode is done
done = False
while not done:
    a = policy(s)
    print(a)
    (s, r, done) = env.step(a)
    data['t'].append(data['t'][-1] + 1)
    data['s'].append(s)
    data['a'].append(a)
    data['r'].append(r)

# Parse data from simulation
data['s'] = np.array(data['s'])
theta = data['s'][:, 0]
thetadot = data['s'][:, 1]
tau = [env._a_to_u(a) for a in data['a']]

# Plot data and save to png file
fig, ax = plt.subplots(3, 1, figsize=(10, 10))
ax[0].plot(data['t'], theta, label='theta')
ax[0].plot(data['t'], thetadot, label='thetadot')
ax[0].legend()
ax[1].plot(data['t'][:-1], tau, label='tau')
ax[1].legend()
ax[2].plot(data['t'][:-1], data['r'], label='r')
ax[2].legend()
ax[2].set_xlabel('time step')
plt.tight_layout()
plt.savefig('figures/agent_trajectories.png')


# Create a meshgrid for the x and y axes
theta_x = np.linspace(-np.pi, np.pi,100)
theta_dot_y = np.linspace(-env.max_thetadot,env.max_thetadot,100)
x_axis, y_axis = np.meshgrid(theta_x, theta_dot_y)

policy_array = np.zeros_like(x_axis)
for i in range(len(theta_x)):
    for j in range(len(theta_x)):
        s = np.array((x_axis[i,j], y_axis[i,j]))
        policy_array[i,j] = policy(s)
  
V_array = np.zeros_like(x_axis)
for i in range(len(theta_x)):
    for j in range(len(theta_x)):
        s = np.array((x_axis[i,j], y_axis[i,j]))
        V_array[i,j] = torch.max(Q(torch.from_numpy(s).float())).item()


# Plot the policy using pcolor
plt.figure(plt.gcf().number+1)
plt.pcolor(x_axis, y_axis, policy_array)
plt.xlabel('Theta')
plt.ylabel('Theta_dot')
plt.colorbar()
plt.show()
plt.savefig('figures/policy.png')

# Plot the state value function using pcolor
plt.figure(plt.gcf().number+1)
plt.pcolor(x_axis, y_axis, V_array)
plt.xlabel('Theta')
plt.ylabel('Theta_dot')
plt.colorbar()
plt.show()
plt.savefig('figures/valuefunction.png')