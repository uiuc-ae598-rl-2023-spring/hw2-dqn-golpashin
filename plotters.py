import matplotlib.pyplot as plt
import numpy as np
import torch

def plotters(env,episodes,Gs,Q,policy):
    # Plot the learning curve
    plt.figure(plt.gcf().number+1)
    plt.plot(episodes,Gs)
    plt.xlabel('episodes')
    plt.ylabel('G')
    plt.savefig('figures/learning_curve.png')   

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
    plt.title('Optimal Polciy')
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\dot{\theta}$")
    plt.colorbar()
    plt.show()
    plt.savefig('figures/policy.png')

    # Plot the state value function using pcolor
    plt.figure(plt.gcf().number+1)
    plt.pcolor(x_axis, y_axis, V_array)
    plt.title('State Value Function')
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\dot{\theta}$")
    plt.colorbar()
    plt.show()
    plt.savefig('figures/valuefunction.png')