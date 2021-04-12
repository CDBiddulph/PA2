import gym
import numpy as np
import utils
import matplotlib.pyplot as plt


def sample(theta, env, N):
    """ samples N trajectories using the current policy

    :param theta: the model parameters (shape d x 1)
    :param env: the environment used to sample from
    :param N: number of trajectories to sample
    :return:
        trajectories_gradients: lists with sublists for the gradients for each trajectory rollout
        trajectories_rewards:  lists with sublists for the rewards for each trajectory rollout

    Note: the maximum trajectory length is 200 steps
    """
    total_rewards = []
    total_grads = []

    for _ in range(N):
        observation = env.reset()
        traj_rewards = []
        traj_grads = []
        for _ in range(200):
            action_distribution = utils.compute_action_distribution(theta, observation)
            action = np.random.choice(action_distribution)
            traj_grads.append(utils.compute_log_softmax_grad(theta, observation, action))
            observation, cost, done, _ = env.step(action)
            traj_rewards.append(-cost)
            if done:
                break
        total_rewards.append(traj_rewards)
        total_grads.append(traj_grads)

    return total_grads, total_rewards


def train(N, T, delta):
    """

    :param N: number of trajectories to sample in each time step
    :param T: number of iterations to train the model
    :param delta: trust region size
    :return:
        theta: the trained model parameters
        avg_episodes_rewards: list of average rewards for each time step
    """
    theta = np.random.rand(100, 1)
    env = gym.make('CartPole-v0')
    env.seed(12345)

    episode_rewards = []

    for _ in range(T):
        grads, rewards = sample(theta, env, N)
        value_grad = utils.compute_value_gradient(grads, rewards)
        fisher = utils.compute_fisher_matrix(grads)
        eta = utils.compute_eta(delta, fisher, value_grad)
        episode_rewards.append(np.mean(rewards))
        theta = theta + eta * np.linalg.inv(fisher) @ value_grad

    return theta, episode_rewards


if __name__ == '__main__':
    np.random.seed(1234)
    theta, episode_rewards = train(N=100, T=20, delta=1e-2)
    plt.plot(episode_rewards)
    plt.title("avg rewards per timestep")
    plt.xlabel("timestep")
    plt.ylabel("avg rewards")
    plt.show()
