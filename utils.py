from sklearn.kernel_approximation import RBFSampler
import numpy as np

rbf_feature = RBFSampler(gamma=1, random_state=12345)


def extract_features(state, num_actions):
    """ This function computes the RFF features for a state for all the discrete actions

    :param state: column vector of the state we want to compute phi(s,a) of (shape |S|x1)
    :param num_actions: number of discrete actions you want to compute the RFF features for
    :return: phi(s,a) for all the actions (shape 100x|num_actions|)
    """
    s = state.reshape(1, -1)
    s = np.repeat(s, num_actions, 0)
    a = np.arange(0, num_actions).reshape(-1, 1)
    sa = np.concatenate([s, a], -1)
    feats = rbf_feature.fit_transform(sa)
    feats = feats.T
    return feats


def compute_softmax(logits, axis):
    """ computes the softmax of the logits

    :param logits: the vector to compute the softmax over
    :param axis: the axis we are summing over
    :return: the softmax of the vector

    Hint: to make the softmax more stable, subtract the max from the vector before applying softmax
    """
    # TODO
    logits -= max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=axis)


def compute_action_distribution(theta, phis):
    """ compute probability distribution over actions

    :param theta: model parameter (shape d x 1)
    :param phis: RFF features of the state and actions (shape d x |A|)
    :return: softmax probability distribution over actions (shape 1 x |A|)
    """

    return compute_softmax(theta.T @ phis, 1)


def compute_log_softmax_grad(theta, phis, action_idx):
    """ computes the log softmax gradient for the action with index action_idx

    :param theta: model parameter (shape d x 1)
    :param phis: RFF features of the state and actions (shape d x |A|)
    :param action_idx: The index of the action you want to compute the gradient of theta with respect to
    :return: log softmax gradient (shape d x 1)
    """

    action_distribution = compute_action_distribution(theta, phis)
    return phis[:, action_idx] - sum(phis @ action_distribution, axis=1)


def compute_fisher_matrix(grads):
    """ computes the fisher information matrix using the sampled trajectories gradients

    :param grads: list of list of gradients, where each sublist represents a trajectory (each gradient has shape d x 1)
    :return: fisher information matrix (shape d x d)

    Note: don't forget to take into account that trajectories might have different lengths
    """
    if not grads or not grads[0]:
        return None

    d = grads[0][0].shape[0]

    result = np.zeros((d, d))

    N = len(grads)
    for traj in grads:
        H = len(traj)
        for grad in traj:
            result += grad @ grad.T / (N * H)

    return result + 1e-6*np.eye(d)


def compute_value_gradient(grads, rewards):
    """ computes the value function gradient with respect to the sampled gradients and rewards

    :param grads: list of list of gradients, where each sublist represents a trajectory
    :param rewards: list of list of rewards, where each sublist represents a trajectory
    :return: value function gradient with respect to theta (shape d x 1)
    """

    result = np.zeros_like(grads[0][0])

    N = len(grads)
    b = np.mean(np.sum(r_traj) for r_traj in rewards)
    for grad_traj, reward_traj in zip(grads, rewards):
        H = len(grad_traj)
        reward_sum = np.sum(reward_traj) - b
        for h, grad in enumerate(grad_traj):
            result += grad * reward_sum / (N * H)
            reward_sum -= reward_traj[h]

    return result


def compute_eta(delta, fisher, v_grad):
    """ computes the learning rate for gradient descent

    :param delta: trust region size
    :param fisher: fisher information matrix (shape d x d)
    :param v_grad: value function gradient with respect to theta (shape d x 1)
    :return: the maximum learning rate that respects the trust region size delta
    """

    denom = v_grad.T @ fisher @ v_grad + 1e-6
    return np.sqrt(delta / denom)
