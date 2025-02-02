import numpy as np
import pandas as pd
from typing import List


def update(q: float, r: float, k: int) -> float:
    """
    Update the Q-value using the given reward and number of times the action has been taken.

    Parameters:
    q (float): The current Q-value.
    r (float): The reward received for the action.
    k (int): The number of times the action has been taken before.

    Returns:
    float: The updated Q-value.
    """
    alpha = 1 / (k + 1)  # Learning rate decreases with the number of visits
    q_new = q + alpha * (r - q)
    return q_new


def greedy(q_estimate: np.ndarray) -> int:
    """
    Selects the action with the highest Q-value.

    Parameters:
    q_estimate (numpy.ndarray): 1-D Array of Q-values for each action.

    Returns:
    int: The index of the action with the highest Q-value.
    """
    return np.argmax(q_estimate)


def egreedy(q_estimate: np.ndarray, epsilon: float) -> int:
    """
    Implements the epsilon-greedy exploration strategy for multi-armed bandits.

    Parameters:
    q_estimate (numpy.ndarray): 1-D Array of estimated action values.
    epsilon (float): Exploration rate, determines the probability of selecting a random action.
    n_arms (int): Number of arms in the bandit. default is 10.

    Returns:
    int: The index of the selected action.
    """
    if np.random.rand() < epsilon:
        # Explore: choose a random action
        return np.random.randint(0, len(q_estimate))  # Or q_estimate.size
    else:
        # Exploit: choose the action with the highest Q-value
        return np.argmax(q_estimate)


def empirical_egreedy(epsilon: float, n_trials: int, n_arms: int, n_plays: int) -> List[List[float]]:
    """
    Run the epsilon-greedy algorithm on a multi-armed bandit problem. For each play,
    the algorithm selects an action based on the epsilon-greedy strategy and updates
    the Q-value of the selected action. For each trial, the algorithm returns
    the rewards for each play, a total of n_plays.

    Args:
        epsilon (float): epsilon value for the epsilon-greedy algorithm.
        n_trials (int): number of trials to run the algorithm
        n_arms (int): number of arms in the bandit
        n_plays (int): number of plays in each trial

    Returns:
        List[List[float]]: A list of rewards for each play in each trial.
    """
    rewards = []  # stores the rewards for each trial

    for _ in range(n_trials):
        trial_rewards = []
        q_estimate = np.zeros(n_arms)  # Initialize Q-values to 0 for each arm
        action_counts = np.zeros(n_arms)  # Keep track of how many times each action was taken.

        for _ in range(n_plays):
            action = egreedy(q_estimate, epsilon) # Use the egreedy function

            # Simulate getting a reward (replace with your actual reward function/environment)
            reward = np.random.normal(0, 1)  # Example: reward from a normal distribution
            # In a real setting, this reward would come from interacting with the environment.

            trial_rewards.append(reward)

            # Update Q-value for the chosen action
            action_counts[action] += 1
            q_estimate[action] = update(q_estimate[action], reward, action_counts[action] -1) # Use the update function

        rewards.append(trial_rewards)

    return rewards
