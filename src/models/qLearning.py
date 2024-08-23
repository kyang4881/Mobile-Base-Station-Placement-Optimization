import copy
import numpy as np
import random
import matplotlib.pyplot as plt
import math
from utils.functions import plot_uav_positions

class QLearningBestResponse:
    """
    Initialize the QLearningBestResponse class.

    Args:
        num_uav (int): Number of UAVs.
        m (int): Grid size (m x m).
        learning_rate (float): Learning rate for Q-learning.
        discount_factor (float): Discount factor for future rewards.
        exploration_prob (float): Probability of taking a random action.
        verbose (bool): Whether to print verbose information.
    """
    def __init__(self, num_uav, m, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.1, verbose=False):
        self.num_uav = num_uav
        self.m = m
        self.user_positions = [(x, y) for x in range(1, self.m + 1) for y in range(1, self.m + 1)]
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob

        if self.num_uav > len(self.user_positions):
            raise ValueError("The number of UAVs exceeds the maximum possible unique positions.")

        self.uav_positions = random.sample(self.user_positions, self.num_uav)
        self.q_table = {k: {"N": 0, "E": 0, "S": 0, "W": 0} for k in self.user_positions} #{}  # Q-table to store Q-values
        self.verbose = verbose

    def get_valid_new_state(self, state, max_attempts=10):
        """
        Get a valid new state for a given state.

        Args:
            state (tuple): Current state.
            max_attempts (int): Maximum attempts to find a valid new state.

        Returns:
            tuple: Valid new state.
        """
        # Find the action with the highest Q-value
        max_action = max(self.q_table[state], key=self.q_table[state].get)
        new_state = self.move(state, max_action)
        attempts = 0

        while new_state in self.uav_positions and attempts < max_attempts:
            # If there's an overlap, keep trying different new states based on Q-values until a valid one is found
            actions = sorted(self.q_table[state], key=lambda k: self.q_table[state][k], reverse=True)
            actions.remove(max_action)  # Remove the current max action

            for action in actions:
                new_state = self.move(state, action)
                if new_state not in self.uav_positions:
                    return new_state
            attempts += 1

        return new_state

    def move(self, position, direction):
        """
        Move from a current position to a new position based on a given direction.

        Args:
            position (tuple): Current position.
            direction (str): Direction to move ('N', 'E', 'S', 'W').

        Returns:
            tuple: New position after the move.
        """
        x, y = position
        if direction == 'N':
            new_position = (x, y + 1)
        elif direction == 'E':
            new_position = (x + 1, y)
        elif direction == 'S':
            new_position = (x, y - 1)
        elif direction == 'W':
            new_position = (x - 1, y)
        else:
            raise ValueError("Invalid direction")

        # Check if the new position is within the grid boundaries
        new_x, new_y = new_position
        if 1 <= new_x <= self.m and 1 <= new_y <= self.m:
            return new_position
        else:
            return position  # Return the current position if it's outside the boundaries

    def euclidean_distance(self, point1, point2):
        """
        Calculate the Euclidean distance between two points.

        Args:
            point1 (tuple): First point (x, y).
            point2 (tuple): Second point (x, y).

        Returns:
            float: Euclidean distance between the two points.
        """
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def update_uav_positions(self, tuples_list, target_position, new_tuple):
        """
        Update the UAV positions list by replacing a target position with a new position.

        Args:
            tuples_list (list of tuples): List of tuples representing UAV positions.
            target_position (tuple): Position to be replaced.
            new_tuple (tuple): New position to replace the target position.
        """
        for i, t in enumerate(tuples_list):
            if t == target_position:
                tuples_list[i] = new_tuple
        return tuples_list

    def connect_users_to_uavs_original(self, curr_uav_position, uav_positions_):
        """
        Calculate the number of users connected to a specific UAV. The utility of the UAVs.

        Args:
            curr_uav_position (tuple): Current UAV position.
            uav_positions_ (list of tuples): List of UAV positions.

        Returns:
            int: Number of users connected to the current UAV.
        """
        users_connected_to_uav = {p: 0 for p in uav_positions_}

        for user_position in self.user_positions:
            # Find the index of the closest UAV based on minimum Euclidean distance
            #print("user_position", user_position)
            dist = []
            for uav_position in uav_positions_:
                dist.append(self.euclidean_distance(user_position, uav_position))
                #print(f"user_position:{user_position}, uav_position: {uav_position}, euclidean_distance:{euclidean_distance(user_position, uav_position)}, dist: {dist}")
            if len(set(dist)) == 1: # All distances are the same
                #print(f"len(set(dist)) == 1: {len(set(dist)) == 1:}")
                for uav_position in uav_positions_:
                    users_connected_to_uav[uav_position] += 1/len(uav_positions_)
                #print(f"users_connected_to_uav: {users_connected_to_uav}")
            else:
                #print("else:")
                uav_idex = np.argmin(dist)
                users_connected_to_uav[uav_positions_[uav_idex]] += 1
                #print(f"users_connected_to_uav: {users_connected_to_uav}")

        return users_connected_to_uav[curr_uav_position]


    def compute_user_utility_by_uav(self, uav_positions_):
        """
        Calculate the utility based on the total distances of all users relative to the UAVs.

        Args:
            uav_positions_ (list of tuples): List of UAV positions.

        Returns:
            dict: Dictionary with UAV positions as keys and their respective total distance-based utility as values.
        """
        uav_total_utility = {}
        for user_position in self.user_positions:
            uav_distances = []
            uav_positions = []
            for uav_position in uav_positions_:
                uav_distances.append(self.euclidean_distance(user_position, uav_position))
                uav_positions.append(uav_position)

            min_indices = [index for index, value in enumerate(uav_distances) if value == min(uav_distances)]

            for ind in min_indices:
                if len(min_indices) > 1:
                    uav_total_utility[uav_positions[ind]] = round(uav_total_utility.get(uav_positions[ind], 0) - uav_distances[ind]/len(min_indices), 5)

                else:
                    uav_total_utility[uav_positions[ind]] = round(uav_total_utility.get(uav_positions[ind], 0) - uav_distances[ind], 5)

        return uav_total_utility

    def plot_user_and_uav_positions(self):
        """
        Plot the user and UAV positions on a grid.
        """
        # Extract x and y coordinates for users and UAVs
        user_x, user_y = zip(*self.user_positions)
        uav_x, uav_y = zip(*self.uav_positions)
        # Create a plot to display user and UAV positions
        plt.figure(figsize=(3, 3))
        plt.scatter(user_x, user_y, color='grey', label='Users', marker='o', s=30)
        plt.scatter(uav_x, uav_y, color='blue', label='UAVs', marker='^', s=80)
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('User and UAV Positions')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # Set the grid to show 10x10 with 1 spacing
        plt.xticks(range(1, self.m + 1))
        plt.yticks(range(1, self.m + 1))
        plt.grid(True, which='both', linestyle='--', lw=1)
        plt.show()

    def update_q_table(self, state, action, reward, next_state):
        """
        Update the Q-table with new Q-values.

        Args:
            state (tuple): Current state.
            action (str): Chosen action.
            reward (float): Received reward.
            next_state (tuple): Next state.
        """
        # Best value for the next state
        max_next_action_value = max(self.q_table[next_state].values())
        # Update q table
        self.q_table[state][action] = (1 - self.learning_rate) * self.q_table[state][action] + self.learning_rate * (reward + self.discount_factor * max_next_action_value)

    def select_action(self, state):
        """
        Select an action based on the Q-table.

        Args:
            state (tuple): Current state.

        Returns:
            str: Chosen action ('N', 'E', 'S', 'W').
        """
        # Choose a random action
        if random.random() < self.exploration_prob:
            if self.verbose: print("Random Action")
            return random.choice(["N", "E", "S", "W"])
        else:
            # Choose the available best action
            max_actions = [action for action in self.q_table[state] if self.q_table[state][action] == max(self.q_table[state].values())]
            action = random.choice(max_actions)
            if self.verbose: print(f"self.q_table[state]: {self.q_table[state]}")
            if self.verbose: print(f"Action: max(self.q_table[state], key=self.q_table[state].get): max({self.q_table[state]}, key=self.q_table[state].get): {max(self.q_table[state], key=self.q_table[state].get)}")
            return action

    def transition(self, uav_index, verbose=False):
        """
        Perform a state transition for a UAV.

        Args:
            uav_index (int): Index of the UAV.
            verbose (bool): Whether to print verbose information.

        Returns:
            tuple: New state, flag indicating if the transition is done, and the new utility value.
        """
        state = self.uav_positions[uav_index]
        occupied_positions = set(self.uav_positions)
        uav_curr_utility = self.connect_users_to_uavs_original(state, self.uav_positions)
        action = self.select_action(state)
        new_state = self.move(state, action)

        if verbose:
            print(f"UAV {uav_index} - state: {state}, uav_curr_utility: {uav_curr_utility}, action: {action}, new_state: {new_state}")
        # Check if the new state is an occupied position by another UAV
        if new_state not in occupied_positions:
            uav_positions_copy = copy.deepcopy(self.uav_positions) # Keep a copy of the UAV's positions
            uav_positions_copy = self.update_uav_positions(uav_positions_copy, state, new_state) # Update the UAV's copied positions
            uav_new_utility = self.connect_users_to_uavs_original(new_state, uav_positions_copy) # Number of users connected to the new position
            # Check if the new position has a better utility
            if uav_new_utility >= uav_curr_utility:
                self.update_q_table(state, action, uav_new_utility, new_state)  # Update the q table
                self.uav_positions = uav_positions_copy  # Update the UAV's actual positions
                return new_state, False, uav_new_utility

        return state, True, uav_curr_utility

    def train(self, num_episodes=1000):
        """
        Train the Q-learning agent over a specified number of episodes.

        Args:
            num_episodes (int): Number of episodes to train.
        """
        # Train for specified number of episodes
        for episode in range(num_episodes):
            self.uav_positions = random.sample(self.user_positions, self.num_uav)  # Randomly initialize a new starting state
            is_done = False
            # Loop through all UAVs until termination
            while not is_done:
                for uav_index in range(len(self.uav_positions)):
                    uav_new_position, uav_is_done, utility = self.transition(uav_index, verbose=False)  # Transition to the next step
                    is_done = uav_is_done

    def run(self, max_iter=1000, max_dev_not_improved=10, verbose=False, verbose_plot=True):
        """
        Run the Q-learning agent to optimize UAV positions.

        Args:
            max_iter (int): Maximum number of iterations.
            max_dev_not_improved (int): Maximum number of iterations with no improvement.
            verbose (bool): Whether to print verbose information.
        """
        if verbose_plot:
            print("Iteration: 0")
            self.plot_user_and_uav_positions()

        iteration = 1
        stdev_not_improved_cnt = 0
        stdev_list = [float("Inf")]
        stdev_best = float("Inf")
        all_uav_positions_list = [[(0,0), (0,0)]]
        all_utilities = [[0,0]]

        # Loop until the max iteration threshold is reached or the terminate proess kicks in
        while iteration < max_iter:
            utilities = []
            uav_positions_list = []
            for uav_index in range(len(self.uav_positions)): # Loop through all UAVs
                state = self.uav_positions[uav_index]
                #action = max(self.q_table[state], key=self.q_table[state].get)  # Choose an action based on the best Q value
                action = self.select_action(state)
                new_state = self.move(state, action)
                new_state = self.get_valid_new_state(new_state)  # Check if the new state is valid position that is not already occupied by another UAV
                self.uav_positions = self.update_uav_positions(self.uav_positions, state, new_state) # Update the UAV's position
                state = new_state
                uav_positions_list.append(self.uav_positions[uav_index])

            utilities.append([self.connect_users_to_uavs_original(position, self.uav_positions) for position in self.uav_positions])
            stdev_list.append(np.std(utilities))
            all_uav_positions_list.append(uav_positions_list)
            all_utilities.append(utilities)

            if verbose_plot: print(f"\nIteration: {iteration}, \nall_uav_positions_list: {all_uav_positions_list}, \nall_utilities: {all_utilities}, \nstdev_list: {stdev_list}, \nmin(stdev_list): {min(stdev_list)}")
            # Check if there's improvement in the standard deviation of the UAV's utility
            if stdev_list[-1] < stdev_best:
                stdev_best = stdev_list[-1]
                stdev_not_improved_cnt = 0
                if verbose: print("improved:", stdev_not_improved_cnt)
            else:
                stdev_not_improved_cnt += 1
                if verbose: print("NOT improved:", stdev_not_improved_cnt)

            if stdev_not_improved_cnt == max_dev_not_improved:
                if verbose: print(f"\nTerminating. No improvement from the previous standard deviation of the UAV utilities in the last {max_dev_not_improved} iterations.")
                break

            iteration += 1
            if verbose_plot: self.plot_user_and_uav_positions()

        # Display the positions of the UAVs with the first best utility standard deviation
        best_index = np.argmin(stdev_list)
        if verbose: print(f"Best_index: {best_index}")
        if verbose_plot: print(f"Best positions for minimum utility standard deviation: {all_uav_positions_list[best_index]}, Utilities: {all_utilities[best_index]}, StDev: {stdev_list[best_index]}")
        if verbose_plot: plot_uav_positions(self.user_positions, all_uav_positions_list[best_index], self.m)

        return all_uav_positions_list[best_index], all_utilities[best_index]