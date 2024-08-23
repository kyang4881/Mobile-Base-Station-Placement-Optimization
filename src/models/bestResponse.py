import copy
import numpy as np
import random
import matplotlib.pyplot as plt
import math

class BestResponseLearning:
    """
    Initialize the BestResponseLearning class.

    Args:
        num_uav (int): The number of UAVs.
        m (int): The size of the grid in both X and Y dimensions.

    """
    def __init__(self, num_uav, m):
        self.num_uav = num_uav
        self.m = m
        self.user_positions = [(x, y) for x in range(1, self.m + 1) for y in range(1, self.m + 1)]  # Generate user positions uniformly across the grid

        # Ensure unique UAV positions
        if self.num_uav > len(self.user_positions):
            raise ValueError("The number of UAVs exceeds the maximum possible unique positions.")

        unique_positions = random.sample(self.user_positions, self.num_uav)
        self.uav_positions = unique_positions

    def move(self, position, direction):
        """
        Move a position in a specified direction.

        Args:
            position (tuple): The current (x, y) position.
            direction (str): The direction to move ('N', 'E', 'S', 'W').

        Returns:
            tuple: The new (x, y) position after the move.

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
        return new_position

    # Utility function to calculate Euclidean distance
    def euclidean_distance(self, point1, point2):
        """
        Calculate the Euclidean distance between two points.

        Args:
            point1 (tuple): The (x, y) coordinates of the first point.
            point2 (tuple): The (x, y) coordinates of the second point.

        Returns:
            float: The Euclidean distance between the two points.

        """
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def connect_users_to_uavs_original(self, curr_uav_position, uav_positions_):
        """
        Connect users to UAVs based on proximity.

        Args:
            curr_uav_position (tuple): The current (x, y) position of the UAV.
            uav_positions_ (list of tuples): List of UAV positions.

        Returns:
            dict: A dictionary where keys are UAV positions and values are the number of connected users.

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
        Plot the positions of users and UAVs on a 2D grid.

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

    def update_uav_positions(self, tuples_list, target_position, new_tuple):
        """
        Update the UAV positions in a list.

        Args:
            tuples_list (list of tuples): List of UAV positions.
            target_position (tuple): The position to be updated.
            new_tuple (tuple): The new position to replace the target.
        """
        for i, t in enumerate(tuples_list):
            if t == target_position:
                tuples_list[i] = new_tuple
        return tuples_list

    def transition(self, verbose):
        """
        Perform a transition step for the UAVs based on their best response strategy.

        Args:
            verbose (bool): Whether to print verbose information.

        Returns:
            tuple: A tuple containing lists of new UAV positions, and a flag indicating if UAVs are done moving.

        """
        occupied_positions = set(self.uav_positions)
        is_done = []
        uav_moves = []
        utilities = []
        for i in range(len(self.uav_positions)): # Loop through all uavs
            directions = ["N", "E", "S", "W"]
            possible_utilities = []
            possible_directions = []
            curr_pos = self.uav_positions[i] # Current position
            uav_curr_utility = self.connect_users_to_uavs_original(curr_pos, self.uav_positions) # Current utility

            for dir in directions:  # Loop through all possible actions
                new_pos = self.move(curr_pos, dir)  # Get new position if UAV moves in the specified direction
                # Check if the new position is not already occupied by another UAV
                if new_pos not in uav_moves and new_pos not in occupied_positions:
                    uav_positions_copy = copy.deepcopy(self.uav_positions)
                    uav_positions_copy = self.update_uav_positions(uav_positions_copy, curr_pos, new_pos)  # Update the UAV's hypothetical position
                    uav_new_utility = self.connect_users_to_uavs_original(new_pos, uav_positions_copy)  # Retrieve utility for the new position
                    possible_utilities.append(uav_new_utility)
                    possible_directions.append(dir)

            if verbose: print(f"curr position: {curr_pos}, curr utility: {uav_curr_utility}")
            if verbose: print(f"possible_directions: {possible_directions}, possible_utilities: {possible_utilities}\n")

            if len(possible_utilities) > 0 and uav_curr_utility <= max(possible_utilities):
                best_direction = possible_directions[np.argmax(possible_utilities)]  # Best direction to move towards
                move_to_best_direction = self.move(curr_pos, best_direction) # Move to the best direction
                uav_moves.append(move_to_best_direction)
                is_done.append(False)
                utilities.append(max(possible_utilities))
                if verbose: print(f"if: best_direction: {best_direction}, best_direction coord: {move_to_best_direction}, new_utility: {max(possible_utilities)}, uav_moves: {uav_moves}, is_done: {is_done}\n")
                #self.plot_user_and_uav_positions()

            else:
                uav_moves.append(curr_pos)
                is_done.append(True)
                utilities.append(uav_curr_utility)
                if verbose: print(f"else: utility: {uav_curr_utility}, uav_moves: {uav_moves}, is_done: {is_done}\n")
                #self.plot_user_and_uav_positions()

        return uav_moves, is_done

    def run(self, max_iter, max_dev_not_improved, verbose=False, verbose_plot=True):
        """
        Run the BestResponseLearning algorithm.

        Args:
            max_iter (int): The maximum number of iterations.
            max_dev_not_improved (int): The maximum number of iterations with no improvement in standard deviation.
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

        # Loop through all UAVs and make transition
        while iteration < max_iter:
            uav_new_positions, uav_is_done = self.transition(verbose=False)
            utilities = []
            uav_positions_list = []
            for uav_index in range(len(self.uav_positions)):
                utilities.append(self.connect_users_to_uavs_original(self.uav_positions[uav_index], self.uav_positions))
                uav_positions_list.append(self.uav_positions[uav_index])
            # Save intermediate outputs
            stdev_list.append(np.std(utilities))
            all_uav_positions_list.append(uav_positions_list)
            all_utilities.append(utilities)

            if verbose_plot: print(f"\nIteration: {iteration}, \nall_uav_positions_list: {all_uav_positions_list}, \nall_utilities: {all_utilities}, \nstdev_list: {stdev_list}, \nmin(stdev_list): {min(stdev_list)}")
            # Check if the standard deviation has improved
            if stdev_list[-1] < stdev_best:
                stdev_best = stdev_list[-1]
                stdev_not_improved_cnt = 0
                if verbose: print("improved:", stdev_not_improved_cnt)
            else:
                stdev_not_improved_cnt += 1
                if verbose: print("NOT improved:", stdev_not_improved_cnt)

            if stdev_not_improved_cnt == max_dev_not_improved:
                #print("stdev_not_improved_cnt:", stdev_not_improved_cnt)
                if verbose: print(f"\nTerminating. No improvement from the previous standard deviation of the UAV utilities in the last {max_dev_not_improved} iterations.")
                break

            iteration += 1
            # Update the UAV positions
            self.uav_positions = uav_new_positions
            if verbose_plot: self.plot_user_and_uav_positions()

        # Show the positions with the first best standard deviation in UAV utilities
        best_index = np.argmin(stdev_list)
        if verbose_plot: print(f"Best positions for minimum utility standard deviation: {all_uav_positions_list[best_index]}, Utilities: {all_utilities[best_index]}, StDev: {stdev_list[best_index]}")
        if verbose_plot: plot_uav_positions(self.user_positions, all_uav_positions_list[best_index], self.m)

        return all_uav_positions_list[best_index], all_utilities[best_index]