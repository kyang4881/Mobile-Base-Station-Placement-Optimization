from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import math

def agent_combinations(agents):
    """
    Generate all possible combinations of agents.

    Args:
        agents (list): A list of agent objects.

    Returns:
        list: A list of tuples representing agent combinations.
    """
    combinations_result = []
    for r in range(1, len(agents)+1):
        combinations_result.extend(combinations(agents, r))
    return combinations_result

def agent_coalitions(elements):
    """
    Generate all possible coalitions of a set of elements.

    Args:
        elements (list): A list of elements.

    Returns:
        set: A set of unique coalition tuples.
    """
    all_combinations = set()  # Use a set to store unique combinations
    if len(elements) > 1:
        for r in range(1, len(elements)):
            for combo in combinations(elements, r):
                group1 = combo
                group2 = tuple(e for e in elements if e not in combo)
                # Sort the groups to ensure uniqueness, as order doesn't matter
                unique_combination = tuple(sorted((group1, group2)))
                all_combinations.add(unique_combination)
        #all_combinations.add(elements) # add itself
    else:
        all_combinations = tuple(elements)
    return all_combinations

def compute_optimal(coalition_values, agents, verbose=False):
    """
    Compute the optimal coalition structure and values for a given set of agents.

    This function iterates through all possible coalitions and their sub-coalitions
    to find the optimal coalition structure that maximizes the total value.

    Args:
        coalition_values (dict): A dictionary where keys are coalitions (tuples of agents) and values are the values associated with those coalitions.
        agents (list): A list of agent objects.
        verbose (bool, optional): If True, print debugging information. Default is False.

    Returns:
        dict: A dictionary where keys are coalitions (tuples of agents) and values arethe optimal values for those coalitions.
    """
    best_split_values = {}

    for coalition in agent_combinations(agents):
        if verbose: print(f"coalition_values: {coalition_values}")
        if verbose: print(f"best_split_values: {best_split_values}\n")
        if verbose: print(f"coalition: {coalition}")
        coalition_split = agent_coalitions(coalition)
        if verbose: print(f"coalition_split: {coalition_split}")

        all_c_split = []
        if len(coalition_split) > 1:
            values = [coalition_values[coalition]] # input value
            coalitions = [coalition]
            best_coalition_index = None

            for c_split in coalition_split:
                c_split_tuple = tuple(c_split)

                split_value = coalition_values[c_split_tuple[0]] + coalition_values[c_split_tuple[1]]
                if verbose: print(f"len(coalition_split) > 1: split_value = coalition_values[c_split_tuple[0]] + coalition_values[c_split_tuple[1]]: {coalition_values[c_split_tuple[0]]} + {coalition_values[c_split_tuple[1]]} = {split_value}")

                values.append(split_value)  # append best value

                coalitions.append(c_split_tuple)

            best_coalition_index = np.argmax(values)

            if verbose: print(f"len(coalition_split) > 1: coalitions: {coalitions}")
            if verbose: print(f"len(coalition_split) > 1: values: {values}")
            if verbose: print(f"len(coalition_split) > 1: best_coalition_index: {best_coalition_index}")

            if best_coalition_index == 0:
                best_split_values[coalition] = values[best_coalition_index]
                coalition_values[coalition] = values[best_coalition_index]
            else:
                best_split_values[coalitions[best_coalition_index]] = values[best_coalition_index]
                coalition_values[coalition] = values[best_coalition_index]

        else:
            if len(coalition) == 1:
                best_split_values[coalition] = coalition_values[coalition]
                if verbose: print("len(coalition) == 1")
            else:
                values = [coalition_values[coalition]] # input value
                coalitions = [coalition]
                best_coalition_index = None

                split_value = coalition_values[tuple(coalition[0])] + coalition_values[tuple(coalition[1])]
                if verbose: print(f"len(coalition) > 1: split_value = coalition_values[tuple(coalition[0])] + coalition_values[tuple(coalition[1])]: {coalition_values[tuple(coalition[0])]} + {coalition_values[tuple(coalition[1])]} = {split_value}")

                values.append(split_value)
                coalitions.append(list(coalition_split)[0])

                best_coalition_index = np.argmax(values)
                if best_coalition_index == 0:
                    best_split_values[coalition] = values[best_coalition_index]
                    coalition_values[coalition] = values[best_coalition_index]
                else:
                    best_split_values[coalitions[best_coalition_index]] = values[best_coalition_index]
                    coalition_values[coalition] = values[best_coalition_index]

                if verbose: print(f"len(coalition) > 1: coalitions: {coalitions}")
                if verbose: print(f"len(coalition) > 1: values: {values}")
                if verbose: print(f"len(coalition) > 1: best_coalition_index: {best_coalition_index}")

    if verbose: print(f"coalition_values: {coalition_values}")
    if verbose: print(f"best_split_values: {best_split_values}\n")
    return best_split_values

def plot_uav_positions(user_positions, uav_positions, m):
    """
    Plot the positions of users and UAVs in a 2D grid.

    Args:
        user_positions (list of tuples): List of (x, y) coordinates of user positions.
        uav_positions (list of tuples): List of (x, y) coordinates of UAV positions.
        m (int): The size of the grid in both X and Y dimensions.

    Returns:
        None
    """
    # Extract x and y coordinates for users and UAVs
    user_x, user_y = zip(*user_positions)
    uav_x, uav_y = zip(*uav_positions)
    # Create a plot to display user and UAV positions
    plt.figure(figsize=(5, 5))
    plt.scatter(user_x, user_y, color='grey', label='Users', marker='o', s=30)
    plt.scatter(uav_x, uav_y, color='blue', label='UAVs', marker='^', s=80)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('User and UAV Positions')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # Set the grid to show 10x10 with 1 spacing
    plt.xticks(range(1, m + 1))
    plt.yticks(range(1, m + 1))
    plt.grid(True, which='both', linestyle='--', lw=1)
    plt.show()

def social_opt_position(m):
    """
    Compute the socially optimal positions for 2 UAVs

    Args:
        m (int): the size of the grid

    Returns:
        A list of socially optimum positions
    """
    x_min = y_min = 1
    x_max = y_max = m
    x_mid = y_mid = (m+1)/2

    if m % 2 == 0: # If even
        # Even
        even_opt_1 = [
            (math.floor((x_min + x_mid)/2), math.floor((y_min + y_max)/2)), (math.ceil((x_mid + x_max)/2), math.ceil((y_min + y_max)/2))
        ]

        even_opt_2 = [
            (math.floor((x_min + x_mid)/2), math.ceil((y_min + y_max)/2)), (math.ceil((x_mid + x_max)/2), math.floor((y_min + y_max)/2))
        ]

        even_opt_3 = [
            (math.floor((x_min + x_max)/2), math.floor((y_min + y_mid)/2)), (math.ceil((x_min + x_max)/2), math.ceil((y_mid + y_max)/2))
        ]

        even_opt_4 = [
            (math.floor((x_min + x_max)/2), math.ceil((y_mid + y_max)/2)), (math.ceil((x_min + x_max)/2), math.floor((y_min + y_mid)/2))
        ]

        return [even_opt_1, even_opt_2, even_opt_3, even_opt_4]

    else: # Odd
        odd_opt_1 = [
            (int((x_min + x_mid)/2), int((y_min + y_max)/2)), (int((x_mid + x_max)/2), int((y_min + y_max)/2))
        ]

        odd_opt_2 = [
            (int((x_min + x_max)/2), int((y_min + y_mid)/2)), (int((x_min + x_max)/2), int((y_mid + y_max)/2))
        ]

        return [odd_opt_1, odd_opt_2]

def user_utility(model, uav_positions_):
    """
    Calculate the utility based on the total distances of all users relative to the UAVs.

    Args:
        uav_positions_ (list of tuples): List of UAV positions.

    Returns:
        float: Total distance-based utility.
    """
    total_distance_utility = 0.0

    for user_position in model.user_positions:
        min_distance = float('inf')

        for uav_position in uav_positions_:
            distance = model.euclidean_distance(user_position, uav_position)
            if distance < min_distance:
                min_distance = distance

        total_distance_utility += min_distance

    return -total_distance_utility