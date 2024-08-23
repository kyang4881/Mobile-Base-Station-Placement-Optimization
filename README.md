# Mobile Base Station Placement

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Multi-Agent%20Systems%20Projects/Drone%20Project/docs/images/background.jpg" width="1200" />
</p>


---

## Problem Statement 

A large sporting event can easily draw crowds of spectators that completely overwhelm local communication system. A recent solution is to contract telecommunication companies to provide temporary network connectivity by using a number of elevated mobile base stations (hosted on airborne unmanned aerial vehicles (UAVs), or drones). The question is how we fly these UAV nodes in order to provide optimal connectivity for the many ground-based mobile phone users.

In this problem we will try to answer this question using game-theoretic approaches. We assume self-interested agents: mobile phone users who aim to optimize their connectivity (i.e., choose base station based on signal power; for simplicity, assume that the closest base station provides the strongest signal) and each base station maximizes the number of served users (i.e., count the number of connected devices). More specifically, we define agent utilities below:

For mobile phone user 𝑖, his/her utility is defined as $u_i = \min_{k=1,...,K}d_ik$ where 𝐾 is the number of UAVs, and $d_ik$ is the Euclidean distance between user 𝑖 and UAV 𝑘. For UAV 𝑘, it’s utility is defined as the number of users connected to it.

Consider the following assumptions:
* Mobile phone users are uniformly spread out over an m-by-m area. For simplicity, we assume that each user occupies a (𝑥,𝑦) coordinate, where 1≤𝑥,𝑦≤𝑚, and both 𝑥 and 𝑦 are integers.
* All UAVs’ coordinates are also integers.

---

## Notebook 

Assume that the two UAV nodes are initially placed uniformly randomly in the 2D square area. Also assume that a UAV could move one unit to the direction of either north, east, south, or west per time step. Implement the following simple movement algorithm as the baseline (this is sometimes called the Best-Response Learning algorithm, or simply BRL

### What We Know

* Players: The UAVs
* Actions: Set of directions to move the UAVs {North, East, South, West} in an m by m gridworld
* Utility Fuction: $UAV_k = \sum_{k=1,...,K}$ users {1 if user i is has the shortest distance with $UAV_k$, 1/2 if equidistant, 0 otherwise}

Nash Equilibrium (NE): The utility for each UAV is determined by the number of users it serves. A Nash equilibrium is reached when neither UAV has an incentive to unilaterally change its position, given the position of the other UAV. In other words, each UAV is serving as many users as it can given its current position, and moving to a different location would not improve its utility. In this case, in a best case scenario, NE occurs when each UAV obtains a utility of $(m*m)/2$, which is equivalent to splitting the users evenly among the UAVs. When m is odd, the utility cannot be split evenly among the uavs if they are side-by-side against each other but can if m is even. Also, the location of the UAVs at NE may not be optimal for the users.

Socially Optimal (SO): The socially optimal solution would be to find a placement of the two UAVs that maximizes the overall utility (minimizing the distance between the UAVs and the users) for all users while ensuring fair distribution as well as the utility for the UAVs. This means positioning the UAVs to serve users in a way that minimizes the overall sum of distance (sum of distances to the nearest UAV) across all users. The objective is then to minimize $sum_{i} (d_{i1} + d_{i2})$, the sum of the distances for both UAVs.

When m is odd, the grid has a central point. The socially optimal solution occurs when the UAVs can be positioned in a way that each UAV serves roughly half of the users relative to the center and positioned in any of the 4 directions to balance the user distribution for minimal total user distance. There are two cases that satisfy those conditions, as defined in the social_opt_position function.

When m is even, there is no central point. The UAVs can be positioned such that each UAV serves approximately half of the users, but the division is not necessarily centered in the grid. A possible approach is to position the UAVs along one of the central rows or columns to balance user allocation. There are 4 cases that satisfy these conditions, as defined in the social_opt_position function.

---

## Best-Response Learning Algorithm

Helper functions to visualize drone positions, and to determine the socially optimum postiions and user utility.

```python

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
```

A BestResponseLearning class that includes method for moving the drones, calculating the distance between drones, finding the closest drone to a user, calculating utility, performing transition step for the UAVs based on their best response strategy, and plotting the results on a graph.

### Instantiate the learning process

```python
brl = BestResponseLearning(num_uav=2, m=10)
brl_position_result = brl.run(max_iter=1000, max_dev_not_improved=10, verbose=False)
```

The termination process in the provided BestResponseLearning class is part of the algorithm's run loop and determines when to stop the iterative process. The termination process is based on a criterion related to the improvement of the standard deviation of UAV utilities over a set number of iterations. The goal is to stop the loop from going to infinity when there is no further significant improvement or change in the transition process.

1. The algorithm starts with an initial set of UAV positions, and a list to keep track of the standard deviation of utilities across UAVs at each iteration.
2. The algorithm enters an iterative loop with a maximum number of iterations specified by the max_iter parameter.
3. In each iteration, the algorithm performs a transition for each UAV. The transition involves considering possible movements for each UAV, calculating the utility for the current position, and evaluating the utilities for potential new positions based on the Best-Response Learning (BRL) algorithm.
4. After the transition for all UAVs, the algorithm calculates the standard deviation of utilities for all UAVs. The standard deviation reflects how evenly users are distributed among UAVs.
5. The algorithm checks if the current standard deviation is better (lower) than the best standard deviation observed so far. If it is, the current standard deviation becomes the new best standard deviation, and a counter for the number of iterations with no improvement (stdev_not_improved_cnt) is reset to zero.
6. If the current standard deviation is not better than the best standard deviation, the counter for the number of iterations with no improvement is incremented (stdev_not_improved_cnt).
7. The termination process checks whether the counter stdev_not_improved_cnt has reached a predefined threshold, which is specified by the max_dev_not_improved parameter. If the counter reaches this threshold, it means that there has been no improvement in the standard deviation of utilities for the past max_dev_not_improved iterations.
8. If the termination condition is met (stdev_not_improved_cnt equals max_dev_not_improved), the algorithm terminates and stops further iterations

Depending on the random initialization, the two-node case may or may not end up in a Nash equilibrium. Since the UAVs take turn to move, the UAV that reaches the center sooner (captures higher utility due to closer proximity to more users) may execute best responses to prevent the other UAV from taking a better position with a higher utility. This happens due to the game being turn-based and multiple UAVs cannot occupy the same coordinate in the grid. Theforefore, it's possible for the UAVs to get blocked from achieving the best equilibrium and they may also oscillate in certain positions in an infinite loop.

The Nash equilibrium tends to occur in the center of the grid with the two UAVs positioned side by side, as a result each uav may capture half of all users. While the two UAVs has reached a stable equilibrium because neither has the incentive to deivate, it may not be socially optimal for the users. The total distance of the users to the UAVs could be unproportionately distributed, as those along the center region would have the shortest distances to the UAVs, and those further out along the edges tend to have longer distances, or if the user distribution between the uavs are imbalanced. Therefore, the sum of the distances of all users is not minimized, which is required under social optimality.

As shown below, the social optimum cases have a higher utility than in the case of the Nash equilibrium.

### Nash Equilibrium

```python
>>> print(f"UAV Position: {brl_position_result[0]}, UAV Utility: {brl_position_result[1]}, User Utility: {brl.compute_user_utility_by_uav(brl_position_result[0])}, Total User Utility: {np.sum(list(brl.compute_user_utility_by_uav(brl_position_result[0]).values()))}")

UAV Position: [(7, 7), (8, 6)], UAV Utility: [59.5, 40.5], User Utility: {(7, 7): -245.60921, (8, 6): -142.01965}, Total User Utility: -387.62886000000003
```
    
### Social Optimums

```python
>>> for positions in social_opt_position(m=10):
    print(f"UAV Position: {positions}, UAV Utility: {[brl.connect_users_to_uavs_original(position, positions) for position in positions]}, User Utility: {brl.compute_user_utility_by_uav(positions)}, Total User Utility: {np.sum(list(brl.compute_user_utility_by_uav(positions).values()))}")

UAV Position: [(3, 5), (8, 6)], UAV Utility: [50.0, 50.0], User Utility: {(3, 5): -147.66515, (8, 6): -147.66515}, Total User Utility: -295.3303
UAV Position: [(3, 6), (8, 5)], UAV Utility: [50.0, 50.0], User Utility: {(3, 6): -147.66515, (8, 5): -147.66515}, Total User Utility: -295.3303
UAV Position: [(5, 3), (6, 8)], UAV Utility: [50.0, 50.0], User Utility: {(5, 3): -147.66515, (6, 8): -147.66515}, Total User Utility: -295.3303
UAV Position: [(5, 8), (6, 3)], UAV Utility: [50.0, 50.0], User Utility: {(6, 3): -147.66515, (5, 8): -147.66515}, Total User Utility: -295.3303
```

With the addition of more UAV nodes, the competition among UAVs to serve users intensifies. Each UAV is self-interested and aims to maximize the number of users it serves. The equilibrium outcomes, in this case, will involve a complex interplay of UAV positions as they continuously adjust to optimize their utilities. The algorithm will seek to find positions for UAVs that balance user allocations to maximize their individual utilities. The UAV nodes appear to cluster near the center region. With 3 UAVs, they tend to still stick near each other with no node of separation. Beyond 3 UAV nodes however, it can be observed that the nodes have a tendency to split into groups of two and stick around each other.

These equilibrium outcomes are still not socially optimal because socially optimal outcomes are determined by minimizing the total distance for all users across the grid, ensuring that each UAV serves users efficiently and equally. The self-interested nature of UAVs makes them continue to search for optimal positions to maximize individual utility rather than achieving a globally optimal distribution of users in most cases. However, assuming that m stays fixed, when the number of UAV nodes increase, the distance among the nodes tend to increase, thus the average distance between users and UAVs decreases, although is still not socially optimal, the utility of the users should increase.

```python
brl = BestResponseLearning(num_uav=3, m=10)
brl.run(max_iter=1000, max_dev_not_improved=10, verbose=False)

brl = BestResponseLearning(num_uav=4, m=10)
brl.run(max_iter=1000, max_dev_not_improved=10, verbose=False)
```

Q learning is a reinforcement learning algorithm well-suited for solving problems with discrete state and action spaces, making it an appropriate choice for optimizing UAV action selection. In this problem, the state space represents the positions of UAVs and users within an m-by-m area, while actions are discrete movements: North, East, South, and West. Q-Learning could navigates this discrete landscape. The algorithm balances exploration and exploitation by allowing UAVs to explore different positions while also exploiting known good ones, crucial for optimizing connectivity. It does so through an exploration probability, ensuring a probabilistic approach to action selection.

The Q learning model's goal is to learn the optimal action-value function, representing expected cumulative rewards for taking actions in given states. This aligns well with the objective of finding the best UAV placement to maximize user connectivity. It does so through iteratively updating Q-values based on observed rewards (user connectivity) and transitions (UAV movements), ultimately converging to the optimal policy. More importantly, Q-Learning is model-free, making it suitable for dynamic, complex scenarios where explicit modeling may be challenging. Like the BRL model, this model also incorporates a termination condition based on the standard deviation of utilities, allowing it to terminate when further optimization is unlikely. Also, the Q learning algorithm is flexible to changes and scales well as the number of UAVs and users increases, with its ability to handle discrete state and action spaces, along with its ability to balance exploration and exploitation, which is makes it suitable for this problem.

---

## Reinforcement Learning

A class that uses Q-Learning to train the agents

```python
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
```

Instantiate the model and train the agent then run the model for 2 UAVs

```python
ql = QLearningBestResponse(num_uav=2, m=10, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.1)
ql.train(num_episodes=1000)

ql.uav_positions = random.sample(ql.user_positions, ql.num_uav)
ql.run(max_iter=1000, max_dev_not_improved=10, verbose=False)
```

We observe that the UAVs have converged to a nash equilibrium 

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Multi-Agent%20Systems%20Projects/Drone%20Project/docs/images/qlearning_uav.png" width="400" />
</p>

Best Response Learning Results

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Multi-Agent%20Systems%20Projects/Drone%20Project/docs/images/br_results.png" width="1200" />
</p>

Reinforcement Learning Results

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Multi-Agent%20Systems%20Projects/Drone%20Project/docs/images/rl_results.png" width="1200" />
</p>


Notice the following:

* The best response learning model saves on training time because it does not need to learn the set of all possible moves to consider when making the next best move, it simply checks the most ideal next move based on the utility of the UAVs for the next move and not future moves.
* The Q learning model needs to discount for future moves' rewards, thus require training with a set number of episodes. This leads to longer initial run time due to the training process which attempts to learn the set of optimal moves that maximize future utilities for the UAVs. The current Q learning model may also require retraining when the number of UAV changes to improve performance, thus increases computational time at the initial phase.
* As the grid size (m) scales larger, it becomes more computational expensive to train the Q learning model. The Q learning model is also heavily dependent on the q table which contains the set of q values, that determine the action to take for a given state. This means it's vital to have enough training episodes for the optimal actions to be learned, otherwise random actions will be selected. While training takes up more time initially, later computation of the action in the Q learning model is more efficient than the BRL model because the Q values are stored in a lookup table and the action can be quickly obtained rather than having to compute the utilities to determine the best next action for each time step.
* For a set of fixed starting positions of the UAVs, it appears that without sufficient training of the Q learning model, the BRL model outperforms it. However, increasing the number of training episodes can improve the performance of the Q learning model to surpass that of the BRL model.
* The obtained equilibrium outcomes are still not socially optimal, thus as we see in socially optimal results below, the utilities of the users in a socially optimal case when m=100 with 2 UAVs is still significantly better than the results from both models. However, as expected, the users' total utility improves as the number of UAVs increases.

---

## Social Optimal

It is possible to coordinate the competing UAV nodes into a socially optimal outcome by altering the objective function. The original objective function was based on the maximization of total utility for the UAVs, derived from counting the number of users connected to each uav based on their relevative minimum distance, thus resulted in the UAVs aggregating close to the center region and close to one another. The updated approach seek to maximize user utility as the objective instead, by minimizing the sum of the users' distance to the nearest uav. Therefore, user utility is presented as a negative value because maximizing this value result in the minimum total distance. It can be seen that changing this objective function lead to the UAVs moving in the direction that would lead to better user outcome

Update the objective function

```python
def connect_users_to_uavs(self, uav_positions_):
    """
    Calculate the utility based on the total distances of all users relative to the UAVs. Modified objective for social optimality

    Args:
        uav_positions_ (list of tuples): List of UAV positions.

    Returns:
        float: Total distance-based utility.
    """
    total_distance_utility = 0.0

    for user_position in self.user_positions:
        min_distance = float('inf')

        for uav_position in uav_positions_:
            distance = self.euclidean_distance(user_position, uav_position)
            if distance < min_distance:
                min_distance = distance

        total_distance_utility += min_distance

    return total_distance_utility

```

Socially Optimal Reinforcement Learning Results

<p align="center">
  <img src="https://github.com/kyang4881/KYGit/blob/master/Multi-Agent%20Systems%20Projects/Drone%20Project/docs/images/social_opt_rl_results.png" width="1200" />
</p>

---


