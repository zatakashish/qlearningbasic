import numpy as np
import random

# Define the environment
class WarehouseEnv:
    def __init__(self, grid_size, pick_up_points, drop_off_points):
        self.grid_size = grid_size
        self.pick_up_points = pick_up_points
        self.drop_off_points = drop_off_points
        self.state = None
        self.reset()
    
    def reset(self):
        self.state = (0, 0, 'pick')  # Starting at the top-left corner, initial task is 'pick'
        return self.state
    
    def step(self, action):
        x, y, task = self.state
        
        if action == 'up' and x > 0:
            x -= 1
        elif action == 'down' and x < self.grid_size[0] - 1:
            x += 1
        elif action == 'left' and y > 0:
            y -= 1
        elif action == 'right' and y < self.grid_size[1] - 1:
            y += 1
        elif action == 'pick' and task == 'pick' and (x, y) in self.pick_up_points:
            task = 'drop'
            reward = 10  # Reward for picking up
            self.state = (x, y, task)
            return self.state, reward, False
        elif action == 'drop' and task == 'drop' and (x, y) in self.drop_off_points:
            task = 'pick'
            reward = 10  # Reward for dropping off
            self.state = (x, y, task)
            return self.state, reward, True  # Episode ends after drop-off
        
        self.state = (x, y, task)
        reward = -1  # Time penalty for each move
        return self.state, reward, False

    def get_state_space_size(self):
        return self.grid_size[0] * self.grid_size[1] * 2  # Including 'pick' and 'drop' states

    def get_action_space_size(self):
        return 6  # up, down, left, right, pick, drop

# Initialize Q-learning parameters
grid_size = (5, 5)
pick_up_points = [(0, 4), (4, 0)]
drop_off_points = [(4, 4), (0, 0)]

env = WarehouseEnv(grid_size, pick_up_points, drop_off_points)
state_space_size = env.get_state_space_size()
action_space_size = env.get_action_space_size()

Q = np.zeros((state_space_size, action_space_size))

# Define the Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate
num_episodes = 1000

# Map actions to index
actions = ['up', 'down', 'left', 'right', 'pick', 'drop']
action_to_index = {action: idx for idx, action in enumerate(actions)}

# Train the agent
for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        x, y, task = state
        state_index = x * grid_size[1] * 2 + y * 2 + (0 if task == 'pick' else 1)

        if random.uniform(0, 1) < epsilon:
            action_index = random.randint(0, action_space_size - 1)  # Explore
        else:
            action_index = np.argmax(Q[state_index])  # Exploit

        action = actions[action_index]
        next_state, reward, done = env.step(action)

        next_x, next_y, next_task = next_state
        next_state_index = next_x * grid_size[1] * 2 + next_y * 2 + (0 if next_task == 'pick' else 1)

        Q[state_index, action_index] += alpha * (reward + gamma * np.max(Q[next_state_index]) - Q[state_index, action_index])

        state = next_state

# Evaluate the agent
def evaluate_agent(env, Q):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        x, y, task = state
        state_index = x * grid_size[1] * 2 + y * 2 + (0 if task == 'pick' else 1)
        action_index = np.argmax(Q[state_index])
        action = actions[action_index]
        next_state, reward, done = env.step(action)
        total_reward += reward
        state = next_state

    return total_reward

total_rewards = [evaluate_agent(env, Q) for _ in range(100)]
print(f"Average Reward over 100 episodes: {np.mean(total_rewards)}")
