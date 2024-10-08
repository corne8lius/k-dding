# You can import matplotlib or numpy, if needed.
# You can also import any module included in Python 3.10, for example "random".
# See https://docs.python.org/3.10/py-modindex.html for included modules.

import numpy as np
import random
import pygame

class Robot:
    def __init__(self):
        self.start_position = (0, 3)  # A4
        self.reward_matrix = self.create_reward_matrix()
        self.q_matrix = np.zeros((6, 6, 4))  # 6x6 grid with 4 actions

    def create_reward_matrix(self):
        
        neutral_reward = -10
        fjell_reward = -100
        vann_reward = -50
        goal_reward = 100

        # Define the reward matrix based on the terrain
        reward_matrix = np.full((6, 6), neutral_reward)
        # Assign rewards/penalties based on the terrain

        reward_matrix[:2, 0] = vann_reward
        reward_matrix[2:, 0] = fjell_reward
        reward_matrix[0, 1] = fjell_reward
        reward_matrix[0, 2]  = fjell_reward
        reward_matrix[1, 1] = vann_reward
        reward_matrix[4, 2] = fjell_reward
        reward_matrix[1:3, 3] = fjell_reward
        reward_matrix[4, 4] = fjell_reward
        reward_matrix[0, 5] = vann_reward
        reward_matrix[2, 5] = fjell_reward
        reward_matrix[-1, :] = vann_reward

        reward_matrix[5, 0] = goal_reward  # Goal position F1

        print(reward_matrix)

        # Add other rewards/penalties based on the terrain
        return reward_matrix

    def move(self, action):
        x, y = self.position
        if action == "up" and x > 0:
            x -= 1
        elif action == "down" and x < 5:
            x += 1
        elif action == "left" and y > 0:
            y -= 1
        elif action == "right" and y < 5:
            y += 1
        self.position = (x, y)

    def monte_carlo_simulation(self, num_simulations = 100_000):
        best_reward = -float('inf')
        best_path = []
        for _ in range(num_simulations):
            self.position = self.start_position  # Reset to start position A4
            path = []
            total_reward = 0
            while self.position != (5, 0):  # Until reaching F1
                action = random.choice(["up", "down", "left", "right"])
                self.move(action)
                path.append(self.position)
                total_reward += self.reward_matrix[self.position]
            if total_reward > best_reward:
                best_reward = total_reward
                best_path = path
        return best_path, best_reward

    def q_learning(self, num_episodes, alpha=0.1, gamma=0.9, epsilon=0.1):
        for _ in range(num_episodes):
            self.position = self.start_position  # Reset to start position A4
            while self.position != (0, 5):  # Until reaching F1
                if random.uniform(0, 1) < epsilon:
                    action = random.choice(["up", "down", "left", "right"])
                    print("if", action)
                else:
                    action = np.argmax(self.q_matrix[self.position])
                    print("else", action)
                old_position = self.position
                self.move(action)
                reward = self.reward_matrix[self.position]
                old_q_value = self.q_matrix[old_position][action]
                future_q_value = np.max(self.q_matrix[self.position])
                new_q_value = old_q_value + alpha * (reward + gamma * future_q_value - old_q_value)
                self.q_matrix[old_position][action] = new_q_value

    def visualize_path(self, path):
        pygame.init()
        screen = pygame.display.set_mode((300, 300))
        pygame.display.set_caption("Robot Path Visualization")
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            screen.fill((255, 255, 255))
            for pos in path:
                pygame.draw.rect(screen, (0, 255, 0), (pos[1]*50, pos[0]*50, 50, 50))
            pygame.display.flip()
            clock.tick(60)
        pygame.quit()

# Example usage
robot = Robot()
best_path, best_reward = robot.monte_carlo_simulation(10)
print(f"Best Path: {best_path} ({len(best_path)} steps), Best Reward: {best_reward}")
robot.q_learning(1000)
robot.visualize_path(best_path)