import numpy as np
import json
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

class MapSimulation:
    def __init__(self, N, M, K, T, R, TIME, num_targets):
        self.N = N  # Number of rows in the matrix
        self.M = M  # Number of columns in the matrix
        self.K = K  # Number of towers
        self.T = T  # Radius of each tower
        self.R = R  # Number of rockets
        self.TIME = TIME  # Total time steps for rocket movement
        self.num_targets = num_targets  # Number of targets
        self.matrix, self.towers = self.generate_matrix()
        self.targets = self.setup_targets()
        self.rockets = self.setup_rockets()
        self.backend = AerSimulator()

    def generate_matrix(self):
        matrix = np.zeros((self.N, self.M))  # Initialize the matrix
        towers = []

        for _ in range(self.K):
            x, y = np.random.randint(0, self.N), np.random.randint(0, self.M)
            towers.append((x, y))
            for i in range(max(0, x-self.T), min(self.N, x+self.T+1)):
                for j in range(max(0, y-self.T), min(self.M, y+self.T+1)):
                    matrix[i][j] = 1

        return matrix, towers

    def setup_targets(self):
        targets = []
        for _ in range(self.num_targets):
            while True:
                x, y = np.random.randint(0, self.N), np.random.randint(0, self.M)
                if self.matrix[x, y] == 0:  # Ensure the target is outside of any tower's range initially
                    targets.append((x, y))
                    break
        return targets

    def setup_rockets(self):
        rockets = []
        for _ in range(self.R):
            trajectory = []
            while True:
                start_x, start_y = np.random.randint(0, self.N), np.random.randint(0, self.M)
                if self.matrix[start_x][start_y] == 0:  # Ensure the start is outside of any tower's range
                    trajectory.append((start_x, start_y))
                    break

            target_x, target_y = self.targets[np.random.randint(0, len(self.targets))]

            # Create a straight path to the target
            current_x, current_y = start_x, start_y
            while (current_x, current_y) != (target_x, target_y):
                if current_x < target_x:
                    current_x += 1
                elif current_x > target_x:
                    current_x -= 1

                if current_y < target_y:
                    current_y += 1
                elif current_y > target_y:
                    current_y -= 1

                trajectory.append((current_x, current_y))
                if len(trajectory) >= self.TIME:  # Limit the trajectory to the TIME steps
                    break

            rockets.append(trajectory)
        return rockets

    def run_simulation(self):
        intercepted = [False] * self.R
        simulation_data = []

        for iteration in range(1, 3):  # Run 2 iterations for example
            for time_step in range(self.TIME):
                current_frame = {
                    "iteration": iteration,
                    "time_step": time_step,
                    "towers": self.towers,
                    "targets": self.targets,
                    "rockets": [],
                }

                # Simulate rocket interceptions and update the rocket's state
                for idx, rocket in enumerate(self.rockets):
                    if len(rocket) > time_step:
                        position = rocket[time_step]
                        rocket_data = {
                            "position": position,
                            "intercepted": intercepted[idx],
                            "start": rocket[0],
                            "end": rocket[-1],
                        }
                        current_frame["rockets"].append(rocket_data)

                        if self.matrix[position] == 1 and not intercepted[idx]:
                            intercepted[idx] = True

                simulation_data.append(current_frame)

        return simulation_data

