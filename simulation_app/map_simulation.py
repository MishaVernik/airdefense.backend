import numpy as np
import random
import json
from itertools import combinations
from tqdm import tqdm
import time
from qiskit_aer import AerSimulator
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from sklearn.decomposition import PCA

n_shots = 100  # Global parameter for the number of evaluation shots
max_qubits = 15  # Adjust this value to the maximum number of qubits you want to use


def clip_coordinates(x, y, height, width):
    """Clip coordinates to ensure they are within the grid boundaries."""
    x_clipped = np.clip(x, 0, height - 1)
    y_clipped = np.clip(y, 0, width - 1)
    return x_clipped, y_clipped

class Grid:
    """Represents the grid of the map."""

    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.grid = np.zeros((height, width), dtype=int)
        self.missile_paths = []
        self.towers = []
        self.missiles = []
        self.cities = []  # Add cities attribute to the grid

    def add_city(self, x, y):
        """Adds a city to the grid."""
        x, y = clip_coordinates(x, y, self.height, self.width)
        self.grid[x, y] = 2
        self.cities.append(City(x, y))  # Store the city in the grid

    def add_tower(self, x, y, radius=1.5, hit_probability=0.85):
        """Adds a defensive tower to the grid."""
        x, y = clip_coordinates(x, y, self.height, self.width)
        self.grid[x, y] = 3
        self.towers.append(Tower(x, y, radius, hit_probability))

    def add_missile_path(self, path, missile):
        """Stores missile paths for plotting."""
        clipped_path = [clip_coordinates(px, py, self.height, self.width) for px, py in path]
        self.missile_paths.append(clipped_path)
        self.missiles.append(missile)

    def clear_towers(self):
        """Clears all towers from the grid."""
        for tower in self.towers:
            self.grid[tower.x, tower.y] = 0
        self.towers = []

class City:
    """Represents a city on the grid."""
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)

class Missile:
    """Represents a missile on the grid."""
    def __init__(self, start_x, start_y, target_city, TIME):
        self.x = int(start_x)
        self.y = int(start_y)
        self.start_point = (self.x, self.y)  # Store the starting point
        self.target_city = target_city
        self.path = self.compute_path(TIME)
        self.hit_target = False
        self.defended_by_air_defense = False

    def compute_path(self, TIME):
        """Computes the path from the missile's start to the target city."""
        path = [self.start_point]  # Start with the initial point
        x, y = self.x, self.y
        for _ in range(TIME):
            if x < self.target_city.x:
                x += 1
            elif x > self.target_city.x:
                x -= 1
            if y < self.target_city.y:
                y += 1
            elif y > self.target_city.y:
                y -= 1
            path.append((x, y))  # Add each step to the path
        return path


    def get_direction(self):
        """Returns the direction of the missile towards the target city."""
        if self.x < self.target_city.x:
            if self.y < self.target_city.y:
                return "SE"
            elif self.y > self.target_city.y:
                return "SW"
            else:
                return "S"
        elif self.x > self.target_city.x:
            if self.y < self.target_city.y:
                return "NE"
            elif self.y > self.target_city.y:
                return "NW"
            else:
                return "N"
        else:
            if self.y < self.target_city.y:
                return "E"
            elif self.y > self.target_city.y:
                return "W"
        return ""


class Tower:
    """Represents a defensive tower on the grid."""

    def __init__(self, x, y, radius=1.5, hit_probability=0.85):
        self.x = x
        self.y = y
        self.radius = radius
        self.hit_probability = hit_probability

    def is_in_defense_range(self, missile_path):
        """Checks if a missile is within the tower's defense range."""
        for (mx, my) in missile_path:
            if self.x - self.radius <= mx <= self.x + self.radius and self.y - self.radius <= my <= self.y + self.radius:
                return True
        return False


class DefenseOptimization:
    """Represents the optimization problem for placing defensive towers."""

    def __init__(self, grid, cities, missiles, towers):
        self.grid = grid
        self.cities = cities
        self.missiles = missiles
        self.towers = towers

    def optimize_tower_placement(self):
        """Finds the optimal placement of defensive towers using QAOA."""
        start_time = time.time()
        best_tower_locations = self.qaoa_optimize()
        end_time = time.time()
        qaoa_optimization_time = end_time - start_time
        print(f"QAOA optimization time: {qaoa_optimization_time:.2f} seconds")

        # Clear existing towers from the grid
        self.grid.clear_towers()

        # Add best towers to the grid
        for x, y in best_tower_locations:
            # Find the corresponding tower from the input towers to get the radius and hit probability
            for tower in self.towers:
                if tower.x == x and tower.y == y:
                    self.grid.add_tower(x, y, tower.radius, tower.hit_probability)
                    break

        start_time = time.time()
        success_rate = self.evaluate_tower_placement()
        end_time = time.time()
        post_processing_time = end_time - start_time
        print(f"Post-processing time: {post_processing_time:.2f} seconds")

        return best_tower_locations, success_rate, qaoa_optimization_time + post_processing_time

    def evaluate_tower_placement(grid, missiles):
        neutralized_missiles = 0
        for missile in tqdm(missiles, desc="Evaluating missiles"):
            hits = 0
            for _ in range(n_shots):
                hit_probability_accumulated = 1.0
                for (mx, my) in missile.path:
                    for tower in grid.towers:
                        if tower.is_in_defense_range([(mx, my)]):
                            hit_probability_accumulated *= (1 - tower.hit_probability)
                if hit_probability_accumulated < 1:
                    hits += 1
            if hits > n_shots / 2:  # Consider the missile intercepted if more than half of the shots result in a hit
                missile.defended_by_air_defense = True
                neutralized_missiles += 1
            else:
                missile.hit_target = True
        return neutralized_missiles / len(missiles)

    def adjust_towers(grid, missiles):
        best_locations = []
        max_interceptions = 0

        for _ in range(10):  # Adjust up to 10 times
            possible_locations = [(x, y) for x in range(grid.height) for y in range(grid.width)]
            grid.clear_towers()

            # Randomly choose tower positions
            chosen_locations = random.sample(possible_locations, len(grid.towers))
            for x, y in chosen_locations:
                grid.add_tower(x, y)

            interceptions = evaluate_tower_placement(grid, missiles)
            if interceptions > max_interceptions:
                max_interceptions = interceptions
                best_locations = chosen_locations

        grid.clear_towers()
        for x, y in best_locations:
            grid.add_tower(x, y)

    def qaoa_optimize(self):
        """Optimize the placement of towers using QAOA."""
        qp = QuadraticProgram()

        # Define variables for each possible tower location
        for x in range(1, self.grid.height - 1):
            for y in range(1, self.grid.width - 1):
                qp.binary_var(f't_{x}_{y}')

        # Objective function: maximize the number of neutralized missiles
        linear = {}
        for missile in self.missiles:
            for x in range(1, self.grid.height - 1):
                for y in range(1, self.grid.width - 1):
                    var_name = f't_{x}_{y}'
                    if any(tower.is_in_defense_range(missile.path) for tower in self.towers if
                           tower.x == x and tower.y == y):
                        tower = next(tower for tower in self.towers if tower.x == x and tower.y == y)
                        if var_name in linear:
                            linear[var_name] += tower.hit_probability
                        else:
                            linear[var_name] = tower.hit_probability
        qp.maximize(linear=linear)

        # Constraint: exactly `num_towers` towers
        qp.linear_constraint(
            linear={f't_{x}_{y}': 1 for x in range(1, self.grid.height - 1) for y in range(1, self.grid.width - 1)},
            sense='==', rhs=len(self.towers))

        # Convert to QUBO
        qubo = QuadraticProgramToQubo().convert(qp)

        # Use QAOA
        sampler = Sampler()
        optimizer = COBYLA()

        # Progress callback function
        def callback(eval_count, params, value, meta):
            print(f"Iteration {eval_count}: Objective value = {value}")

        qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=1, callback=callback)

        optimizer.set_options(maxiter=5, disp=True)

        result = MinimumEigenOptimizer(qaoa).solve(qubo)

        # Extract best locations
        best_locations = []
        for x in range(1, self.grid.height - 1):
            for y in range(1, self.grid.width - 1):
                if result.x[qubo.variables_index[f't_{x}_{y}']] == 1:
                    best_locations.append((x, y))

        return best_locations


def classical_optimize(grid, cities, missiles, num_towers):
    """Solve the tower placement problem classically using brute-force approach."""
    possible_locations = [(x, y) for x in range(1, grid.height - 1) for y in range(1, grid.width - 1)]
    best_locations = []
    best_success_rate = 0

    for tower_combination in combinations(possible_locations, num_towers):
        try:
            towers = [next(tower for tower in grid.towers if tower.x == x and tower.y == y) for x, y in
                      tower_combination]
        except StopIteration:
            continue
        success_rate = evaluate_tower_placement(towers, missiles)
        if success_rate > best_success_rate:
            best_success_rate = success_rate
            best_locations = tower_combination

    return best_locations, best_success_rate


def evaluate_tower_placement(grid, missiles):
    neutralized_missiles = 0
    for missile in tqdm(missiles, desc="Evaluating missiles"):
        hits = 0
        for _ in range(n_shots):
            hit_probability_accumulated = 1.0
            for (mx, my) in missile.path:
                for tower in grid.towers:
                    if tower.is_in_defense_range([(mx, my)]):
                        hit_probability_accumulated *= (1 - tower.hit_probability)
            if hit_probability_accumulated < 1:
                hits += 1
        if hits > n_shots / 2:  # Consider the missile intercepted if more than half of the shots result in a hit
            missile.defended_by_air_defense = True
            neutralized_missiles += 1
        else:
            missile.hit_target = True
    return neutralized_missiles / len(missiles)

def optimize_tower_placement_with_qaoa(grid, missiles):
    """Optimizes the placement of defensive towers using QAOA."""
    qp = QuadraticProgram()

    # Define variables for each possible tower location
    for x in range(1, grid.height - 1):
        for y in range(1, grid.width - 1):
            qp.binary_var(f't_{x}_{y}')

    # Objective function: maximize the number of neutralized missiles
    linear = {}
    for missile in missiles:
        for x in range(1, grid.height - 1):
            for y in range(1, grid.width - 1):
                var_name = f't_{x}_{y}'
                if any(tower.is_in_defense_range(missile.path) for tower in grid.towers if tower.x == x and tower.y == y):
                    tower = next(tower for tower in grid.towers if tower.x == x and tower.y == y)
                    if var_name in linear:
                        linear[var_name] += tower.hit_probability
                    else:
                        linear[var_name] = tower.hit_probability
    qp.maximize(linear=linear)

    # Constraint: exactly `num_towers` towers
    qp.linear_constraint(
        linear={f't_{x}_{y}': 1 for x in range(1, grid.height - 1) for y in range(1, grid.width - 1)},
        sense='==', rhs=len(grid.towers))

    # Convert to QUBO
    qubo = QuadraticProgramToQubo().convert(qp)

    # Use QAOA
    sampler = Sampler()
    optimizer = COBYLA()

    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=1)

    optimizer.set_options(maxiter=5, disp=True)

    result = MinimumEigenOptimizer(qaoa).solve(qubo)

    # Extract best locations
    best_locations = []
    for x in range(1, grid.height - 1):
        for y in range(1, grid.width - 1):
            if result.x[qubo.variables_index[f't_{x}_{y}']] == 1:
                best_locations.append((x, y))

    return best_locations



def compare_solutions(test_case_name, grid, cities, missiles, towers):
    """Compare the solutions from QAOA and classical approaches."""
    print(f"\n--- {test_case_name} ---")
    print(f"Description: {test_case_name}")

    optimization = DefenseOptimization(grid, cities, missiles, towers)
    best_locations_qaoa, success_rate_qaoa, qaoa_time = optimization.optimize_tower_placement()

    print("QAOA Best Tower Locations:", best_locations_qaoa)
    print("QAOA Success Rate:", success_rate_qaoa)

    # Clear existing towers and add new ones for the final state display
    grid.clear_towers()
    for x, y in best_locations_qaoa:
        for tower in towers:
            if tower.x == x and tower.y == y:
                grid.add_tower(x, y, tower.radius, tower.hit_probability)

    # Display the results
    for idx, missile in enumerate(missiles):
        print(f"Missile {idx + 1} -> Start: ({missile.x}, {missile.y}), Direction: {missile.get_direction()}")
        print(
            f"Missile {idx + 1} -> Hit Target: {missile.hit_target}, Defended by Air Defense: {missile.defended_by_air_defense}")

    # Solve the problem classically and compare the solutions
    start_time = time.time()
    best_locations_classical, success_rate_classical = classical_optimize(grid, cities, missiles, len(towers))
    end_time = time.time()
    classical_time = end_time - start_time

    print("Classical Best Tower Locations:", best_locations_classical)
    print("Classical Success Rate:", success_rate_classical)

    # Compare QAOA and classical solutions
    print(f"\nComparison of QAOA and Classical Solutions for {test_case_name}:")
    print("QAOA Best Tower Locations:", best_locations_qaoa)
    print("Classical Best Tower Locations:", best_locations_classical)
    print("QAOA Success Rate:", success_rate_qaoa)
    print("Classical Success Rate:", success_rate_classical)
    print(f"QAOA Time: {qaoa_time:.2f} seconds")
    print(f"Classical Time: {classical_time:.2f} seconds")


def simulation_to_json(grid, missiles, iteration):
    """Convert simulation data to JSON format for the frontend."""
    simulation_data = []

    for time_step in range(max(len(missile.path) for missile in missiles)):
        step_data = {
            "iteration": int(iteration),
            "time_step": int(time_step),
            "towers": [[int(tower.x), int(tower.y), float(tower.radius)] for tower in grid.towers],
            "targets": [[int(city.x), int(city.y)] for city in grid.cities],
            "rockets": [
                {
                    "position": [int(pos) for pos in (missile.path[time_step] if time_step < len(missile.path) else missile.path[-1])],
                    "intercepted": bool(missile.defended_by_air_defense),
                    "start": [int(missile.start_point[0]), int(missile.start_point[1])],
                    "end": [int(missile.target_city.x), int(missile.target_city.y)],
                    "path": missile.path[:time_step + 1]  # Include the path up to the current time step
                }
                for missile in missiles
            ]
        }
        simulation_data.append(step_data)

    return {"simulation_data": simulation_data}



def reduce_dimensions(data, max_qubits):
    """Reduce the dimensions of the data using PCA to fit within max allowed qubits."""
    n_samples, n_features = data.shape
    n_components = min(max_qubits, n_samples, n_features)

    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)

    return reduced_data

def encode_data(cities, missiles, towers, max_qubits, height, width):
    """Encode the data using PCA to fit within max allowed qubits."""
    # Combine the coordinates of cities, missile start points, and towers into a single dataset
    city_data = np.array([[city.x, city.y] for city in cities])
    missile_data = np.array([[missile.x, missile.y] for missile in missiles])
    tower_data = np.array([[tower.x, tower.y] for tower in towers])

    combined_data = np.vstack((city_data, missile_data, tower_data))

    # Reduce dimensions using PCA
    reduced_data = reduce_dimensions(combined_data, max_qubits)

    # Decode the data back into cities, missiles, and towers
    n_cities = len(cities)
    n_missiles = len(missiles)
    n_towers = len(towers)

    reduced_cities = reduced_data[:n_cities]
    reduced_missiles = reduced_data[n_cities:n_cities + n_missiles]
    reduced_towers = reduced_data[n_cities + n_missiles:]

    # Update the original objects with the reduced coordinates
    for i, city in enumerate(cities):
        city.x, city.y = clip_coordinates(int(reduced_cities[i][0]), int(reduced_cities[i][1]), height, width)

    for i, missile in enumerate(missiles):
        missile.x, missile.y = clip_coordinates(int(reduced_missiles[i][0]), int(reduced_missiles[i][1]), height, width)

    for i, tower in enumerate(towers):
        tower.x, tower.y = clip_coordinates(int(reduced_towers[i][0]), int(reduced_towers[i][1]), height, width)


def random_case():
    height = 5
    width = 5
    TIME = np.random.randint(5, 10)  # Set a random TIME duration between 5 and 10

    num_cities = np.random.randint(1, 4)  # Random number of cities between 1 and 3
    num_missiles = np.random.randint(1, 4)  # Random number of missiles between 1 and 3
    num_towers = np.random.randint(1, 4)  # Random number of towers between 1 and 3

    cities = [City(np.random.randint(0, height), np.random.randint(0, width)) for _ in range(num_cities)]
    missiles = [Missile(np.random.randint(0, height), np.random.randint(0, width), random.choice(cities), TIME) for _ in
                range(num_missiles)]
    towers = [
        Tower(np.random.randint(0, height), np.random.randint(0, width), hit_probability=np.random.uniform(0.7, 0.95))
        for _ in range(num_towers)]

    grid = Grid(height, width)

    for city in cities:
        grid.add_city(city.x, city.y)

    for missile in missiles:
        grid.add_missile_path(missile.path, missile)

    # Call encode_data with all required arguments
    encode_data(cities, missiles, towers, max_qubits=15, height=height, width=width)

    compare_solutions("Random Case", grid, cities, missiles, towers)

    # Generate JSON for frontend
    simulation_json = simulation_to_json(grid, missiles, 1)
    print(simulation_json)  # or save to a file


def run_simulation(height, width, TIME, num_cities, num_missiles, num_towers, max_iterations=4):
    cities = [City(np.random.randint(0, height), np.random.randint(0, width)) for _ in range(num_cities)]
    missiles = [Missile(np.random.randint(0, height), np.random.randint(0, width), random.choice(cities), TIME) for _ in
                range(num_missiles)]

    grid = Grid(height, width)
    for city in cities:
        grid.add_city(city.x, city.y)

    # Initial Random Tower Placement
    for _ in range(num_towers):
        grid.add_tower(np.random.randint(0, height), np.random.randint(0, width),
                       hit_probability=np.random.uniform(0.7, 0.95))

    # Store simulation data for each iteration
    all_iterations_data = []

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        # Evaluate the current tower placement
        evaluate_tower_placement(grid, missiles)

        # Store data for this iteration
        iteration_data = simulation_to_json(grid, missiles, iteration + 1)
        all_iterations_data.append(iteration_data["simulation_data"])

        # Optimize tower placement using QAOA for the next iteration
        best_tower_locations = optimize_tower_placement_with_qaoa(grid, missiles)
        grid.clear_towers()
        for x, y in best_tower_locations:
            grid.add_tower(x, y)

    # Combine all iteration data into a single JSON structure
    simulation_json = {"simulation_data": [data for iteration in all_iterations_data for data in iteration]}
    return simulation_json