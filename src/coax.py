import csv
import random
import os
import time
import numpy as np
from pyaedt import Hfss, Desktop

# Initialize AEDT Desktop
desktop = Desktop(non_graphical=False, new_desktop_session=True)

# Provide the full path of the project
project_path = r"D:\AI\coax\coax_act.aedt"
if not os.path.exists(project_path):
    raise FileNotFoundError(f"The project file does not exist: {project_path}")

# Load the AEDT project
hfss = Hfss()
hfss.load_project(project_path)

# Get the setup name (use the first setup if available)
if hfss.existing_analysis_setups:
    setup_name = hfss.existing_analysis_setups[0]
    print(f"Using setup: {setup_name}")
else:
    raise ValueError("No analysis setups found in the design!")

patch_length_min = 13.09
patch_length_max = 26.19
patch_width_min = 14.82
patch_width_max = 29.64
coax_outer_rad = 2
feed_pos_x_min, feed_pos_x_max = -((patch_length_min / 2) - coax_outer_rad), ((patch_length_min / 2) - coax_outer_rad)
feed_pos_y_min, feed_pos_y_max = -((patch_width_min / 2) - coax_outer_rad), ((patch_width_min / 2) - coax_outer_rad)

# Define the bounds for each parameter
lower_bound = [patch_length_min, patch_width_min, feed_pos_x_min, feed_pos_y_min]
upper_bound = [patch_length_max, patch_width_max, feed_pos_x_max, feed_pos_y_max]

# DE Algorithm parameters
F = 0.5  # Scaling factor
Pcr = 0.7  # Crossover probability
dim = 4  # Number of parameters (patch_length, patch_width, feed_pos_x, feed_pos_y)
max_iter = 1  # Number of generations
population_size = 5  # Number of solutions in the population

# Generate 5 random solutions within the boundaries
solutions = {}
for i in range(1,6):
    patch_length = round(np.random.uniform(patch_length_min, patch_length_max), 1)
    patch_width = round(np.random.uniform(patch_width_min, patch_width_max), 1)
    feed_pos_x = round(np.random.uniform(feed_pos_x_min, feed_pos_x_max), 1)
    feed_pos_y = round(np.random.uniform(feed_pos_y_min, feed_pos_y_max), 1)
    solutions[i] = [patch_length, patch_width, feed_pos_x, feed_pos_y]

# Print the initial population
print("\n=== INITIAL POPULATION ===")
for i, sol in solutions.items():
    params_formatted = [f"{v:.1f}" for v in sol]
    print(f"Solution x{i}: {params_formatted}")

def objective_function(params):
    """
    Objective function that prioritizes minimizing S11 at 5.65 GHz while penalizing better responses at other frequencies.
    params: [patch_length, patch_width, feed_pos_x, feed_pos_y]
    """
    patch_length, patch_width, feed_pos_x, feed_pos_y = params
    target_freq = 5.65  # Target frequency in GHz
    freq_window = 0.1   # Frequency window for considering nearby responses (±100 MHz)

    try:
        # Update the project variables with the new parameters
        hfss.variable_manager.set_variable('patchX', f"{patch_length / 10:.1f}cm")
        hfss.variable_manager.set_variable('patchY', f"{patch_width / 10:.1f}cm")
        hfss.variable_manager.set_variable('feedX', f"{feed_pos_x / 10:.1f}cm")
        hfss.variable_manager.set_variable('feedY', f"{feed_pos_y / 10:.1f}cm")

        # Run the analysis
        hfss.analyze_setup(setup_name)

        # Wait for analysis completion
        while True:
            try:
                s_parameters = hfss.post.get_solution_data(
                    expressions=["dB(St(coax_pin_T1,coax_pin_T1))"],
                    primary_sweep_variable="Freq"
                )
                break
            except Exception:
                time.sleep(5)

        frequencies = np.array(s_parameters.primary_sweep_values)
        s11_values = np.array(s_parameters.data_real())

        # Find S11 at target frequency
        target_idx = np.argmin(np.abs(frequencies - target_freq))
        s11_at_target = s11_values[target_idx]

        # Find the minimum S11 in the entire frequency range
        min_s11_overall = np.min(s11_values)

        # Initialize penalty score (not added to S11, just used to guide optimization)
        penalty_score = 0

        # Calculate penalty if better response exists away from target frequency
        if min_s11_overall < s11_at_target:
            # Find where the minimum occurs
            min_freq_idx = np.argmin(s11_values)
            min_freq = frequencies[min_freq_idx]
            
            # Calculate frequency deviation from target
            freq_deviation = abs(min_freq - target_freq)
            
            # Apply penalties
            frequency_penalty = 10 * np.exp(freq_deviation / 0.1) - 10  # Exponential penalty
            magnitude_penalty = (5 * (s11_at_target - min_s11_overall)) ** 2  # Magnitude penalty
            window_penalty = 10 if freq_deviation > freq_window else 0  # Penalty for deviation beyond window
            local_minima_penalty = 15 if np.sum(s11_values < s11_at_target) > 1 else 0  # Multiple minima penalty

            # Sum all penalties
            penalty_score = frequency_penalty + magnitude_penalty + window_penalty + local_minima_penalty

        print(f"Penalty Score: {penalty_score:.1f}")

        # The objective function should return S11 at 5.65 GHz, and penalty is used only for optimization guidance
        return s11_at_target, penalty_score

    except Exception as e:
        print(f"Error in simulation: {e}")
        return float('inf'), float('inf')  # Return invalid values to indicate failure

def run_de_algorithm(solutions, max_iter):
    """
    Run the Differential Evolution optimization algorithm.
    """
    for generation in range(1, max_iter + 1):
        print(f"\n{'='*50}")
        print(f"GENERATION {generation}")
        print(f"{'='*50}")

        # Evaluate the fitness of the current population
        print("\nEvaluating Current Population:")
        function_values = {}
        for key, sol in solutions.items():
            params_formatted = [f"{v:.1f}" for v in sol]
            print(f"Solution x{key}: {params_formatted}")
            s11, penalty = objective_function(sol)
            function_values[key] = (s11, penalty)
            print(f"  -> S11 at 5.65 GHz: {s11:.1f} dB, Penalty: {penalty:.1f}")

        # Mutation, Crossover and Selection
        new_population = []     # To store new solutions for the next generation
        new_function_values = []    # To store their objective function values

        print(f"\nPerforming DE Operations:")
        print("-" * 30)

        # Iterate over each solution and perform mutation, crossover and selection
        for parent_index in solutions.keys():
            parent_vector = solutions[parent_index]
            parent_s11, parent_penalty = function_values[parent_index]

            print(f"\nProcessing Parent x{parent_index}: {[f'{v:.1f}' for v in parent_vector]} (S11: {parent_s11:.1f} dB)")

            # Step 1: Select vectors
            remaining_indices = list(solutions.keys())
            remaining_indices.remove(parent_index)  # Exclude the parent vector

            # Select the target vector (remove its index)
            target_index = random.choice(remaining_indices)
            remaining_indices.remove(target_index)
            target_vector = solutions[target_index]

            # Select two random vectors (remove their indices)
            rand_indices = random.sample(remaining_indices, 2)
            random_vector_1 = solutions[rand_indices[0]]
            random_vector_2 = solutions[rand_indices[1]]

            # Step 2: Mutation (Create trial vector)
            trial_vector = [
                target_vector[0] + F * (random_vector_1[0] - random_vector_2[0]),
                target_vector[1] + F * (random_vector_1[1] - random_vector_2[1]),
                target_vector[2] + F * (random_vector_1[2] - random_vector_2[2]),
                target_vector[3] + F * (random_vector_1[3] - random_vector_2[3])
            ]

            # Ensure trial vector is within bounds and round to 1 decimal place
            trial_vector = [
                round(np.clip(trial_vector[0], lower_bound[0], upper_bound[0]), 1),
                round(np.clip(trial_vector[1], lower_bound[1], upper_bound[1]), 1),
                round(np.clip(trial_vector[2], lower_bound[2], upper_bound[2]), 1),
                round(np.clip(trial_vector[3], lower_bound[3], upper_bound[3]), 1),
            ]

            #Crossover (Using a fixed point crossover strategy)
            crossover_points = set()
            fixed_point = random.randint(0, dim - 1)    # Randomly select one dimension for crossover
            crossover_points.add(fixed_point)
            for d in range(dim):
                if random.random() < Pcr and d != fixed_point:
                    crossover_points.add(d)

            # Perform crossover
            offspring = [
                round(trial_vector[d], 1) if d in crossover_points else round(parent_vector[d], 1)
                for d in range(dim)
            ]

            # Step 4: Selection (only consider S11 at target frequency)
            offspring_s11, offspring_penalty = objective_function(offspring)
            
            offspring_formatted = [f"{v:.1f}" for v in offspring]
            print(f"  Offspring: {offspring_formatted} (S11: {offspring_s11:.1f} dB)")
            
            if offspring_s11 < parent_s11:  # Only compare S11 values
                # If offspring is better, select it
                new_population.append(offspring)
                new_function_values.append((offspring_s11, offspring_penalty))
                print(f"  -> SELECTED: Offspring (better S11)")
            else: 
                # Otherwise, retain the parent
                new_population.append(parent_vector)
                new_function_values.append((parent_s11, parent_penalty))
                print(f"  -> SELECTED: Parent (better S11)")

        # Update solutions for next generation
        solutions = {i+1: new_population[i] for i in range(len(new_population))}

        # Output the generation summary
        print(f"\n{'='*30}")
        print(f"GENERATION {generation} SUMMARY")
        print(f"{'='*30}")
        for i, (sol, val) in enumerate(zip(new_population, new_function_values)):
            sol_formatted = [f"{v:.1f}" for v in sol]
            s11, penalty = val
            print(f"Solution x{i+1}: {sol_formatted} -> S11: {s11:.1f} dB")

    return solutions, new_function_values

# Run the Differential Evolution Algorithm for the specified number of generations
final_solutions, final_values = run_de_algorithm(solutions, max_iter)

# Find and display the optimal solution
print(f"\n{'='*50}")
print("OPTIMIZATION COMPLETE")
print(f"{'='*50}")

best_index = 0
best_s11 = float('inf')
for i, val in enumerate(final_values):
    s11, penalty = val
    if s11 < best_s11:
        best_s11 = s11
        best_index = i

optimal_params = final_solutions[best_index + 1]
optimal_params_formatted = [f"{v:.1f}" for v in optimal_params]

print(f"\nOPTIMAL SOLUTION:")
print(f"Parameters: {optimal_params_formatted}")
print(f"S11 at 5.65 GHz: {best_s11:.1f} dB")
print(f"[patch_length: {optimal_params[0]:.1f}, patch_width: {optimal_params[1]:.1f}, feed_pos_x: {optimal_params[2]:.1f}, feed_pos_y: {optimal_params[3]:.1f}]")

# Keep AEDT session open and close the script
print("\nHFSS project is active. Close it manually when done.")
input("Press Enter to close the Python script (HFSS will remain open)...")