import csv
import random
import os
import time
import numpy as np
from pyaedt import Hfss, Desktop

# Initialize AEDT Desktop
desktop = Desktop(non_graphical=False, new_desktop_session=True)

# Provide the full path of the project
project_path = r"D:\AI\array\coax_array.aedt"
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

# Extract parameter bounds for inter-element distance 'd'
d_min = 23.9  # Minimum distance (in mm) - rounded to 1 decimal
d_max = 31.9  # Maximum distance (in mm) - rounded to 1 decimal

# Define the bounds for 'd'
lower_bound = d_min
upper_bound = d_max

# DE Algorithm parameters
F = 0.5  # Scaling factor
Pcr = 0.7  # Crossover probability
dim = 1  # Only one parameter (d)
max_iter = 1  # Number of generations
population_size = 5  # Number of solutions in the population

# Generate 5 random solutions within the boundaries
solutions = {}
for i in range(1, 6):
    d_value = round(np.random.uniform(d_min, d_max), 1)
    solutions[i] = [d_value]

# Print the initial population
print("\n=== INITIAL POPULATION ===")
for i, sol in solutions.items():
    params_formatted = [f"{v:.1f}" for v in sol]
    print(f"Solution x{i}: {params_formatted}")

def objective_function(params):
    """
    Objective function that minimizes S11 and maximizes gain at 5.65 GHz,
    and prints S11, gain, the penalty, and the inter-element distance `d`.
    Returns tuple (s11_at_target, penalty_score) for consistency with coax code.
    """
    d = params[0]
    target_freq = 5.65  # GHz
    freq_window = 0.1   # ±100 MHz

    # Weights for S11 and gain in the objective
    w1 = 1.0  # Weight for S11
    w2 = 2.0  # Weight for gain (higher importance than S11)

    try:
        # Set inter-element spacing in HFSS (convert mm to cm)
        hfss.variable_manager.set_variable('dis', f"{d / 10:.1f}cm")

        # Run simulation
        hfss.analyze_setup(setup_name)

        # Wait for analysis completion and extract S11
        while True:
            try:
                s_params = hfss.post.get_solution_data(
                    expressions=["dB(S(1,1))"],
                    primary_sweep_variable="Freq"
                )
                break
            except Exception:
                time.sleep(5)

        # Extract frequency sweep and S-parameter data
        freqs = np.array(s_params.primary_sweep_values)
        s11_vals = np.array(s_params.data_real("dB(S(1,1))"))

        # Find closest index to 5.65 GHz
        idx_target = np.argmin(np.abs(freqs - target_freq))
        s11_at_target = s11_vals[idx_target]

        # Retrieve gain at broadside (theta=0, phi=0)
        try:
            ff_setup_name = "Infinite Sphere1"  # Replace if your far-field name differs

            gain_data = hfss.post.get_solution_data(
                expressions=["dB(GainTotal)"],
                setup_sweep_name=setup_name + " : LastAdaptive",
                variations={"Phi": "0deg"},
                primary_sweep_variable="Theta",
                context=ff_setup_name,
                report_category="Far Fields"
            )

            # Extract Theta and Gain values
            # theta_vals = [float(val.replace("deg", "")) for val in gain_data.primary_sweep_values]
            theta_vals = np.array(gain_data.primary_sweep_values,dtype = float)
            gain_vals = gain_data.data_magnitude("dB(GainTotal)")

            # Find the gain at Theta = 0 degrees
            theta_zero_index = np.where(theta_vals == 0)[0]
            if theta_zero_index.size > 0:
                gain_at_theta_zero = gain_vals[theta_zero_index[0]]
            #    print(f"Gain at Theta = 0°: {gain_at_theta_zero:.2f} dB")
            else:
                print("Theta = 0° not found in the data.")
                gain_at_theta_zero = -999  # Default value if not found

        except Exception as e:
            print(f"Error extracting gain: {e}")
            gain_at_theta_zero = -999  # Default value if extraction fails

        # Find the minimum S11 in the entire frequency range
        min_s11_overall = np.min(s11_vals)

        # Initialize penalty score
        penalty_score = 0

        # Calculate penalty if better response exists away from target frequency
        if min_s11_overall < s11_at_target:
            # Find where the minimum occurs
            min_freq_idx = np.argmin(s11_vals)
            min_freq = freqs[min_freq_idx]
           
            # Calculate frequency deviation from target
            freq_deviation = abs(min_freq - target_freq)
           
            # Apply penalties
            frequency_penalty = 10 * np.exp(freq_deviation / 0.1) - 10  # Exponential penalty
            magnitude_penalty = (5 * (s11_at_target - min_s11_overall)) ** 2  # Magnitude penalty
            window_penalty = 10 if freq_deviation > freq_window else 0  # Penalty for deviation beyond window
            local_minima_penalty = 15 if np.sum(s11_vals < s11_at_target) > 1 else 0  # Multiple minima penalty

            # Sum all penalties
            penalty_score = frequency_penalty + magnitude_penalty + window_penalty + local_minima_penalty

        # Print values for debugging - ALWAYS print S11 and gain
        print(f"Distance (d): {d:.1f} mm")
        print(f"S11 at 5.65 GHz: {s11_at_target:.1f} dB")
        print(f"Gain at θ=0°: {gain_at_theta_zero:.1f} dB")
        print(f"Penalty Score: {penalty_score:.1f}")
        print("-" * 50)

        # Calculate weighted objective (minimize S11, maximize gain)
        # For gain, we use negative value since we want to maximize it
        weighted_objective = w1 * abs(s11_at_target) + w2 * (-gain_at_theta_zero) + penalty_score

        # Return S11 at target frequency and penalty score (like coax code)
        return s11_at_target, penalty_score

    except Exception as e:
        print(f"Error in simulation: {e}")
        print(f"Distance (d): {d:.1f} mm")
        print(f"S11 at 5.65 GHz: ERROR")
        print(f"Gain at θ=0°: ERROR")
        print(f"Penalty Score: ERROR")
        print("-" * 50)
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
                target_vector[0] + F * (random_vector_1[0] - random_vector_2[0])
            ]

            # Ensure trial vector is within bounds and round to 1 decimal place
            trial_vector = [
                round(np.clip(trial_vector[0], lower_bound, upper_bound), 1)
            ]

            # Crossover (Using a fixed point crossover strategy)
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
            print(f"Evaluating offspring {[f'{v:.1f}' for v in offspring]}:")
            offspring_s11, offspring_penalty = objective_function(offspring)
           
            if offspring_s11 < parent_s11:  # Only compare S11 values
                # If offspring is better, select it
                new_population.append(offspring)
                new_function_values.append((offspring_s11, offspring_penalty))
                print(f"  -> SELECTED: Offspring (better S11: {offspring_s11:.1f} dB)")
            else:
                # Otherwise, retain the parent
                new_population.append(parent_vector)
                new_function_values.append((parent_s11, parent_penalty))
                print(f"  -> SELECTED: Parent (better S11: {parent_s11:.1f} dB)")

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
print(f"[inter_element_distance: {optimal_params[0]:.1f}]")

# Keep AEDT session open and close the script
print("\nHFSS project is active. Close it manually when done.")
input("Press Enter to close the Python script (HFSS will remain open)...")