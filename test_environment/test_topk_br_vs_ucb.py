import pandas as pd
import numpy as np
import torch
import time
import logging
from objects_gpu import Game, Player, Category

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_game_from_row(row_index=0):
    """Load a game configuration from a specific row in the params file."""
    params_df = pd.read_csv("test_environment/params_file_1002_occupancy0.99_modetop_k.csv")
    row = params_df.iloc[row_index]
    
    base_mu = 5
    base_sigma = 5 / row['mean_variance_ratio']
    base_candidates = 120
    
    players_list = [
        Player(
            win_value=row['win_value_underdog'] if i == 0 else 1-row['win_value_underdog'],
            blind=row[f'blind_combo_{i}'],
            level=row[f'level_{i}'],
            name=f"Player_{i}"
        ) for i in range(2)
    ]
    
    categories_dict = {}
    for i in range(4):
        category_name = f"Q{i+1}"
        mu = base_mu * row['high_low_ratio_mean'] if i in [0, 1] else base_mu
        sigma = base_sigma * row['high_low_ratio_variance'] if i in [0, 2] else base_sigma
        
        if i == 0:
            size = int(base_candidates * row['pct_high_mean'] * row['pct_high_sigma'])
        elif i == 1:
            size = int(base_candidates * row['pct_high_mean'] * (1 - row['pct_high_sigma']))
        elif i == 2:
            size = int(base_candidates * (1 - row['pct_high_mean']) * row['pct_high_sigma'])
        else:
            size = int(base_candidates * (1 - row['pct_high_mean']) * (1 - row['pct_high_sigma']))
        
        categories_dict[category_name] = Category(
            name=category_name, mu=mu, sigma=sigma, size=size,
            log_or_normal="log" if row['lognormal'] == 'log' else "normal"
        )
    
    total_size = sum(category.size for category in categories_dict.values())
    to_admit = min(total_size//2, int((row['pct_high_mean'] * row['pct_total'] * total_size) // 2))
    top_k = max(1, int(to_admit * 0.2))
    
    return Game(
        num_players=len(players_list), to_admit=to_admit, players=players_list,
        categories=categories_dict, game_mode_type=row['game_mode'], top_k=top_k,
        log_normal=row['lognormal'], verbose=False
    ), players_list[0]

def evaluate_strategy_with_model(game, player, strategy, model):
    """Evaluate a strategy using the deep learning model."""
    category_tuple = player.convert_category_strategy_to_evaluator(game, strategy)
    
    # Handle case where a strategy results in zero allocation
    total_n = sum(c[3] for c in category_tuple)
    if total_n == 0:
        return -1e9 # Return a very low score for invalid allocations

    x = torch.tensor([[c[1], c[2], c[3]] for c in category_tuple], dtype=torch.float32).unsqueeze(0)
    k = torch.tensor([[game.top_k / total_n]], dtype=torch.float32)
    
    with torch.no_grad():
        return model(x, k).item()

def evaluate_strategy_with_simulation(game, player, strategy, initial_batch_size=100, confidence_level=0.95, relative_margin_of_error=0.01):
    """
    Evaluates a given strategy using Monte Carlo simulation with dynamic batch sizing.

    The function first runs an initial batch of simulations to estimate the mean and
    variance of the total valuation. Based on these estimates and the desired
    confidence level and margin of error, it calculates the statistically required
    number of simulations. Additional simulations are then performed if needed.

    Args:
        game (Game): The game instance.
        player (Player): The player for whom the strategy is being evaluated.
        strategy (dict): The strategy to evaluate, a dictionary of category allocations.
        initial_batch_size (int): The number of simulations to run initially to estimate variance.
        confidence_level (float): The desired confidence level for the mean estimate (e.g., 0.95 for 95%).
        relative_margin_of_error (float): The desired relative margin of error for the mean estimate (e.g., 0.01 for 1%).

    Returns:
        float: The average total valuation from all simulations for the given strategy.
    """
    logging.info(f"Evaluating strategy: {strategy} with initial batch size {initial_batch_size}.")

    original_strategy = player.strategy.copy()
    player.strategy = strategy

    total_valuations = []

    # --- Phase 1: Initial Batch Simulation ---
    logging.info(f"Running initial batch of {initial_batch_size} simulations...")
    initial_batch_valuations = _run_simulations(game, player, initial_batch_size)
    total_valuations.extend(initial_batch_valuations)

    mean_valuation = np.mean(total_valuations)
    std_valuation = np.std(total_valuations)
    logging.info(f"Initial batch mean valuation: {mean_valuation:.4f}, Std Dev: {std_valuation:.4f}")

    # Avoid division by zero or very small mean if all valuations are zero or near zero
    if mean_valuation <= 1e-9: # if mean is effectively zero, cannot calculate relative error
        logging.warning("Mean valuation is very close to zero, cannot calculate relative margin of error. Returning initial mean.")
        player.strategy = original_strategy
        return mean_valuation
        
    # --- Phase 2: Dynamic Sample Size Calculation ---
    # Z-score for the given confidence level (e.g., 1.96 for 95% confidence)
    # Using a common Z-table value; for more precision, could use scipy.stats.norm.ppf
    if confidence_level == 0.95:
        z_score = 1.96
    elif confidence_level == 0.99:
        z_score = 2.576
    else:
        logging.warning(f"Unsupported confidence level: {confidence_level}. Defaulting to Z-score for 95% confidence (1.96).")
        z_score = 1.96

    # Calculate absolute margin of error based on relative margin and current mean
    desired_absolute_error = relative_margin_of_error * mean_valuation
    
    if desired_absolute_error <= 0 and std_valuation > 0:
        logging.warning("Desired absolute error is zero or negative with non-zero standard deviation. This may lead to infinite sample size. Adjusting to a small positive error.")
        desired_absolute_error = 1e-6 # Small positive value to prevent division by zero

    # Calculate required number of simulations for the mean estimate
    # Formula: n = (Z * std / E)^2
    if std_valuation > 0:
        required_simulations = int(np.ceil((z_score * std_valuation / desired_absolute_error) ** 2))
    else:
        required_simulations = initial_batch_size # If no variance, initial batch is sufficient for a point estimate

    logging.info(f"Required simulations for {confidence_level*100}% confidence and {relative_margin_of_error*100}% relative error: {required_simulations}")

    # --- Phase 3: Perform Remaining Simulations ---
    if required_simulations > len(total_valuations):
        num_additional_sims = required_simulations - len(total_valuations)
        logging.info(f"Running {num_additional_sims} additional simulations...")
        additional_valuations = _run_simulations(game, player, num_additional_sims)
        total_valuations.extend(additional_valuations)
    else:
        logging.info("Initial batch sufficient. No additional simulations needed.")

    final_mean_valuation = np.mean(total_valuations)
    logging.info(f"Final average total valuation after {len(total_valuations)} simulations: {final_mean_valuation:.4f}")

    player.strategy = original_strategy
    return final_mean_valuation

def _run_simulations(game, player, num_simulations):
    """
    Helper function to run a batch of Monte Carlo simulations for a given strategy.

    Args:
        game (Game): The game instance.
        player (Player): The player for whom the strategy is being evaluated.
        num_simulations (int): The number of simulations to run in this batch.

    Returns:
        list: A list of total valuations for each simulation.
    """
    batch_valuations = []
    
    # Pre-calculate allocated sizes for all categories and all simulations in the batch
    # This creates a list of lists, where inner list contains sizes for each category
    # for a single simulation
    
    # This section gets complex with varied sizes and top_k, so let's simplify for now
    # by generating samples per category and then processing them individually per sim.
    # For full vectorization, we'd need category.get_samples to return 2D array (batch_size, num_samples_per_category)

    # Instead of making get_samples return 2D, we make the loop over simulations outside
    # of the category sampling for clarity and manageability given existing structure.
    # If performance becomes an issue, we can optimize `get_samples` to return 2D.
    
    # Let's adjust category.get_samples to take a single int and return a 1D array.
    # No, that's not what it does. It takes a list of sizes, and returns a list of arrays.
    # It already handles batching in a way, but per category.

    # We need to reshape `sizes` to be an array of `num_simulations`
    # e.g., if strat for Q1 is 0.5 and size is 100, then num_samples is 50.
    # So we need `num_simulations` of 50 samples from Q1.
    
    # A list of lists, where inner list contains:
    # [ (category_object, num_samples_for_sim_1), (category_object, num_samples_for_sim_2), ... ]
    
    # This is still not truly vectorized across simulations unless get_samples supports it.
    # Given the current get_samples: `def get_samples(self, sizes:list[int]):`
    # It seems to be designed to return a list of numpy arrays, where each array
    # corresponds to a requested size.
    # So, we can pass `[num_samples_for_this_category_for_each_sim] * num_simulations`
    # and it would return `num_simulations` arrays.

    for sim_idx in range(num_simulations):
        current_sim_valuation = 0.0
        all_samples_for_sim = [] # To collect all samples for top_k calculation across categories
        
        for category_name, allocation_pct in player.strategy.items():
            category = game.categories[category_name]
            
            if np.isnan(allocation_pct):
                num_samples_for_category = 0
            else:
                num_samples_for_category = int(np.round(allocation_pct * category.size))
            
            if num_samples_for_category > 0:
                # category.get_samples([num_samples]) returns a list containing one numpy array
                samples = category.get_samples([num_samples_for_category])[0]
                all_samples_for_sim.extend(samples)
                
        # After collecting all samples for the current simulation from all categories
        if game.game_mode_type == "top_k" and game.top_k is not None:
            if len(all_samples_for_sim) > 0:
                # Ensure we have enough samples before trying to get top_k
                # Take min(game.top_k, len(all_samples_for_sim)) to avoid index error
                top_k_values = np.sort(all_samples_for_sim)[-min(game.top_k, len(all_samples_for_sim)):]
                current_sim_valuation = np.sum(top_k_values)
            else:
                current_sim_valuation = 0.0 # No samples, so valuation is 0
        else:
            current_sim_valuation = np.sum(all_samples_for_sim)
        
        batch_valuations.append(current_sim_valuation)

    return batch_valuations

def deep_learning_guided_ucb(game, player, model, top_n=10, ucb_iterations=50):
    """Use DL model to find top strategies, then UCB to find the best."""
    print("Phase 1: Generating and evaluating a grid of strategies with the DL model...")
    start_time = time.time()

    candidate_strategies = []
    step = 0.1
    for p1_steps in range(int(1/step) + 1):
        p1 = p1_steps * step
        for p2_steps in range(int(1/step) - p1_steps + 1):
            p2 = p2_steps * step
            p3 = 1.0 - p1 - p2
            if p3 < -1e-9: continue
            p3 = max(0, p3)
            candidate_strategies.append({'Q1': p1, 'Q2': p2, 'Q3': p3, 'Q4': 0.0})

    print(f"Generated {len(candidate_strategies)} candidate strategies.")

    evaluated_strategies = [(s, evaluate_strategy_with_model(game, player, s, model)) for s in candidate_strategies]
    evaluated_strategies.sort(key=lambda x: x[1], reverse=True)
    top_strategies = evaluated_strategies[:top_n]

    print(f"Phase 1 finished in {time.time() - start_time:.2f}s. Top {len(top_strategies)} strategies found.")

    print(f"\nPhase 2: Using UCB to find best among top {top_n} strategies via simulation...")
    def ucb_objective(strategy_index):
        strategy, _ = top_strategies[int(round(strategy_index))]
        return evaluate_strategy_with_simulation(game, player, strategy)

    ucb_optimizer = BayesianOptimization(f=ucb_objective, pbounds={'strategy_index': (0, len(top_strategies) - 1)}, random_state=42, verbose=False)
    ucb_optimizer.maximize(init_points=min(len(top_strategies), 5), n_iter=ucb_iterations)
    
    best_strategy, _ = top_strategies[int(round(ucb_optimizer.max['params']['strategy_index']))]
    return best_strategy, ucb_optimizer.max['target'], len(ucb_optimizer.res)

def pure_ucb_sampling(game, player, total_iterations=150):
    """Use pure UCB sampling across the entire strategy domain."""
    print("Using pure UCB sampling across entire strategy domain...")
    
    def objective(**kwargs):
        total = sum(kwargs.values())
        if total == 0: return -1e9
        strategy = {key: value / total for key, value in kwargs.items()}
        return evaluate_strategy_with_simulation(game, player, strategy)

    optimizer = BayesianOptimization(f=objective, pbounds={f'Q{i+1}': (0.0, 1.0) for i in range(4)}, random_state=42, verbose=False)
    optimizer.maximize(init_points=20, n_iter=total_iterations)
    
    best_params = optimizer.max['params']
    total = sum(best_params.values())
    best_strategy = {key: value/total for key, value in best_params.items()}
    return best_strategy, optimizer.max['target'], len(optimizer.res)

def test_convergence_and_speed():
    """Test that both methods converge to similar results and compare speed."""
    game, player = load_game_from_row(row_index=0)
    model = torch.jit.load("runs/topk_experiment/best_model_biased.pt.jit")
    model.eval()

    print("\n" + "="*60 + "\nTESTING DEEP LEARNING GUIDED UCB\n" + "="*60)
    start_time = time.time()
    dl_strategy, dl_score, dl_iterations = deep_learning_guided_ucb(game, player, model)
    dl_time = time.time() - start_time
    
    print(f"\nDL Guided Results:\n  Best strategy: {dl_strategy}\n  Best score: {dl_score:.4f}\n  Time: {dl_time:.2f}s")

    print("\n" + "="*60 + "\nTESTING PURE UCB SAMPLING\n" + "="*60)
    start_time = time.time()
    pure_strategy, pure_score, pure_iterations = pure_ucb_sampling(game, player)
    pure_time = time.time() - start_time
    
    print(f"\nPure UCB Results:\n  Best strategy: {pure_strategy}\n  Best score: {pure_score:.4f}\n  Time: {pure_time:.2f}s")
    
    score_diff = abs(dl_score - pure_score)
    time_speedup = pure_time / dl_time if dl_time > 0 else float('inf')

    print("\n" + "="*60 + "\nCOMPARISON RESULTS\n" + "="*60)
    print(f"Score difference: {score_diff:.4f} -> {'PASSED' if score_diff < 0.1 * abs(pure_score) else 'FAILED'}")
    print(f"Speedup factor: {time_speedup:.2f}x -> {'PASSED' if time_speedup > 1.5 else 'FAILED'}")

if __name__ == "__main__":
    test_convergence_and_speed()
