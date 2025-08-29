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
    
    players_list = [Player(win_value=0.5, blind=False, level=100, name=f"Player_{i}") for i in range(2)]
    
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
    
    total_n = sum(c[3] for c in category_tuple)
    if total_n == 0:
        return 0.0

    x = torch.tensor([[c[1], c[2], c[3]] for c in category_tuple], dtype=torch.float32).unsqueeze(0)
    k_tensor = torch.tensor([[game.top_k / total_n if total_n > 0 else 0]], dtype=torch.float32)
    
    with torch.no_grad():
        return model(x, k_tensor).item()

def evaluate_strategy_with_simulation(game, player, strategy, num_simulations=5000):
    """Evaluate a strategy using a robust Monte Carlo simulation."""
    logging.info(f"Running {num_simulations} simulations for strategy: {strategy}...")
    
    original_strategy = player.strategy.copy()
    player.strategy = strategy
    
    total_valuations = []
    for _ in range(num_simulations):
        current_sim_valuation = 0.0
        all_samples_for_sim = []
        
        for category_name, allocation_pct in player.strategy.items():
            category = game.categories[category_name]
            num_samples_for_category = int(np.round(allocation_pct * category.size))
            
            if num_samples_for_category > 0:
                samples = category.get_samples([num_samples_for_category])[0]
                all_samples_for_sim.extend(samples)
                
        if game.game_mode_type == "top_k" and game.top_k is not None and len(all_samples_for_sim) > 0:
            top_k_values = np.sort(all_samples_for_sim)[-min(game.top_k, len(all_samples_for_sim)):]
            current_sim_valuation = np.sum(top_k_values)
        else:
            current_sim_valuation = np.sum(all_samples_for_sim)
        
        total_valuations.append(current_sim_valuation)

    player.strategy = original_strategy
    final_mean_valuation = np.mean(total_valuations)
    logging.info(f"Mean valuation from simulation: {final_mean_valuation:.4f}")
    return final_mean_valuation

def test_model_accuracy_on_fixed_strategy():
    """
    Tests the accuracy of the deep learning model on a single, fixed strategy
    by comparing its prediction to a high-fidelity Monte Carlo simulation.
    """
    logging.info("--- Starting Model Accuracy Test ---")
    
    # 1. Load Game State
    game, player = load_game_from_row(row_index=0)
    logging.info(f"Game loaded with top_k = {game.top_k} and to_admit = {game.to_admit}")

    # 2. Define a Fixed Strategy
    fixed_strategy = {'Q1': 0.7, 'Q2': 0.2, 'Q3': 0.1, 'Q4': 0.0}
    logging.info(f"Using fixed strategy: {fixed_strategy}")

    # 3. Get Model's Prediction
    model = torch.jit.load("runs/topk_experiment/best_model_biased.pt.jit")
    model.eval()
    predicted_value = evaluate_strategy_with_model(game, player, fixed_strategy, model)
    logging.info(f"Model Predicted Value: {predicted_value:.4f}")

    # 4. Get "Exact" Value via Simulation
    simulated_value = evaluate_strategy_with_simulation(game, player, fixed_strategy, num_simulations=20000)

    # 5. Compare and Report
    error_pct = abs(predicted_value - simulated_value) / simulated_value * 100 if simulated_value != 0 else 0
    
    print("\n" + "="*50)
    print("      Model Prediction vs. Simulation      ")
    print("="*50)
    print(f"Fixed Strategy:       {fixed_strategy}")
    print(f"Model Prediction:     {predicted_value:.4f}")
    print(f"Simulated 'Exact' Value: {simulated_value:.4f}")
    print("-" * 50)
    print(f"Percentage Error:     {error_pct:.2f}%")
    print("="*50)
    
    # Assert that the error is within a reasonable threshold, e.g., 10%
    assert error_pct < 10, "Model prediction error is too high!"
    logging.info("--- Test Completed Successfully ---")

if __name__ == "__main__":
    test_model_accuracy_on_fixed_strategy() 