from scipy.stats import norm
import numpy as np
import warnings
import time as t
import logging
from bayes_opt import BayesianOptimization
import torch

# Configure logging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

def calculate_replicates_for_top_k(
    params,          # List of (dist_type, mu, sigma, sample_size) tuples
    k,               # Number of top-k values to consider
    confidence=0.95,      # Desired confidence level (e.g., 0.95 for 95% confidence)
    error_rate=0.01,      # Allowed relative error (e.g., 0.01 for 1% error)
    max_iters=50,    # Max iterations for convergence
    tol=1e-3         # Tolerance for replicate change
):
    """
    Calculate the number of replicates required for estimating the mean of the top-k
    values from a fixed-size combined sample of distributions.

    Parameters:
        params (list of tuples): Each tuple (dist_type, mu, sigma, n) specifies:
                                 - dist_type: 'normal' or 'lognormal'
                                 - mu: mean of the distribution
                                 - sigma: standard deviation of the distribution
                                 - n: number of samples to draw from the distribution
        k (int): Number of top-k values to estimate.
        confidence (float): Desired confidence level (e.g., 0.95).
        error_rate (float): Allowed relative error in the top-k mean estimate.
        max_iters (int): Maximum number of iterations to converge.
        tol (float): Tolerance for replicate count change between iterations.

    Returns:
        int: Estimated required number of replicates.
    """
    # Calculate z-score for the confidence level
    z = norm.ppf(1 - (1 - confidence) / 2)

    # Initial guess for number of replicates
    num_replicates = 100

    # Total combined sample size from all distributions
    combined_sample_size = sum(n for _, _, _, n in params)
    k = int(min(combined_sample_size, k))

    for _ in range(max_iters):
        # Generate the combined sample based on specified sample sizes and distribution types
        combined_data = np.concatenate([
            np.random.normal(mu, sigma, int(n)) if dist_type == 'normal' else
            np.random.lognormal(mu, sigma, int(n))
            for dist_type, mu, sigma, n in params
        ])

        # Extract top-k values
        top_k_values = np.sort(combined_data)[-k:]
        top_k_mean = np.mean(top_k_values)
        top_k_std = np.std(top_k_values, ddof=1) if len(top_k_values) > 1 else np.std(top_k_values)

        # Estimate the required number of replicates
        required_replicates = max(1, int(((z * top_k_std) / (error_rate * top_k_mean))**2))

        # Convergence check
        if abs(required_replicates - num_replicates) / num_replicates < tol:
            return required_replicates

        # Update replicate count for the next iteration
        num_replicates = required_replicates

    # If convergence is not achieved, return the last estimate
    return num_replicates

def transform_mu_sigma_to_log(mu, sigma):
    sigma_log = np.log(sigma) / 2
    mu_log = np.log(mu) / 2
    return mu_log, sigma_log

def generate_samples_top_k(categories, top_k=None):
    """
    Generate samples for the top-k computation using GPU acceleration.

    Parameters:
        categories (list of tuples): List of (dist_type, mu, sigma, n) for each category.
        top_k (int): Optional, number of top values to return.

    Returns:
        np.ndarray: Array of sampled outcomes (sorted if top_k is specified).
    """
    outcomes = np.concatenate([
        np.random.lognormal(mu, sigma, int(n)) if dist_type == 'log' else
        np.random.normal(mu, sigma, int(n))
        for dist_type, mu, sigma, n in categories
    ])

    if top_k is not None:
        outcomes = np.sort(outcomes)[-top_k:]
    return outcomes



class Category:
    def __init__(self, name, mu, sigma, size, log_or_normal):
        self.name = name
        self.size = size
        self.log_normal = log_or_normal
        self.mu, self.sigma = (
            transform_mu_sigma_to_log(mu, sigma) if log_or_normal == 'log' else (mu, sigma)
        )
        self.mean = np.exp((self.mu + self.sigma ** 2) / 2) if log_or_normal == 'log' else self.mu
        self.std = (np.exp(2*self.mu)*(np.exp(self.sigma**2)-1))**(1/2) if log_or_normal == 'log' else self.sigma

    def get_samples(self, sizes):
        #print(self.name, sizes)
        if np.any(sizes > self.size):
            raise ValueError("Cannot sample more than available elements.")
        samples = []
        for n in sizes:
            if self.log_normal == 'log':
                # Generate lognormal samples and clamp to >= 0
                sample = np.random.lognormal(mean=self.mu, sigma=self.sigma, size=int(n))
            else:
                # Generate normal samples and clamp to >= 0
                sample = np.random.normal(loc=self.mean, scale=self.std, size=int(n))
            # Ensure all samples are >= 0
            sample = np.maximum(sample, 0)
            samples.append(sample)
        return samples


class Player:
    def __init__(self, win_value: float, blind: bool, level: int, name: str):
        """
        Initializes a Player object.

        Parameters:
            win_value (float): Value used to calculate probability of victory in collisions.
            blind (bool): Whether the player is blind to detailed strategy optimizations.
            level (int): Level of the player (0 for basic, higher for iterated responses).
            name (str): Unique identifier for the player.
        """
        self.strategy = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
        self.blind_strategy = {"high": 0, "low": 0}
        self.win_value = win_value
        self.blind = blind
        self.level = level
        self.name = name

    ...
    def convert_category_strategy_to_evaluator(self, game, temp_strategy):
        """
        Converts the player's category-level strategy into a tuple representation for evaluation.
        
        Parameters:
            game (Game): The game instance containing category details.
            
        Returns:
            tuple: A tuple of (distribution type, mean, std, size) for each category.
        """
        return tuple(
            (
                "log" if category.log_normal else "normal",
                category.mu,
                category.sigma,
                int(category.size * temp_strategy[category_name])
            )
            for category_name, category in game.categories.items()
        )

    def update_strategy(self, game):
        """
        Update player strategy based on blind strategy proportions.

        Parameters:
            blind_strategy (dict): Blind strategy with high and low group allocations.
            game (Game): The current game object.
        """

        self.strategy = {category_name: np.round(self.blind_strategy[group] * game.categories[category_name].size) / game.categories[category_name].size  for category_name, group in zip(["Q1", "Q2", "Q3", "Q4"], ["high", "high", "low", "low"])}

    def calculate_win_chance(self, players):
        """
        Calculate this player's win chance based on relative win values.

        Parameters:
            players (list[Player]): List of all players in the game.

        Returns:
            float: Win probability for this player.
        """
        total_win_value = sum(player.win_value for player in players)
        return self.win_value / total_win_value

    def calc_expected_attendees(self, strategy, game):
        """
        Calculate expected attendees for the player under a given strategy. (This is only for the case where there's only two players)

        Parameters:
            strategy (dict): Strategy dictionary specifying category allocations.
            game (Game): The game object with categories and players.

        Returns:
            dict: Expected attendees per category.
        """
        other_player = [player for player in game.players if player != self][0]
        return {category_name: strategy[category_name] * (1 - other_player.strategy[category_name]*other_player.win_value/(other_player.win_value + self.win_value)) for category_name, category in game.categories.items()}

    def greedy_top_k_br(self, game, feasible_strategy_numbers, blind=False, verbose=False):
        """
        Perform Bayesian optimization for top-k allocation using the trained model.
        """
        # Load the trained model
        model_path = "runs/topk_experiment/best_model_biased.pt.jit"
        model = torch.jit.load(model_path)
        model.eval()
        
        # Define the search space bounds for each category
        pbounds = {
            f'Q{i+1}': (0.0, 1.0) for i in range(4)  # Each category can take values 0-1
        }
        
        # Define the objective function for Bayesian optimization
        def objective(**kwargs):
            # Convert kwargs to strategy dictionary
            strategy = {f'Q{i+1}': kwargs[f'Q{i+1}'] for i in range(4)}
            
            # Convert to evaluator format
            category_tuple = self.convert_category_strategy_to_evaluator(game, strategy)
            
            # Prepare input for the model
            x = torch.tensor([
                [category[1], category[2], category[3]]  # mu, sigma, n
                for category in category_tuple
            ], dtype=torch.float32).unsqueeze(0)  # Add batch dimension
            
            k = torch.tensor([[game.top_k / sum(category[3] for category in category_tuple)]], 
                            dtype=torch.float32)
            
            # Get prediction
            with torch.no_grad():
                pred = model(x, k)
            
            return pred.item()
        
        # Set up Bayesian optimizer
        optimizer = BayesianOptimization(
            f=objective,
            pbounds=pbounds,
            random_state=42,
            verbose=verbose
        )
        
        # Add constraint that sum of allocations <= to_admit
        def constraint(**params):
            total = sum(params.values())
            return min(1.0, game.to_admit / sum(feasible_strategy_numbers.values())) - total
        
        optimizer.set_gp_params(alpha=1e-3)  # Adjust noise parameter
        optimizer.maximize(
            init_points=10,
            n_iter=50,
            kappa=2.576,  # Exploration parameter
        )
        
        # Get the best strategy found
        best_strategy = {k: v for k, v in optimizer.max['params'].items()}
        
        # Normalize to ensure feasibility
        total = sum(best_strategy.values())
        if total > 1.0:
            best_strategy = {k: v/total for k, v in best_strategy.items()}
        
        self.strategy = best_strategy

    def best_response(self, game, verbose=False):
        """
        Compute the best response for the player in the current game state.

        Parameters:
            game (Game): The game object.
        """
        #print(game.game_mode_type)
        max_strategy = {key: 1 for key in self.strategy.keys()}
        feasible_strategies = self.calc_expected_attendees(max_strategy, game)
        if verbose:
            print(self.name, feasible_strategies)
        feasible_numbers = {
            key: feasible_strategies[key] * game.categories[key].size
            for key in feasible_strategies
        }
        if verbose:
            print("Prints player name and the \"achievable\" numbers from maxing out each category",self.name, feasible_numbers)
        if game.game_mode_type == "top_k":
            self.greedy_top_k_br(game, feasible_numbers, self.blind, verbose=False)
        elif game.game_mode_type == "expected":
            to_admit = game.to_admit
            new_strategy = {key: 0 for key in self.strategy.keys()}

            for category_name in sorted(self.strategy.keys(), key=lambda x: -game.categories[x].mean):
                #print(category_name, to_admit)
                feasible = feasible_numbers[category_name]
                if to_admit > feasible:
                    new_strategy[category_name] = 1
                    to_admit -= feasible
                else:
                    new_strategy[category_name] = to_admit / feasible
                    to_admit = 0
                    break

            self.strategy = new_strategy

class Game:
    def __init__(self, num_players, to_admit, players, categories, game_mode_type, top_k=None, log_normal=False, verbose=False):
        """
        Initialize a Game object.

        Parameters:
            num_players (int): Number of players in the game.
            to_admit (int): Number of students each player aims to admit.
            players (list[Player]): List of player objects.
            categories (dict): Dictionary of category objects.
            game_mode_type (str): Game type (e.g., "top_k" or "expected").
            top_k (int, optional): Number of top values to consider (for "top_k" mode).
            log_normal (bool, optional): Whether distributions are log-normal.
            verbose (bool, optional): Verbosity of debug output.
        """
        self.num_players = num_players
        self.to_admit = to_admit
        self.players = players
        self.categories = categories
        self.game_mode_type = game_mode_type
        self.top_k = top_k
        self.log_normal = log_normal
        self.verbose = verbose

        self.memo = {}

    def find_strategies_iterated_br(self, verbose=False):
        """
        Find Nash equilibrium by iterated best responses.

        Each player updates their strategy until a stable profile is found.
        """
        previous_strategies = None
        level = 0
        while previous_strategies != [player.strategy for player in self.players]:
            previous_strategies = [player.strategy.copy() for player in self.players]
            for player in self.players:
                if player.level >= level:
                    player.best_response(self, verbose)
            level += 1

    def eval_particular_distribution(self, categories, verbose=False):
        """
        Evaluate the expected value for a particular distribution using GPU.

        Parameters:
            categories (list of tuples): List of (n, mu, sigma) for each category.
            dist_type (str): Either 'normal' or 'lognormal'.
            memo (dict): Cache to avoid redundant calculations.
            top_k (int): Optional, number of top values to consider.
            verbose (bool): If True, prints progress.

        Returns:
            float: Estimated expected value.
        """
        # Check the memo cache
        key = (categories, self.top_k)
        if key in self.memo:
            return self.memo[key]


        replicates = calculate_replicates_for_top_k(categories, self.top_k)
        # Estimate the number of replicates
        if categories[0][0] == 'log':
            num_replicates = max(50, min(replicates, 50000))
        else:
            num_replicates = max(10, min(replicates, 10000))  # Fixed range for normal distributions

        outcome = np.mean(np.stack([
            np.mean(generate_samples_top_k(categories, top_k=self.top_k))
            for _ in range(num_replicates)
        ]))

        # Store the result in memo and return the mean outcome
        self.memo[key] = outcome
        return self.memo[key]



    def calculate_utilities(self, attendees):
        """
        Calculate utilities for each player based on attendee outcomes.

        Parameters:
            attendees (dict): Mapping of player names to their attendees' values.

        Returns:
            dict: Utilities and additional stats for each player.
        """
        results = {}
        for player in self.players:
            admitted_values = np.array(attendees[player.name])
            if self.game_mode_type == "top_k":
                top_k_values = np.sort(admitted_values)[-self.top_k:]
                utility = np.sum(top_k_values) - abs(len(admitted_values) - self.to_admit)
            else:
                utility = np.sum(admitted_values) - abs(len(admitted_values) - self.to_admit)

            results[player.name] = {
                "utility": utility.get(),
                "admitted": len(admitted_values),
                "top_k": top_k_values.get().tolist() if self.game_mode_type == "top_k" else []
            }

        total_utility = sum(res["utility"] for res in results.values())
        for res in results.values():
            res["pct_total_util"] = res["utility"] / total_utility

        return results

    def simulate_game_batch(self, batch_size):
        """
        Simulate multiple games based on player strategies and resolve outcomes.

        Parameters:
            batch_size (int): Number of games to simulate in parallel.

        Returns:
            np.ndarray: Array of 'pct_total_util' for each player in each game.
        """
        # Initialize an array to store results for each game in the batch
        batch_results = np.zeros((batch_size, len(self.players)))

        # Initialize a dictionary to store all sampled values for each player
        all_attendees = {player.name: [[] for _ in range(batch_size)] for player in self.players}

        for category in self.categories.values():
            # make candidates have shape (batch_size, category.size)
            candidates = np.array([np.arange(category.size) for _ in range(batch_size)])
            

            # Vectorized allocation of candidates to players for each game in the batch
            allocations = {
                player.name: np.array([
                    np.random.choice(
                        candidates[_], 
                        size=min(len(candidates[_]), max(0, np.round(player.strategy[category.name] * category.size).astype(int))),  # Ensure size is valid
                        replace=False
                    ) if not np.isnan(player.strategy[category.name] * category.size) else logging.warning(
                    f"NaN encountered for player {player.name} in category {category.name}. "
                    f"Player strategy: {player.strategy[category.name]}, Category size: {category.size}. Skipping allocation.")
                    for _ in range(batch_size)
                ])
                for player in self.players    
            }
            
            #print(allocations)

            # Initialize a dictionary to store masks for each player
            masks = {}

            # Iterate over each player to calculate their mask
            for player in self.players:
                # Use broadcasting to create a mask for all games at once
                player_allocations = allocations[player.name]
                # Reshape candidates and player_allocations for broadcasting
                candidates_expanded = candidates[:, :, None]  # Shape: (batch_size, category.size, 1)
                player_allocations_expanded = player_allocations[:, None, :]  # Shape: (batch_size, 1, num_allocations)

                # Use broadcasting to compare candidates with player allocations
                player_mask = np.any(candidates_expanded == player_allocations_expanded, axis=2)
                #print(player_mask)
                # Store the mask for the current player
                masks[player.name] = player_mask

            #print("Player occupancy masks", masks)

            # Calculate win probabilities for each player
            win_probs = np.array([player.win_value for player in self.players])
            
            # Normalize win probabilities for each candidate
            win_probs_normalized = win_probs / np.sum(win_probs, axis=0)
            #print(win_probs_normalized)
            # Randomly select winners based on win probabilities for each possible candidate

            # THIS LINE IS THE PROBLEM
            '''
            Here we're assigning a winner for each candidate in the category as if there were collisions at every possible candidate.
            The reason this is problematic is because we want to check the conditionals for each player mask.
                So the logic we need to deploy is:
                    for each candidate, 
                    check the mask of each player.
                    if both masks are true, then we consult the winners array to see who wins, else we set the winners array to be either the sole player competing or -1
            '''
            winners = np.random.choice(len(self.players), size=(batch_size, category.size), p=win_probs_normalized)
            
            # Apply the logic described above in a vectorized way
            # Initialize a new winners array with -1, indicating no winner by default
            new_winners = np.full((batch_size, category.size), -1, dtype=int)

            # Iterate over each player to update the new_winners array
            for i, player in enumerate(self.players):
                # Get the mask for the current player
                player_mask = masks[player.name]
                
                # Determine where the current player is the sole competitor
                sole_competitor = player_mask & (np.sum(np.stack([masks[p.name] for p in self.players]), axis=0) == 1)
                #print("Player", i, "is the sole competitor in", sole_competitor)
                # Update new_winners where the current player is the sole competitor
                new_winners = np.where(sole_competitor, i, new_winners)

            # Determine where there are multiple competitors
            multiple_competitors = np.sum(np.stack([masks[p.name] for p in self.players]), axis=0) > 1

            # Use the original winners array to resolve ties where there are multiple competitors
            new_winners = np.where(multiple_competitors, winners, new_winners)
            #print(new_winners)


            # Collect samples for winners
            for i, player in enumerate(self.players):
                winner_mask = (new_winners == i)
                attendees = category.get_samples(np.sum(winner_mask, axis=1))
                
                # Append sampled values to the player's list for each game
                for game_idx in range(batch_size):
                    all_attendees[player.name][game_idx].extend(attendees[game_idx])

        # After processing all categories, handle top-k if needed
        if self.game_mode_type == "top_k":
            for i, player in enumerate(self.players):
                for game_idx in range(batch_size):
                    # Select the top-k values for this game
                    top_k_values = np.sort(all_attendees[player.name][game_idx])[-self.top_k:]
                    batch_results[game_idx, i] = np.sum(top_k_values)
        else:
            # For non-top-k mode, sum all sampled values for each game
            for i, player in enumerate(self.players):
                for game_idx in range(batch_size):
                    batch_results[game_idx, i] = np.sum(all_attendees[player.name][game_idx])

        
        # make sure we're not getting exactly 0 value
        batch_results = np.where(batch_results == 0, 1e-10, batch_results)

        # Calculate total_utilities
        total_utilities = np.sum(batch_results, axis=1, keepdims=True)

        

        # Calculate pct_total_util_array
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered
            warnings.simplefilter("always")
            
            # Perform the division that might raise a RuntimeWarning
            pct_total_util_array = batch_results / total_utilities
            # Check if any RuntimeWarning was raised
            if w and any(issubclass(warn.category, RuntimeWarning) for warn in w):
                print()
                print("Game Conditions:")
                print(f"Game Mode: {self.game_mode_type}")
                print(f"Top K: {self.top_k}")
                print(f"To Admit: {self.to_admit}")
                print(f"Log Normal: {self.log_normal}")
                print("\nCategory Parameters:")
                for category in self.categories.values():
                    print(f"Category {category.name}: mu={category.mu}, sigma={category.sigma}, size={category.size}, distribution={category.log_normal}")
                print("\nPlayer Strategies:")
                for player in self.players:
                    print(f"Player {player.name} (Win Value: {player.win_value}, Blind: {player.blind}, Level: {player.level}):")
                    for category, strategy in player.strategy.items():
                        print(f"  {category}: {strategy}")
                # Debug: Check batch_results
                print("batch_results:", batch_results)
                print("Any negative values in batch_results:", np.any(batch_results < 0))
                print("Any NaN values in batch_results:", np.any(np.isnan(batch_results)))
                print("Any inf values in batch_results:", np.any(np.isinf(batch_results)))
                # Debug: Check total_utilities
                print("total_utilities:", total_utilities)
                print("Any zero or negative values in total_utilities:", np.any(total_utilities <= 0))
                print("Any NaN values in total_utilities:", np.any(np.isnan(total_utilities)))
                print("Any inf values in total_utilities:", np.any(np.isinf(total_utilities)))

                    # Debug: Check pct_total_util_array
                print("pct_total_util_array:", pct_total_util_array)
                print("Any values < 0 in pct_total_util_array:", np.any(pct_total_util_array < 0))
                print("Any values > 1 in pct_total_util_array:", np.any(pct_total_util_array > 1))
                print("Any NaN values in pct_total_util_array:", np.any(np.isnan(pct_total_util_array)))
                print("Any inf values in pct_total_util_array:", np.any(np.isinf(pct_total_util_array)))

                print(w)

        

        # Add validation check
        if np.any(pct_total_util_array < float(0)) or np.any(pct_total_util_array > float(1)):

            print("Game Conditions:")
            print(f"Game Mode: {self.game_mode_type}")
            print(f"Top K: {self.top_k}")
            print(f"To Admit: {self.to_admit}")
            print(f"Log Normal: {self.log_normal}")
            print("\nCategory Parameters:")
            for category in self.categories.values():
                print(f"Category {category.name}: mu={category.mu}, sigma={category.sigma}, size={category.size}, distribution={category.log_normal}")
            print("\nPlayer Strategies:")
            for player in self.players:
                print(f"Player {player.name} (Win Value: {player.win_value}, Blind: {player.blind}, Level: {player.level}):")
                for category, strategy in player.strategy.items():
                    print(f"  {category}: {strategy}")
            # Debug: Check batch_results
            print("batch_results:", batch_results)
            print("Any negative values in batch_results:", np.any(batch_results < 0))
            print("Any NaN values in batch_results:", np.any(np.isnan(batch_results)))
            print("Any inf values in batch_results:", np.any(np.isinf(batch_results)))
            # Debug: Check total_utilities
            print("total_utilities:", total_utilities)
            print("Any zero or negative values in total_utilities:", np.any(total_utilities <= 0))
            print("Any NaN values in total_utilities:", np.any(np.isnan(total_utilities)))
            print("Any inf values in total_utilities:", np.any(np.isinf(total_utilities)))

                # Debug: Check pct_total_util_array
            print("pct_total_util_array:", pct_total_util_array)
            print("Any values < 0 in pct_total_util_array:", np.any(pct_total_util_array < 0))
            print("Any values > 1 in pct_total_util_array:", np.any(pct_total_util_array > 1))
            print("Any NaN values in pct_total_util_array:", np.any(np.isnan(pct_total_util_array)))
            print("Any inf values in pct_total_util_array:", np.any(np.isinf(pct_total_util_array)))
            raise ValueError("expected all percentages to be between 0 and 1")

        return pct_total_util_array




'''Outdated code below this line'''
'''
    def simulate_game(self):
        """
        Simulate the game based on player strategies and resolve outcomes.

        Returns:
            np.ndarray: Array of 'pct_total_util' for each player.
        """
        attendees = {player.name: [] for player in self.players}
        
        for category in self.categories.values():
            candidates = np.arange(category.size)
            
            # Vectorized allocation of candidates to players
            allocations = {
                player.name: np.random.choice(candidates, size=int(np.round(player.strategy[category.name] * category.size)), replace=False)
                for player in self.players
            }
            print(allocations)
            # Create a boolean mask for each player indicating which candidates they have
            masks = {player.name: np.isin(candidates, allocations[player.name]) for player in self.players}
            
            
            # Calculate win probabilities for each player
            win_probs = np.array([player.win_value for player in self.players])
            
            # Normalize win probabilities for each candidate
            win_probs_normalized = win_probs / np.sum(win_probs, axis=0)
            
            # Randomly select winners based on win probabilities for each possible candidate
            winners = np.random.choice(len(self.players), size=category.size, p=win_probs_normalized)
            print(winners)
            # If the boolean mask overlaps with winners for a player, add that sample to the player's attendees
            for i, player in enumerate(self.players):
                winner_mask = (winners == i)
                attendees[player.name].extend(category.get_samples(np.sum(winner_mask)))

        # Calculate utilities and extract 'pct_total_util' for each player
        utilities = self.calculate_utilities(attendees)
        pct_total_util_array = np.array([utilities[player.name]['pct_total_util'] for player in self.players])

        return pct_total_util_array
'''