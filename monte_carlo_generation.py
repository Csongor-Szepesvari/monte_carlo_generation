import numpy as np
import pandas as pd
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

def stars_and_bars_partition(total_n, k=4, rng=None):
    """
    Generate a random partition of total_n into k parts using the stars and bars method.
    
    Args:
        total_n (int): The total number to partition
        k (int): Number of parts to divide into (default: 4)
        rng (numpy.random.Generator): Random number generator
        
    Returns:
        list: A list of k integers that sum to total_n
    """
    if rng is None:
        rng = np.random.default_rng()
    cuts = rng.integers(0, total_n + 1, size=k - 1)
    cuts.sort()
    parts = [cuts[0]] + [cuts[i] - cuts[i - 1] for i in range(1, k - 1)] + [total_n - cuts[-1]]
    return parts


def biased_stars_and_bars_partition(total_n, k=4, rng=None, bias_factor=2.0):
    """
    Generate a random partition of total_n into k parts with bias towards first partition.
    
    Args:
        total_n (int): The total number to partition
        k (int): Number of parts to divide into (default: 4)
        rng (numpy.random.Generator): Random number generator
        bias_factor (float): How much to bias towards first partition (default: 2.0)
        
    Returns:
        list: A list of k integers that sum to total_n
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Generate weights for each partition
    weights = np.ones(k)
    weights[0] = bias_factor  # Higher weight for first partition
    
    # Generate cuts with probability proportional to weights
    cuts = []
    remaining = total_n
    for i in range(k-1):
        if remaining == 0:
            cuts.append(0)
            continue
            
        # Calculate probabilities for next cut
        probs = weights[i+1:] / weights[i+1:].sum()
        # Generate next cut position
        next_cut = rng.integers(0, remaining+1)
        # Apply probability weights
        next_cut = int(next_cut * (1 - probs[0]))
        cuts.append(next_cut)
        remaining -= next_cut
    
    cuts.sort()
    parts = [cuts[0]] + [cuts[i] - cuts[i-1] for i in range(1, k-1)] + [total_n - cuts[-1]]
    return parts




def simulate_topk_given_fixed_config(mu, sigma, n_parts, k, rng):
    """
    Simulate drawing from normal distributions and calculate the sum of top-k elements.
    
    Args:
        mu (list): Mean values for each distribution
        sigma (list): Standard deviation values for each distribution
        n_parts (list): Number of samples to draw from each distribution
        k (int): Number of top elements to sum
        rng (numpy.random.Generator): Random number generator
        
    Returns:
        float: Sum of the top-k elements
    """
    all_samples = []
    for i in range(4):
        draws = n_parts[i]
        if draws > 0:
            samples = mu[i] + sigma[i] * rng.standard_normal(draws)
            all_samples.append(samples)
    combined = np.concatenate(all_samples)
    top_k = np.sort(combined)[-k:] if len(combined) >= k else combined
    return float(np.sum(top_k))

def monte_carlo_adaptive_estimate(mu, sigma, n_parts, k, epsilon=0.01, max_rep=10000):
    """
    Perform adaptive Monte Carlo estimation for the sum of top-k elements.
    Continues sampling until the relative error is below epsilon or max_rep is reached.
    
    Args:
        mu (list): Mean values for each distribution
        sigma (list): Standard deviation values for each distribution
        n_parts (list): Number of samples to draw from each distribution
        k (int): Maximum number of top elements to sum
        epsilon (float): Target relative error threshold (default: 0.01)
        max_rep (int): Maximum number of replications (default: 10000)
        
    Returns:
        tuple: (dict of top-k sums for k=1 to k, number of replications performed)
    """
    rng = np.random.default_rng()
    replicate_values = []
    all_topk_values = {}  # Dictionary to store results for all k values from 1 to k

    R = 10
    for _ in range(R):
        # Generate samples
        all_samples = []
        for i in range(4):
            draws = n_parts[i]
            if draws > 0:
                samples = mu[i] + sigma[i] * rng.standard_normal(draws)
                all_samples.append(samples)
        
        combined = np.concatenate(all_samples)
        sorted_values = np.sort(combined)
        
        # Store top-k values for all k from 1 to k
        for j in range(1, k+1):
            top_j = sorted_values[-j:] if len(combined) >= j else sorted_values
            if j not in all_topk_values:
                all_topk_values[j] = []
            all_topk_values[j].append(float(np.sum(top_j)))
        
        # For convergence check, we use the original k value
        replicate_values.append(float(np.sum(sorted_values[-k:] if len(combined) >= k else sorted_values)))
    
    replicate_values = np.array(replicate_values)
    mean = replicate_values.mean()
    std = replicate_values.std(ddof=1)
    se = std / np.sqrt(R)
    rel_error = se / mean if mean != 0 else np.inf

    while rel_error > epsilon and R < max_rep:
        for _ in range(R):
            # Generate samples
            all_samples = []
            for i in range(4):
                draws = n_parts[i]
                if draws > 0:
                    samples = mu[i] + sigma[i] * rng.standard_normal(draws)
                    all_samples.append(samples)
            
            combined = np.concatenate(all_samples)
            sorted_values = np.sort(combined)
            
            # Store top-k values for all k from 1 to k
            for j in range(1, k+1):
                top_j = sorted_values[-j:] if len(combined) >= j else sorted_values
                all_topk_values[j].append(float(np.sum(top_j)))
            
            # For convergence check, we use the original k value
            replicate_values = np.append(replicate_values, 
                                        float(np.sum(sorted_values[-k:] if len(combined) >= k else sorted_values)))
        
        R = len(replicate_values)
        mean = replicate_values.mean()
        std = replicate_values.std(ddof=1)
        se = std / np.sqrt(R)
        rel_error = se / mean if mean != 0 else np.inf

    # Calculate means for all k values
    topk_means = {j: np.mean(all_topk_values[j]) for j in range(1, k+1)}
    
    return topk_means, R

def process_row_with_multiple_configs(
        row, n_values=None, num_partitions=5, epsilon=0.01, 
        bias_fraction=0.5, bias_factor=2.0
    ):
    """
    Process a single row of parameters with a mix of biased and non-biased partitions.
    
    Args:
        row (array): Array containing mu and sigma values
        n_values (list): List of total sample sizes to try
        num_partitions (int): Number of different partitions to try per n value
        epsilon (float): Target relative error threshold
        bias_fraction (float): Fraction of partitions to bias (default: 0.5)
        bias_factor (float): Bias factor for first partition (default: 2.0)
        
    Returns:
        list: List of result rows
    """
    rng = np.random.default_rng()
    mu = row[0::2]
    sigma = row[1::2]
    results = []

    for total_n in n_values:
        k = int(total_n * 0.25)
        for i in range(num_partitions):
            # Decide whether to use biased or non-biased partition
            use_biased = rng.random() < bias_fraction
            if use_biased:
                n_parts = biased_stars_and_bars_partition(total_n, rng=rng, bias_factor=bias_factor)
            else:
                n_parts = stars_and_bars_partition(total_n, rng=rng)
            
            topk_means, num_reps = monte_carlo_adaptive_estimate(mu, sigma, n_parts, k, epsilon=epsilon)
            
            # Create a base result with common values
            base_result = list(mu) + list(sigma) + [total_n] + n_parts
            
            # Add a row for each k value
            for j in range(1, k + 1):
                results.append(base_result + [j, topk_means[j], num_reps])
    
    return results

def process_row_wrapper(row, n_values, num_partitions, epsilon, bias_fraction, bias_factor):
    """Wrapper function for parallel processing (must be at module level for pickling)."""
    return process_row_with_multiple_configs(
        row, n_values, num_partitions, epsilon, bias_fraction, bias_factor
    )

def generate_expanded_monte_carlo_dataset(
        csv_path,
        n_values=None,
        num_partitions=5,
        epsilon=0.01,
        parallel=True,
        output_csv="expanded_monte_carlo_topk.csv",
        bias_fraction=0.5,
        bias_factor=2.0
    ):
    """
    Generate a Monte Carlo dataset with a mix of biased and non-biased partitions.
    
    Args:
        csv_path (str): Path to input CSV with mu and sigma values
        n_values (list): List of total sample sizes to try
        num_partitions (int): Number of different partitions to try per n value
        epsilon (float): Target relative error threshold
        parallel (bool): Whether to use parallel processing
        output_csv (str): Path to output CSV file
        bias_fraction (float): Fraction of partitions to bias (default: 0.5)
        bias_factor (float): Bias factor for first partition (default: 2.0)
        
    Returns:
        None: Results are saved to output_csv
    """
    import time
    start_time = time.time()
    
    if n_values is None:
        n_values = list(range(10, 61, 3))

    df = pd.read_csv(csv_path)
    param_array = df.values
    total_rows = len(param_array)
    print(f"Starting processing of {total_rows} parameter combinations...")
    
    if parallel:
        with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = list(executor.map(
                process_row_wrapper,
                param_array,
                [n_values] * len(param_array),
                [num_partitions] * len(param_array),
                [epsilon] * len(param_array),
                [bias_fraction] * len(param_array),
                [bias_factor] * len(param_array)
            ))
            nested_results = []
            for i, future in enumerate(futures):
                nested_results.append(future)
                if (i+1) % max(1, total_rows//10) == 0:  # Report every 10%
                    elapsed = time.time() - start_time
                    percent_complete = (i+1) / total_rows * 100
                    print(f"Progress: {percent_complete:.1f}% complete ({i+1}/{total_rows}), "
                          f"Time elapsed: {elapsed:.1f}s, "
                          f"Est. remaining: {(elapsed/(i+1))*(total_rows-i-1):.1f}s")
    else:
        nested_results = []
        for i, row in enumerate(param_array):
            result = process_row_with_multiple_configs(
                row, n_values=n_values, num_partitions=num_partitions, 
                epsilon=epsilon, bias_fraction=bias_fraction, bias_factor=bias_factor
            )
            nested_results.append(result)
            if (i+1) % max(1, total_rows//10) == 0:  # Report every 10%
                elapsed = time.time() - start_time
                percent_complete = (i+1) / total_rows * 100
                print(f"Progress: {percent_complete:.1f}% complete ({i+1}/{total_rows}), "
                      f"Time elapsed: {elapsed:.1f}s, "
                      f"Est. remaining: {(elapsed/(i+1))*(total_rows-i-1):.1f}s")

    results = [item for sublist in nested_results for item in sublist]

    columns = [f"mu{i+1}" for i in range(4)] + [f"sigma{i+1}" for i in range(4)] + \
              ["total_n"] + [f"n{i+1}" for i in range(4)] + ["k", "avg_topk_sum", "num_replicates"]

    result_df = pd.DataFrame(results, columns=columns)
    result_df.to_csv(output_csv, index=False)
    
    total_time = time.time() - start_time
    print(f"Saved expanded MC results to '{output_csv}' with shape {result_df.shape}")
    print(f"Total execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)")

def main():
    """
    Main function to run the Monte Carlo simulation.
    """
    generate_expanded_monte_carlo_dataset(
        csv_path="mu_sigma_combinations.csv",
        n_values=None,                    # Run the default
        num_partitions=10,                # 10 stars-and-bars per n
        epsilon=0.01,                     # relative error threshold
        parallel=True,
        output_csv="mixed_monte_carlo_topk.csv",
        bias_fraction=0.5,                # 50% of partitions will be biased
        bias_factor=2.0                   # First partition gets 2x weight
    )

if __name__ == "__main__":
    main()
