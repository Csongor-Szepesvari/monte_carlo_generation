high_low_ratio_means = [i / 10 for i in range(12, 21, 2)]  # 1.2, 1.4, ..., 2.0
high_low_ratio_variances = [i / 10 for i in range(12, 21, 2)]
mean_variance_ratios = [i / 100 for i in range(50, 151, 5)]  # 0.5, 0.75, ..., 1.5

import csv

# Create a list to store all combinations
combinations = []

# Iterate through all possible combinations of parameters
for high_low_ratio_mean in high_low_ratio_means:
    for high_low_ratio_variance in high_low_ratio_variances:
        for mean_variance_ratio in mean_variance_ratios:
            # Calculate mu values
            mu_low = 5.0  # Base value for low mean
            mu_high = round(mu_low * high_low_ratio_mean, 2)
            
            # Calculate sigma values
            sigma_low = round(mu_low * mean_variance_ratio, 2)
            sigma_high = round(sigma_low * high_low_ratio_variance, 2)
            
            # Append the combination to the list
            combinations.append([mu_high, sigma_high, mu_high, sigma_low, mu_low, sigma_high, mu_low, sigma_low])

# Write combinations to CSV
with open('mu_sigma_combinations.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['mu_1',"sigma_1","mu_2","sigma_2","mu_3","sigma_3","mu_4","sigma_4"])  # Write header
    writer.writerows(combinations)  # Write all combinations
