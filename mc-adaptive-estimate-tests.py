import pandas as pd
import numpy as np
from monte_carlo_generation import monte_carlo_adaptive_estimate
import ast

def convert_string_to_list(s):
    """Convert string representation of list to actual list"""
    return ast.literal_eval(s)

def main():
    # Read the test cases
    df = pd.read_csv('Theoretical_Monte_Carlo_Test_Cases_with_Expected_R.csv')
    
    # Convert string columns to actual lists
    df['mu'] = df['mu'].apply(convert_string_to_list)
    df['sigma'] = df['sigma'].apply(convert_string_to_list)
    df['n_parts'] = df['n_parts'].apply(convert_string_to_list)
    
    # Initialize list to store computed R values
    R_computed = []
    errors = []
    
    # Process each test case
    for idx, row in df.iterrows():
        try:
            # Extract parameters
            mu = row['mu']
            sigma = row['sigma']
            n_parts = row['n_parts']
            k = row['k']
            epsilon = row['epsilon']
            
            # Check if all n_parts are zero
            if all(n == 0 for n in n_parts):
                print(f"\nTest case {idx + 1}: All n_parts are zero, skipping...")
                R_computed.append(0)
                errors.append("All n_parts are zero")
                continue
            
            # Run the monte carlo estimation
            print(f"\nProcessing test case {idx + 1}:")
            print(f"mu: {mu}")
            print(f"sigma: {sigma}")
            print(f"n_parts: {n_parts}")
            print(f"k: {k}")
            print(f"epsilon: {epsilon}")
            
            _, R = monte_carlo_adaptive_estimate(
                mu=mu,
                sigma=sigma,
                n_parts=n_parts,
                k=k,
                epsilon=epsilon,
                max_rep=10000
            )
            
            R_computed.append(R)
            errors.append(abs(R - row['R_expected']))
            print(f"Computed R: {R}")
            
        except Exception as e:
            print(f"Error in test case {idx + 1}: {str(e)}")
            R_computed.append(0)
            errors.append(str(e))
    
    # Add computed R values and errors to dataframe
    df['R_computed'] = R_computed
    df['error'] = errors
    
    # Print the results
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print("\nTest Results:")
    print("=" * 100)
    print(df)
    print("=" * 100)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 50)
    print(f"Mean R Expected: {df['R_expected'].mean():.2f}")
    print(f"Mean R_computed: {df['R_computed'].mean():.2f}")
    print(f"R matches R_computed: {sum(df['R_expected'] == df['R_computed'])} out of {len(df)} cases")
    print(f"Number of errors: {sum(df['error'].notna())}")

if __name__ == "__main__":
    main()
