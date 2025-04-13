import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils import standardize
import pandas as pd
from top_k_predictor import DeepSetsTopKModel, TopKDataset
from torch.utils.data import DataLoader

class BayesianOptimizer:
    def __init__(self, model_path, data_path, n_bounds=None):
        """
        Initialize the Bayesian optimizer.
        
        Args:
            model_path (str): Path to the trained model
            data_path (str): Path to the training data
            n_bounds (dict): Dictionary of bounds for each n parameter. If None, uses default bounds.
                            Example: {'n1': (1, 30), 'n2': (1, 30), 'n3': (1, 30), 'n4': (1, 30)}
        """
        # Load the trained model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DeepSetsTopKModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.double()  # Convert model to double precision
        self.model.eval()
        
        # Load the data
        self.df = pd.read_csv(data_path)
        
        # Get average mu and sigma values from the dataset
        self.mu_values = np.array([
            self.df[f'mu{i+1}'].mean() for i in range(4)
        ])
        self.sigma_values = np.array([
            self.df[f'sigma{i+1}'].mean() for i in range(4)
        ])
        
        # Set parameter bounds
        if n_bounds is None:
            # Default bounds if none provided
            self.n_bounds = {
                f'n{i+1}': (1, 30) for i in range(4)
            }
        else:
            self.n_bounds = n_bounds
        
        # Store bounds as tensors for optimization
        self.lower_bounds = torch.tensor([self.n_bounds[f'n{i+1}'][0] for i in range(4)], device=self.device).double()
        self.upper_bounds = torch.tensor([self.n_bounds[f'n{i+1}'][1] for i in range(4)], device=self.device).double()
        
    def objective_function(self, X):
        """
        Objective function for Bayesian optimization.
        
        Args:
            X (torch.Tensor): Input parameters of shape [batch_size, 3]
                             (n values for n1, n2, n3)
        
        Returns:
            torch.Tensor: Predicted top-k sum
        """
        batch_size = X.shape[0]
        
        # Create full input tensor with constant mu and sigma
        x = torch.zeros(batch_size, 4, 3, device=self.device, dtype=torch.float64)
        
        # For each sample, use its corresponding k value from the dataset
        k_values = torch.zeros(batch_size, 1, device=self.device, dtype=torch.float64)
        
        for i in range(batch_size):
            # Randomly sample a row from the dataset
            sample_idx = torch.randint(0, len(self.df), (1,)).item()
            k = self.df.iloc[sample_idx]['k']
            total_n = self.df.iloc[sample_idx]['total_n']
            k_values[i] = k / total_n  # Normalize k by total_n
            
            # Calculate n4
            n1, n2, n3 = X[i].cpu().numpy()
            n4 = total_n - (n1 + n2 + n3)
            
            # Set all values from X
            x[i, :, 0] = torch.tensor(self.mu_values, device=self.device, dtype=torch.float64)  # mu
            x[i, :, 1] = torch.tensor(self.sigma_values, device=self.device, dtype=torch.float64)  # sigma
            x[i, 0, 2] = n1  # n1
            x[i, 1, 2] = n2  # n2
            x[i, 2, 2] = n3  # n3
            x[i, 3, 2] = n4  # n4
        
        with torch.no_grad():
            predictions = self.model(x, k_values)
        
        return predictions.unsqueeze(-1).double()  # Ensure double precision output
    
    def optimize(self, n_init=10, n_iter=20):
        """
        Run Bayesian optimization.
        
        Args:
            n_init (int): Number of initial random samples
            n_iter (int): Number of optimization iterations
        
        Returns:
            dict: Results of the optimization
        """
        # Generate initial random samples for n1, n2, n3
        train_X = torch.zeros(n_init, 3, device=self.device)
        for i in range(3):  # Only for n1, n2, n3
            train_X[:, i] = torch.randint(
                self.n_bounds[f'n{i+1}'][0],
                self.n_bounds[f'n{i+1}'][1] + 1,
                (n_init,)
            ).double()  # Use double precision
        
        # Scale to unit cube
        train_X = (train_X - self.lower_bounds[:3]) / (self.upper_bounds[:3] - self.lower_bounds[:3])
        
        train_Y = self.objective_function(train_X)
        
        # Standardize the outputs with numerical stability
        Y_mean = train_Y.mean(dim=0, keepdim=True)
        Y_std = train_Y.std(dim=0, keepdim=True) + 1e-6  # Add small epsilon
        train_Y = (train_Y - Y_mean) / Y_std
        
        # Optimization loop
        for i in range(n_iter):
            # Fit GP model
            gp = SingleTaskGP(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            
            # Define acquisition function
            ei = LogExpectedImprovement(gp, best_f=train_Y.max())
            
            # Optimize acquisition function
            candidates, _ = optimize_acqf(
                ei,
                bounds=torch.tensor([
                    [0.0] * 3,  # Scaled bounds for n1, n2, n3
                    [1.0] * 3
                ]).to(self.device).double(),  # Use double precision
                q=1,
                num_restarts=20,
                raw_samples=100
            )
            
            # Evaluate new candidate
            new_Y = self.objective_function(candidates)
            new_Y = (new_Y - Y_mean) / Y_std  # Use same standardization
            
            # Verify each candidate solution
            valid_candidates = []
            valid_new_Y = []
            for j, candidate in enumerate(candidates):
                n1, n2, n3 = candidate.cpu().numpy()  # Only unpack n1, n2, n3
                total_n = self.df.iloc[torch.randint(0, len(self.df), (1,)).item()]['total_n']
                is_valid, n4, message = self.verify_solution(n1, n2, n3, total_n)
                if is_valid:
                    valid_candidates.append(candidate)
                    valid_new_Y.append(new_Y[j])
                else:
                    print(f"Iteration {i+1}: {message}")

            # Update training data with valid solutions
            if valid_candidates:
                valid_candidates = torch.stack(valid_candidates).to(self.device)
                valid_new_Y = torch.stack(valid_new_Y).to(self.device)
                train_X = torch.cat([train_X, valid_candidates])
                train_Y = torch.cat([train_Y, valid_new_Y])
            
            print(f"Iteration {i+1}/{n_iter}: Best value = {train_Y.max().item():.4f}")
        
        # Get posterior distributions
        gp = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        # Find best parameters
        best_idx = train_Y.argmax()
        best_params = train_X[best_idx]
        
        # Rescale parameters back to original scale
        best_params = best_params * (self.upper_bounds[:3] - self.lower_bounds[:3]) + self.lower_bounds[:3]
        train_X = train_X * (self.upper_bounds[:3] - self.lower_bounds[:3]) + self.lower_bounds[:3]
        
        return {
            'best_params': best_params.cpu().numpy(),
            'best_value': train_Y[best_idx].item(),
            'train_X': train_X.cpu().numpy(),
            'train_Y': train_Y.cpu().numpy(),
            'gp': gp
        }
    
    def plot_posteriors(self, results, output_dir='bayesian_optimization_results'):
        """
        Plot posterior distributions of the parameters.
        
        Args:
            results (dict): Results from the optimization
            output_dir (str): Directory to save plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot parameter distributions
        param_names = [f"n{i+1}" for i in range(3)]  
        train_X = results['train_X']
        
        plt.figure(figsize=(15, 5))
        for i, name in enumerate(param_names):
            plt.subplot(1, 4, i+1)  # Changed to 4 subplots
            sns.histplot(train_X[:, i], kde=True, bins=20)
            plt.title(f'Distribution of {name}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'parameter_distributions.png'))
        plt.close()
        
        # Plot optimization progress
        plt.figure(figsize=(10, 6))
        plt.plot(results['train_Y'], 'b-', label='Objective Value')
        plt.xlabel('Iteration')
        plt.ylabel('Standardized Objective Value')
        plt.title('Optimization Progress')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'optimization_progress.png'))
        plt.close()
        
        # Plot correlation matrix
        plt.figure(figsize=(10, 8))
        corr_matrix = np.corrcoef(train_X.T)
        sns.heatmap(corr_matrix, 
                   xticklabels=param_names,
                   yticklabels=param_names,
                   cmap='coolwarm',
                   center=0)
        plt.title('Parameter Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
        plt.close()

    def verify_solution(self, n1, n2, n3, total_n):
        """
        Verify if a solution is valid and calculate n4.
        
        Args:
            n1, n2, n3 (float): Values for n1, n2, n3
            total_n (float): Total n value from the dataset
        
        Returns:
            tuple: (is_valid, n4, message)
        """
        n4 = total_n - (n1 + n2 + n3)
        
        # Check if n4 is positive
        if n4 < 1:
            return False, n4, f"Invalid solution: n4 = {n4:.2f} is negative or too small"
        
        # Check if sum equals total_n (with some tolerance)
        if abs((n1 + n2 + n3 + n4) - total_n) > 1e-6:
            return False, n4, f"Invalid solution: Sum {n1 + n2 + n3 + n4:.2f} != total_n {total_n:.2f}"
        
        # Check if n4 is not too large (optional constraint)
        if n4 > total_n * 0.5:
            return False, n4, f"Invalid solution: n4 = {n4:.2f} is too large (>50% of total_n)"
        
        return True, n4, "Valid solution"

def main():
    # Define custom bounds for each n parameter to be optimized within.
    n_bounds = {
        'n1': (5, 30),   
        'n2': (1, 20),  
        'n3': (1, 20),  
        'n4': (1, 15)  
    }
    
    # Initialize optimizer with custom bounds
    optimizer = BayesianOptimizer(
        model_path='runs/topk_experiment/best_model.pt',
        data_path='expanded_monte_carlo_topk.csv',
        n_bounds=n_bounds
    )
    
    # Run optimization
    results = optimizer.optimize(n_init=30, n_iter=20)
    
    # Plot results
    optimizer.plot_posteriors(results)
    
    # Print best parameters
    print("\nBest parameters found:")
    print("Constant mu values:", optimizer.mu_values)
    print("Constant sigma values:", optimizer.sigma_values)
    print("\nOptimized n values:")
    for i, value in enumerate(results['best_params']):
        print(f"n{i+1}: {value:.1f} (range: {optimizer.n_bounds[f'n{i+1}']})")
    print(f"\nBest objective value: {results['best_value']:.4f}")

if __name__ == '__main__':
    main() 