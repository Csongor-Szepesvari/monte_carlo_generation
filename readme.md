# Top-K Neural Network Project

This project implements a neural network for Top-K predictions using PyTorch. Below are the instructions for setting up the environment and running the project.

## Prerequisites

### 1. Install Anaconda
Download and install Anaconda from [here](https://www.anaconda.com/products/distribution).

### 2. Install CUDA Toolkit (Optional, for GPU support)
If you want to use GPU acceleration, download and install the CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads). Ensure the version matches your PyTorch installation (e.g., CUDA 12.1 for PyTorch 2.5.1).

### 3. Install Visual Studio Runtime (Windows Only)
Download and install the Visual Studio runtime from [Microsoft's website](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).

## Setup Instructions

### 1. Clone the Repository
git clone https://github.com/Csongor-Szepesvari/top_k_nn.git
cd top_k_nn

### 2. Create the Conda Environment
conda env create -f environment.yml

### 3. Activate the Environment
conda activate top_k_nn_env

### 4. Install GPU-Specific PyTorch (Optional, for GPU support)
If you installed CUDA, install the GPU version of PyTorch:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

### 5. Verify CUDA Installation (Optional)
Check if PyTorch can detect CUDA:
import torch
print(torch.cuda.is_available()) # Should return True
print(torch.version.cuda) # Should match your CUDA version

## Running the Project

### 1. Start Jupyter Notebook
jupyter notebook

### 2. Open and Run the Notebooks
Navigate to the `notebooks/` directory and open the desired notebook (e.g., `top_k_nn.ipynb`).

### 3. Train the Model
Follow the instructions in the notebook to train and evaluate the model.

## Project Structure
- `environment.yml` # Conda environment configuration
- `import_model.py` # grabs the trained neural network and uses it for prediction
- `runs/` # information from the runs
- `best_model.pt` # the stored best model 
- `expanded_monte_carlo_topk.csv` # training data
- `mu_sigma_combinations.csv` # used to generate monte carlo data
- `monte_carlo_generation.py` # pipeline for taking combinations and finding ground truth labels using adaptive Monte Carlo sampling
- `parameter_csv_writer.py` # used to combine mu and sigma values into a csv usable by monte_carlo_generation.py
- `top_k_predictor.py` # file containing neural network architecture and training
- `README.md` # This file
- `requirements.txt` # Pip requirements (optional)

## Troubleshooting

### 1. CUDA Not Detected
- Ensure the CUDA Toolkit and Visual Studio runtime are installed correctly.
- Verify the CUDA version matches the PyTorch version.

### 2. Environment Creation Fails
- Ensure Anaconda is installed and up to date.
- Check for conflicting packages in the `environment.yml` file.

### 3. GPU Not Utilized
- Ensure the GPU version of PyTorch is installed.
- Verify CUDA is available using `torch.cuda.is_available()`.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss the changes.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.




