# --- START OF FILE training_neurons_with_optuna.py ---

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib
from datetime import datetime
import optuna # Import Optuna
from optuna.trial import TrialState # For pruning (early stopping)

# --- Data Handling ---

class NutrientDataset(Dataset):
    """Dataset for food nutrient data"""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        # Ensure y is long type for CrossEntropyLoss
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_nutrient_columns() -> list:
    """Get the ordered list of nutrient columns used for classification"""
    # Ensure this list matches the columns expected by your model and data
    return [
        'Protein', 'Total Fat', 'Carbohydrate', 'Alcohol', 'Water', 'Caffeine',
        'Theobromine', 'Sugars, total', 'Fiber, total dietary', 'Calcium', 'Iron',
        'Magnesium', 'Phosphorus', 'Potassium', 'Sodium', 'Zinc', 'Copper',
        'Selenium', 'Retinol', 'Carotene, beta', 'Carotene, alpha',
        'Vitamin E (alpha-tocopherol)', 'Vitamin D (D2 + D3)', 'Cryptoxanthin, beta',
        'Lycopene', 'Lutein + zeaxanthin', 'Vitamin C', 'Thiamin', 'Riboflavin',
        'Niacin', 'Vitamin B-6', 'Folate, total', 'Vitamin B-12', 'Choline, total',
        'Vitamin K (phylloquinone)', 'Folic acid', 'Folate, food', 'Vitamin E, added',
        'Vitamin B-12, added', 'Cholesterol', 'Fatty acids, total saturated',
        'Fatty acids, total monounsaturated', 'Fatty acids, total polyunsaturated'
    ]

# --- Model Definition ---

class FoodProcessingNN(nn.Module):
    """Neural network for food processing classification"""
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.3): # Default values can be overridden
        super(FoodProcessingNN, self).__init__()
        # Added print statement for confirmation during tuning/training
        # print(f"Initializing FoodProcessingNN with input_size={input_size}, hidden_size={hidden_size}, dropout_rate={dropout_rate}")

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 4)  # 4 NOVA classes
        )

    def forward(self, x):
        return self.model(x)

# --- Model Wrapper (for final saving/loading) ---

class PyTorchModelWrapper:
    """Wrapper class to make PyTorch model compatible with sklearn interface"""
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict(self, X):
        """Predict class labels"""
        if isinstance(X, pd.DataFrame):
            X = X[get_nutrient_columns()].values # Ensure correct columns and order
        elif isinstance(X, torch.Tensor):
            X = X.numpy()

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.numpy()

    def predict_proba(self, X):
        """Predict class probabilities"""
        if isinstance(X, pd.DataFrame):
            X = X[get_nutrient_columns()].values # Ensure correct columns and order
        elif isinstance(X, torch.Tensor):
            X = X.numpy()

        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        return probabilities.numpy()

# --- Optuna Objective Function (for Hyperparameter Tuning) ---

def train_evaluate_objective(trial: optuna.Trial, train_data: pd.DataFrame, max_epochs=75) -> float:
    """
    Trains and evaluates a model for a single Optuna trial.
    Accepts hyperparameters suggested by Optuna.
    Returns the best validation accuracy achieved during training for this trial.
    """
    # 1. Suggest hyperparameters using the 'trial' object
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 384, 512])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.6, step=0.05)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"]) # Example: Can tune optimizer

    print(f"\n--- Optuna Trial {trial.number} ---")
    print(f"Params: hidden={hidden_size}, dropout={dropout_rate:.3f}, lr={learning_rate:.6f}, batch={batch_size}")

    # 2. Prepare Data (using a consistent validation split across trials)
    nutrient_cols = get_nutrient_columns()
    missing_cols = [col for col in nutrient_cols if col not in train_data.columns]
    if missing_cols:
        print(f"Warning: Missing columns in trial {trial.number}: {missing_cols}. Skipping trial.")
        return 0.0 # Return low accuracy for invalid data
    if 'nova-class' not in train_data.columns:
        print(f"Warning: Missing 'nova-class' in trial {trial.number}. Skipping trial.")
        return 0.0

    # Simple NaN handling for tuning - consider more sophisticated imputation if needed
    data_clean = train_data.dropna(subset=nutrient_cols + ['nova-class']).copy()
    if len(data_clean) < 0.5 * len(train_data): # Check if too much data was dropped
         print(f"Warning: Significant data loss ({len(train_data)} -> {len(data_clean)}) after dropna in trial {trial.number}.")
    if len(data_clean) < 50:
        print(f"Warning: Insufficient data ({len(data_clean)}) after dropna in trial {trial.number}. Skipping trial.")
        return 0.0

    X = data_clean[nutrient_cols].values
    y = data_clean['nova-class'].astype(int).values

    # Stratified split with fixed random state for comparable validation sets
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y # Use 25% for validation during tuning
        )
    except ValueError: # Handle potential stratification issues with small classes
        print(f"Warning: Stratification failed in trial {trial.number}. Using non-stratified split.")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42)

    # Scale features (fit ONLY on train split)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Create datasets and dataloaders
    train_dataset = NutrientDataset(X_train_scaled, y_train)
    val_dataset = NutrientDataset(X_val_scaled, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0) # num_workers=0 for simplicity
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 3. Initialize Model & Optimizer with suggested params
    input_size = len(nutrient_cols)
    model = FoodProcessingNN(input_size, hidden_size=hidden_size, dropout_rate=dropout_rate)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # if optimizer_name == "RMSprop": optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    # 4. Training Loop with Validation and Pruning/Early Stopping
    best_val_acc = 0.0
    patience_counter = 0
    early_stopping_patience = 15 # Stop trial if no improvement for 15 epochs

    for epoch in range(max_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if torch.isnan(loss) or torch.isinf(loss):
                 print(f"  WARN: NaN/Inf loss during training in trial {trial.number}, epoch {epoch+1}. Pruning.")
                 raise optuna.TrialPruned() # Prune trial if training becomes unstable
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"  WARN: NaN/Inf loss during validation in trial {trial.number}, epoch {epoch+1}. Using current best acc.")
                    break # Stop validation for this epoch, use accuracy so far
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate metrics
        val_acc = 100.0 * correct / total if total > 0 else 0.0

        # Update best validation accuracy for THIS trial
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0 # Reset patience
        else:
            patience_counter += 1

        # Optuna Pruning: Report intermediate value and check if trial should be stopped
        trial.report(val_acc, epoch)
        if trial.should_prune():
            print(f"  Trial {trial.number} pruned by Optuna at epoch {epoch+1}.")
            raise optuna.TrialPruned()

        # Custom Early Stopping Check
        if patience_counter >= early_stopping_patience:
            print(f"  Early stopping triggered at epoch {epoch+1} for trial {trial.number}.")
            break

        # Optional: Print progress less frequently during tuning
        # if (epoch + 1) % 10 == 0:
        #     print(f"  Epoch [{epoch+1}/{max_epochs}], Val Acc: {val_acc:.2f}% (Best: {best_val_acc:.2f}%)")


    print(f"--- Trial {trial.number} Completed --- Best Val Acc: {best_val_acc:.2f}% ---")
    # Return the single best validation accuracy achieved during this trial's training
    return best_val_acc


# --- Function to Run Optuna Study ---

def run_optuna_tuning(train_data: pd.DataFrame, n_trials: int = 50, study_name: str = "pytorch_nova_tuning"):
    """
    Performs hyperparameter optimization using Optuna.
    """
    # Define the objective function with data passed in
    objective_with_data = lambda trial: train_evaluate_objective(trial, train_data, max_epochs=75) # Max 75 epochs per trial

    # Create study: maximize validation accuracy using TPE sampler (Bayesian Optimization)
    # Use a fixed seed for the sampler for reproducibility of the optimization process
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", study_name=study_name, sampler=sampler, load_if_exists=True) # Can resume study

    print(f"\nStarting/Resuming Optuna optimization: '{study_name}' with {n_trials} trials...")
    study.optimize(objective_with_data, n_trials=n_trials, timeout=None) # timeout in seconds if needed

    # Print results
    print("\nOptimization Finished!")
    print(f"Study Name: {study.study_name}")
    print("Number of finished trials: ", len(study.trials))

    # Find and print the best trial
    try:
        best_trial = study.best_trial
        print("\nBest trial:")
        print(f"  Value (Best Validation Accuracy): {best_trial.value:.4f}%")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
        return best_trial.params # Return the best parameters found
    except ValueError:
        print("\nNo completed trials found in the study. Could not determine best parameters.")
        return None

# --- Function to Train and Save the FINAL Model ---

def train_and_save_final_model(train_data: pd.DataFrame, best_params: dict,
                               model_dir: str = 'models', final_epochs=100):
    """
    Trains the final model using the best hyperparameters found by Optuna
    on the full training dataset and saves it.
    """
    print("\n--- Training Final Model with Best Parameters ---")
    print("Using parameters:", best_params)

    # Ensure required params are present
    required_params = ['hidden_size', 'dropout_rate', 'learning_rate', 'batch_size']
    if not all(p in best_params for p in required_params):
        raise ValueError(f"best_params dictionary is missing required keys: {required_params}")

    # Create models directory
    os.makedirs(model_dir, exist_ok=True)

    # Prepare FULL training data
    nutrient_cols = get_nutrient_columns()
    # Final check and cleaning of the full dataset
    missing_cols = [col for col in nutrient_cols if col not in train_data.columns]
    if missing_cols: raise ValueError(f"Missing columns in final training data: {missing_cols}")
    if 'nova-class' not in train_data.columns: raise ValueError("Missing 'nova-class' column")

    data_clean = train_data.dropna(subset=nutrient_cols + ['nova-class']).copy()
    print(f"Using {len(data_clean)} samples for final training after NaN removal.")
    if len(data_clean) < 100: # Arbitrary threshold
        print("Warning: Low number of samples for final training.")

    X_full_train = data_clean[nutrient_cols].values
    y_full_train = data_clean['nova-class'].astype(int).values

    # Scale features using the FULL training data
    scaler = StandardScaler()
    X_full_train_scaled = scaler.fit_transform(X_full_train)

    # Create Dataset & DataLoader for final training (no validation split needed here)
    full_train_dataset = NutrientDataset(X_full_train_scaled, y_full_train)
    # Use the best batch size found during tuning
    final_batch_size = best_params['batch_size']
    full_train_loader = DataLoader(full_train_dataset, batch_size=final_batch_size, shuffle=True, num_workers=0)

    # Initialize model with best hyperparameters
    input_size = len(nutrient_cols)
    model = FoodProcessingNN(input_size,
                             hidden_size=best_params['hidden_size'],
                             dropout_rate=best_params['dropout_rate'])

    # Define loss and optimizer with best hyperparameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    # Add other optimizers if tuned:
    # if best_params.get("optimizer") == "RMSprop": optimizer = optim.RMSprop(...)

    # Final Training Loop (no validation needed, train on all data)
    print(f"Starting final training for {final_epochs} epochs...")
    model.train()
    for epoch in range(final_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in full_train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if torch.isnan(loss) or torch.isinf(loss):
                 print(f"ERROR: NaN/Inf loss during final training at epoch {epoch+1}. Stopping.")
                 # Consider saving the model state from the previous epoch if needed
                 return None # Indicate failure

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(full_train_loader) if len(full_train_loader) > 0 else 0.0
        epoch_acc = 100.0 * correct / total if total > 0 else 0.0

        if (epoch + 1) % 10 == 0 or epoch == final_epochs - 1:
            print(f"Epoch [{epoch+1}/{final_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    print("Final training completed!")

    # Wrap the trained model and scaler
    wrapper_model = PyTorchModelWrapper(model, scaler)

    # Save the final model and scaler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"nova_pytorch_model_final_{timestamp}.joblib"
    model_path = os.path.join(model_dir, model_filename)
    feature_filename = f"nova_pytorch_features_{timestamp}.txt"
    feature_path = os.path.join(model_dir, feature_filename)

    print(f"Saving final model to {model_path}...")
    try:
        joblib.dump(wrapper_model, model_path)
        print("Model saved successfully!")
        # Save feature names used by the final model
        with open(feature_path, 'w') as f:
            f.write('\n'.join(nutrient_cols))
        print(f"Feature names saved to {feature_path}")
        return model_path
    except Exception as e:
        print(f"Error saving final model: {e}")
        return None


# --- Main Execution Logic ---

def main():
    # --- Configuration ---
    DATA_FILE = "food_train_data.csv"
    MODEL_DIR = "models"
    RUN_MODE = "train_final" # Options: "tune", "train_final"
    N_OPTUNA_TRIALS = 30 # Number of trials for tuning
    FINAL_TRAINING_EPOCHS = 100 # Epochs for final model training

    # --- Load Data ---
    print(f"Loading training data from {DATA_FILE}...")
    try:
        # Load the full training data ONCE
        train_data = pd.read_csv(DATA_FILE)
        print(f"Loaded {len(train_data)} samples")
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found. Please ensure the data file exists.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- Execution Based on Mode ---
    if RUN_MODE == "tune":
        print("\n--- Running Hyperparameter Tuning ---")
        best_params = run_optuna_tuning(train_data, n_trials=N_OPTUNA_TRIALS)

        if best_params:
            print("\n--- Tuning Complete ---")
            print("Best parameters found:")
            print(best_params)
            print(f"\nTo train the final model with these parameters:")
            print(f"1. Set RUN_MODE = 'train_final' in the script.")
            print(f"2. Manually update the 'best_params_for_final_training' dictionary below with these values.")
            print(f"3. Rerun the script.")
        else:
            print("\n--- Tuning Failed or No Trials Completed ---")

    elif RUN_MODE == "train_final":
        print("\n--- Running Final Model Training ---")
        # IMPORTANT: Update this dictionary with the results from your tuning run!
        best_params_for_final_training = {
            'hidden_size': 512,      # Replace with your best value
            'dropout_rate': 0.1,    # Replace with your best value
            'learning_rate': 0.0003574927606865839, # Replace with your best value
            'batch_size': 16         # Replace with your best value
            # Add other parameters if you tuned them (e.g., 'optimizer': 'Adam')
        }
        print("Using predefined best parameters for final training:")
        print(best_params_for_final_training)

        model_path = train_and_save_final_model(train_data,
                                                best_params=best_params_for_final_training,
                                                model_dir=MODEL_DIR,
                                                final_epochs=FINAL_TRAINING_EPOCHS)
        if model_path:
            print(f"\nFinal model saved to: {model_path}")
            print("Remember to evaluate this model on a separate, unseen test dataset.")
        else:
            print("\nFinal model training or saving failed.")

    else:
        print(f"Error: Invalid RUN_MODE '{RUN_MODE}'. Choose 'tune' or 'train_final'.")


if __name__ == "__main__":
    # Set random seeds for reproducibility (optional but recommended)
    # np.random.seed(42)
    # torch.manual_seed(42)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(42)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    main()

# --- END OF FILE ---