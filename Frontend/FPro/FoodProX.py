import numpy as np
import pandas as pd
from typing import List, Union, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import os
import datetime

class FoodProXNN(nn.Module):
    """
    Neural network model for NOVA food classification
    """
    def __init__(self, input_size: int, num_classes: int = 4):
        super(FoodProXNN, self).__init__()
        self.model = nn.Sequential(
            # Input layer
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Hidden layer 1
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Hidden layer 2
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Output layer
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess the food nutrient data
    
    Args:
        file_path: Path to the CSV file containing food nutrient data
        
    Returns:
        Preprocessed DataFrame
    """
    df = pd.read_csv(file_path)
    return df

def get_nutrient_columns() -> List[str]:
    """
    Get the ordered list of nutrient columns used for classification
    
    Returns:
        List of column names in the correct order for the classifier
    """
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

def train_nn_model(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    epochs: int = 50, 
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = None
) -> Tuple[FoodProXNN, StandardScaler, dict]:
    """
    Train a PyTorch neural network model for NOVA classification
    
    Args:
        X_train: Training feature data
        y_train: Training target data (NOVA classes 1-4)
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cuda' or 'cpu', None for auto-detection)
    
    Returns:
        Tuple of (trained model, fitted scaler, training history)
    """
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_train_scaled)
    y_tensor = torch.LongTensor(y_train.values - 1)  # Adjust to 0-based indexing
    
    # Create dataset and data loader
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Split into training and validation
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = FoodProXNN(input_size=X_train.shape[1])
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # For early stopping
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    best_model_state = None
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, scaler, history

def predict_NOVA_class(
    model: FoodProXNN,
    scaler: StandardScaler,
    nutrients_df: pd.DataFrame,
    pos_insert_preds: int = 0,
    new_cols_name_prefix: str = "",
    device: str = None
) -> pd.DataFrame:
    """
    Predict NOVA classification for food items based on their nutrient composition using PyTorch
    
    Args:
        model: Trained PyTorch neural network model
        scaler: Fitted StandardScaler for feature normalization
        nutrients_df: DataFrame containing nutrient information
        pos_insert_preds: Position to insert prediction columns
        new_cols_name_prefix: Prefix for the new prediction columns
        device: Device to use for prediction ('cuda' or 'cpu', None for auto-detection)
    
    Returns:
        DataFrame with added prediction columns
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if new_cols_name_prefix:
        new_cols_name_prefix = f"{new_cols_name_prefix.strip()} "
    
    # Get nutrient columns in correct order
    clf_cols_order = get_nutrient_columns()
    
    # Extract and process nutrient values
    try:
        nutrients_df_nuts = nutrients_df[clf_cols_order].copy()
    except KeyError as e:
        raise KeyError(f"Missing required nutrient column: {e}")
    
    # Fill any remaining NaN values with -20 (log(e^-20))
    nutrients_df_nuts = nutrients_df_nuts.fillna(-20)
    
    # Scale the input features
    nutrients_scaled = scaler.transform(nutrients_df_nuts)
    
    # Convert to PyTorch tensor
    nutrients_tensor = torch.FloatTensor(nutrients_scaled).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Make predictions
    try:
        with torch.no_grad():
            outputs = model(nutrients_tensor)
            predict_prob = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            predicted_class = np.argmax(predict_prob, axis=1) + 1  # Add 1 to match NOVA classes (1-4)
    except Exception as e:
        raise RuntimeError(f"Error during prediction: {e}")
    
    # Insert predictions and probabilities
    result_df = nutrients_df.copy()
    new_columns = {
        f"{new_cols_name_prefix}nova-class": predicted_class,
        f"{new_cols_name_prefix}p1": predict_prob[:, 0],  # Unprocessed
        f"{new_cols_name_prefix}p2": predict_prob[:, 1],  # Processed culinary ingredients
        f"{new_cols_name_prefix}p3": predict_prob[:, 2],  # Processed
        f"{new_cols_name_prefix}p4": predict_prob[:, 3]   # Ultra-processed
    }
    
    for i, (col_name, values) in enumerate(new_columns.items()):
        result_df.insert(pos_insert_preds + i, col_name, values)
    
    return result_df

def save_model(model: FoodProXNN, scaler: StandardScaler, path: str = None):
    """
    Save the trained model and scaler
    
    Args:
        model: Trained PyTorch model
        scaler: Fitted StandardScaler
        path: Path to save the model (if None, creates a timestamped file in models directory)
        
    Returns:
        The path where the model was saved
    """
    # Create a models directory if it doesn't exist
    models_dir = "foodprox_models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created directory: {os.path.abspath(models_dir)}")
    
    # Generate a default path with timestamp if none provided
    if path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(models_dir, f"foodprox_model_{timestamp}.pt")
    
    # Save the model and scaler
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler
        }, path)
        print(f"Model saved successfully at: {os.path.abspath(path)}")
        return os.path.abspath(path)
    except Exception as e:
        print(f"Error saving model: {e}")
        return None

def load_model(path: str, input_size: int, device: str = None) -> Tuple[FoodProXNN, StandardScaler]:
    """
    Load a saved model and scaler
    
    Args:
        path: Path to the saved model
        input_size: Number of input features
        device: Device to load the model on ('cuda' or 'cpu', None for auto-detection)
        
    Returns:
        Tuple of (loaded model, loaded scaler)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        checkpoint = torch.load(path, map_location=device)
        model = FoodProXNN(input_size=input_size)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        scaler = checkpoint['scaler']
        print(f"Model loaded successfully from: {path}")
        return model, scaler
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def example_usage():
    """Example of how to use the FoodProX PyTorch neural network classifier"""
    # Load data
    try:
        df = load_and_preprocess_data("food_test_data.csv")
        print(f"Loaded data with {len(df)} rows and {df.shape[1]} columns")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    try:
        # Get features and target for training
        # Using 'nova-class' column instead of 'NOVA_class'
        X_train = df[get_nutrient_columns()]
        y_train = df['nova-class']  # Updated column name to match your data
        print(f"Prepared training data with {len(X_train)} samples and {X_train.shape[1]} features")
        
        # Rest of the function remains the same
        # Train PyTorch neural network model
        print("Starting model training...")
        model, scaler, history = train_nn_model(X_train, y_train, epochs=50, batch_size=32)
        print("Model training completed successfully")
        
        # Make predictions
        print("Making predictions...")
        results = predict_NOVA_class(
            model=model,
            scaler=scaler,
            nutrients_df=df,
            pos_insert_preds=0,
            new_cols_name_prefix=""
        )
        print(f"Generated predictions for {len(results)} samples")
        
        # Save the model for future use
        saved_path = save_model(model, scaler)
        if saved_path:
            print(f"Complete model pipeline executed successfully. Model saved at: {saved_path}")
        
        return results
    
    except Exception as e:
        print(f"Error in example usage: {e}")
        import traceback
        traceback.print_exc()
        return None
    
# If this script is run directly (not imported as a module)
if __name__ == "__main__":
    print("Running FoodProX PyTorch model training and prediction...")
    results = example_usage()
    if results is not None:
        print("Process completed successfully!")
    else:
        print("Process completed with errors.")