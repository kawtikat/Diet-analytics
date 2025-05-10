import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime
import os

def get_nutrient_columns() -> list:
    """Get the ordered list of nutrient columns used for classification"""
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

def train_and_save_model(train_data: pd.DataFrame, model_dir: str = 'models') -> str:
    """
    Train the NOVA classifier and save it to a file
    
    Args:
        train_data: DataFrame containing nutrient data and NOVA classifications
        model_dir: Directory to save the model file
        
    Returns:
        Path to the saved model file
    """
    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Get nutrient columns
    nutrient_cols = get_nutrient_columns()
    
    # Validate input data
    missing_cols = [col for col in nutrient_cols if col not in train_data.columns]
    if missing_cols:
        raise ValueError(f"Missing nutrient columns in training data: {missing_cols}")
    if 'nova-class' not in train_data.columns:
        raise ValueError("Missing 'nova-class' column in training data")
    
    # Prepare training data
    X = train_data[nutrient_cols]
    y = train_data['nova-class']
    
    # Initialize and train classifier
    print("Training Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=1,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42
    )
    
    
    clf.fit(X, y)
    print("Training completed successfully!")
    
    # Generate model filename with timestamp
    model_filename = f"modell2.joblib"
    model_path = os.path.join(model_dir, model_filename)
    
    # Save the model
    print(f"Saving model to {model_path}...")
    joblib.dump(clf, model_path)
    print("Model saved successfully!")
    
    # Save feature names for future reference
    feature_filename = f"modell.txt"
    feature_path = os.path.join(model_dir, feature_filename)
    with open(feature_path, 'w') as f:
        f.write('\n'.join(nutrient_cols))
    print(f"Feature names saved to {feature_path}")
    
    return model_path

def main():
    # Load training data
    print("Loading training data...")
    train_data = pd.read_csv("food_train_data.csv")
    print(f"Loaded {len(train_data)} training samples")
    
    # Train and save the model
    model_path = train_and_save_model(train_data)
    
    # Print model information
    print("\nModel training summary:")
    print(f"Total samples: {len(train_data)}")
    print("NOVA class distribution:")
    print(train_data['nova-class'].value_counts().sort_index())
    print(f"\nModel saved to: {model_path}")

if __name__ == "__main__":
    main()