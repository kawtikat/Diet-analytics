import numpy as np
import pandas as pd
import joblib
from typing import List, Optional

def get_nutrient_columns() -> List[str]:
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

def calculate_fpro(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the Food Processing Score (FPro) for each food item
    
    Args:
        data: DataFrame with p1 and p4 probability columns
        
    Returns:
        DataFrame with added FPro column
    """
    result = data.copy()
    result['FPro'] = (1 - result['p1'] + result['p4']) / 2
    return result

def load_model(model_path: str):
    """
    Load the saved FoodProX classifier model
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded classifier model
    """
    print(f"Loading model from {model_path}...")
    clf = joblib.load(model_path)
    print("Model loaded successfully!")
    return clf

def analyze_foods_with_fpro(
    food_data: pd.DataFrame,
    model_path: str,
    food_description_col: str = 'Main food description'
) -> pd.DataFrame:
    """
    Run the FoodProX classifier on food data and calculate FPro scores
    
    Args:
        food_data: DataFrame containing food nutrient data
        model_path: Path to the saved classifier model
        food_description_col: Name of the column containing food descriptions
        
    Returns:
        DataFrame with NOVA classes, probabilities, and FPro scores
    """
    # Load the model
    clf = load_model(model_path)
    
    # Get necessary columns
    nutrient_cols = get_nutrient_columns()
    
    # Validate input data
    missing_cols = [col for col in nutrient_cols if col not in food_data.columns]
    if missing_cols:
        raise ValueError(f"Missing nutrient columns in food data: {missing_cols}")
    
    # Prepare input data
    X = food_data[nutrient_cols].copy()
    
    # Log transform and handle zeros/NaNs
    # Note: If data is already log-transformed, skip this step
    X = X.fillna(-20)
    X = X.replace([np.inf, -np.inf], -20)
    
    # Make predictions
    print("Running classifier on food data...")
    predicted_class = clf.predict(X)
    predict_prob = clf.predict_proba(X)
    
    # Add predictions to results
    results = food_data.copy()
    results['nova-class'] = predicted_class
    results['p1'] = predict_prob[:, 0]  # Unprocessed
    results['p2'] = predict_prob[:, 1]  # Processed culinary ingredients
    results['p3'] = predict_prob[:, 2]  # Processed
    results['p4'] = predict_prob[:, 3]  # Ultra-processed
    
    # Calculate FPro score
    results = calculate_fpro(results)
    
    # Format FPro to 5 decimal places
    results['FPro'] = results['FPro'].round(5)
    
    print("Analysis completed successfully!")
    return results

def main():
    # Load food data
    print("Loading food data...")
    food_data = pd.read_csv("food_train_data.csv")
    
    # Path to saved model
    model_path = "models/foodprox_model_20250301_120000.joblib"  # Update with your actual model path
    
    # Analyze foods and calculate FPro
    results = analyze_foods_with_fpro(food_data, model_path)
    
    # Display example results
    print("\nExample Foods with FPro Scores:")
    example_cols = ['Main food description', 'nova-class', 'p1', 'p4', 'FPro']
    print(results[example_cols].head(10))
    
    # Save results to CSV
    output_file = "food_fpro_results.csv"
    results.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Display FPro statistics
    print("\nFPro Statistics:")
    print(f"Min FPro: {results['FPro'].min()}")
    print(f"Max FPro: {results['FPro'].max()}")
    print(f"Mean FPro: {results['FPro'].mean()}")
    
    # Group by NOVA class and show average FPro
    print("\nAverage FPro by NOVA Class:")
    nova_fpro = results.groupby('nova-class')['FPro'].mean().reset_index()
    for _, row in nova_fpro.iterrows():
        print(f"NOVA {int(row['nova-class'])}: {row['FPro']:.5f}")

if __name__ == "__main__":
    main()