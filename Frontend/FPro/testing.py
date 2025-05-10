# --- START OF FILE testing.py ---

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # Import accuracy_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import os # Import os for path joining potentially

# --- IMPORT THE REQUIRED CLASSES ---
# Import the custom classes from the script where they were defined and saved.
# Replace 'training_neurons_with_optuna' with the actual filename of your training script if different.
try:
    from training_neurons_with_optuna import PyTorchModelWrapper, FoodProcessingNN, get_nutrient_columns
except ImportError:
    print("\n--- ERROR ---")
    print("Could not import required classes (PyTorchModelWrapper, FoodProcessingNN, get_nutrient_columns).")
    print("Make sure the training script (e.g., 'training_neurons_with_optuna.py')")
    print("is in the same directory as this testing script or accessible in the Python path.")
    print("-----------")
    exit() # Stop execution if imports fail
# --- END OF IMPORTS ---


def load_model_and_predict(model_path: str, test_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the saved model and make predictions on test data

    Args:
        model_path: Path to the saved model file (.joblib)
        test_data: DataFrame containing test data

    Returns:
        Tuple of predicted classes and prediction probabilities
    """
    # Load the model
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
         print(f"ERROR: Model file not found at {model_path}")
         return None, None # Return None to indicate failure

    try:
        # Now Python knows what PyTorchModelWrapper and FoodProcessingNN are
        clf = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


    # --- Use the imported get_nutrient_columns ---
    # Get nutrient columns in the exact order used during training
    try:
        nutrient_cols = get_nutrient_columns()
    except NameError: # Fallback if import failed but script didn't exit (shouldn't happen with exit())
         print("ERROR: get_nutrient_columns function not found. Ensure it's imported correctly.")
         return None, None


    # --- Data Preparation ---
    # 1. Check for missing columns in test_data
    missing_test_cols = [col for col in nutrient_cols if col not in test_data.columns]
    if missing_test_cols:
        print(f"ERROR: The following required columns are missing in the test data: {missing_test_cols}")
        return None, None

    # 2. Select only the necessary columns in the correct order
    X_test = test_data[nutrient_cols]

    # 3. Handle Missing Values (NaNs) - CRITICAL STEP
    # Option A: Drop rows with any NaNs in the nutrient columns (simplest if acceptable)
    initial_rows = len(X_test)
    X_test_clean = X_test.dropna()
    rows_dropped = initial_rows - len(X_test_clean)
    if rows_dropped > 0:
        print(f"Warning: Dropped {rows_dropped} rows from test data due to NaN values in nutrient columns.")
    # Keep track of original indices if needed later
    original_indices = X_test_clean.index

    # Option B: Imputation (requires an imputer fitted on TRAINING data)
    # If you used imputation during training, you MUST load and use the SAME imputer here.
    # Example (if you saved an imputer):
    # imputer = joblib.load('path/to/your/saved_imputer.joblib')
    # X_test_imputed = imputer.transform(X_test)
    # X_test_clean = pd.DataFrame(X_test_imputed, columns=nutrient_cols, index=X_test.index)

    # ----> AVOID fillna(-20) <----
    # Filling with an arbitrary value like -20 AFTER the scaler was fitted during training
    # will likely lead to incorrect scaling and poor predictions. Handle NaNs *before* scaling.
    # X = test_data[nutrient_cols].fillna(-20) # <-- REMOVE THIS LINE

    if X_test_clean.empty:
        print("ERROR: No valid data remaining in the test set after handling NaNs.")
        return None, None

    # Make predictions using the cleaned data
    print(f"Making predictions on {len(X_test_clean)} samples...")
    try:
        y_pred = clf.predict(X_test_clean) # Pass the DataFrame directly
        y_pred_proba = clf.predict_proba(X_test_clean) # Pass the DataFrame directly
        # Return predictions along with the indices they correspond to
        return y_pred, y_pred_proba, original_indices
    except Exception as e:
        print(f"Error during prediction: {e}")
        # This might happen if the input data shape or type is wrong after cleaning
        return None, None


def evaluate_predictions(test_data: pd.DataFrame, y_pred: np.ndarray, y_pred_proba: np.ndarray, prediction_indices: pd.Index) -> pd.DataFrame:
    """
    Evaluate model predictions and return results, aligning with original data using indices.

    Args:
        test_data: Original full test data DataFrame.
        y_pred: Predicted NOVA classes.
        y_pred_proba: Prediction probabilities for each class.
        prediction_indices: The indices from the original test_data that correspond
                           to the y_pred and y_pred_proba arrays.

    Returns:
        DataFrame with evaluation results for the successfully predicted samples.
    """
    # Select the subset of test_data for which predictions were made
    test_data_evaluated = test_data.loc[prediction_indices]

    # Create results DataFrame
    results = pd.DataFrame({
        'Food Description': test_data_evaluated['Main food description'],
        'True NOVA Class': test_data_evaluated['nova-class'],
        'Predicted NOVA Class': y_pred,
        'P1 (Unprocessed)': y_pred_proba[:, 0],
        'P2 (Processed Culinary)': y_pred_proba[:, 1],
        'P3 (Processed)': y_pred_proba[:, 2],
        'P4 (Ultra-processed)': y_pred_proba[:, 3]
    }, index=prediction_indices) # Keep original index

    # Add correct/incorrect column
    results['Correct Prediction'] = results['True NOVA Class'] == results['Predicted NOVA Class']

    return results

def print_evaluation_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels = [0, 1, 2, 3]):
    """Print detailed evaluation metrics"""
    print("\n--- Evaluation Metrics ---")
    # Ensure consistent labels for report and matrix
    target_names = [f'NOVA {i+1}' for i in labels] # Assumes labels are 0, 1, 2, 3

    print("\nClassification Report:")
    # Use labels parameter to handle cases where some classes might be missing in y_true or y_pred
    print(classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0))

    print("\nConfusion Matrix:")
    # Rows: True Class, Columns: Predicted Class
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    print(f"Labels order: {labels}")
    print(conf_matrix)

    # Plot confusion matrix
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.title('Confusion Matrix')
        # Save or show the plot
        plt.savefig('confusion_matrix.png')
        print("\nConfusion matrix plot saved as confusion_matrix.png")
        # plt.show() # Uncomment to display the plot interactively
        plt.close() # Close the plot figure
    except Exception as e:
        print(f"\nWarning: Could not generate confusion matrix plot. Error: {e}")


    # Calculate overall accuracy
    # Use accuracy_score for clarity and robustness
    accuracy = accuracy_score(y_true, y_pred) * 100
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    print("-------------------------")

def main():
    # --- Configuration ---
    TEST_DATA_FILE = "food_test_data.csv"
    # !!! IMPORTANT: Update this path to your actual saved final model file !!!
    MODEL_PATH = "nova_pytorch_model_final_20250420_225335.joblib"  # <-- REPLACED WITH CORRECT FILENAME
    RESULTS_FILE = "evaluation_results.csv"
    # --- End Configuration ---


    # Load test data
    print(f"Loading test data from {TEST_DATA_FILE}...")
    try:
        test_data = pd.read_csv(TEST_DATA_FILE)
        print(f"Loaded {len(test_data)} test samples")
        # Basic validation
        if 'nova-class' not in test_data.columns:
            print(f"ERROR: Required column 'nova-class' not found in {TEST_DATA_FILE}")
            return
        if 'Main food description' not in test_data.columns:
            print(f"Warning: Column 'Main food description' not found in {TEST_DATA_FILE}. Results table will lack descriptions.")
            test_data['Main food description'] = 'N/A' # Add placeholder if missing
    except FileNotFoundError:
        print(f"ERROR: Test data file not found at {TEST_DATA_FILE}")
        return
    except Exception as e:
        print(f"Error loading test data: {e}")
        return

    # Load model and make predictions
    # The function now returns indices as well
    prediction_result = load_model_and_predict(MODEL_PATH, test_data)

    # Check if prediction was successful
    if prediction_result is None or prediction_result[0] is None:
        print("Failed to load model or make predictions. Exiting.")
        return
    y_pred, y_pred_proba, prediction_indices = prediction_result


    # Evaluate predictions using only the successfully predicted samples
    results = evaluate_predictions(test_data, y_pred, y_pred_proba, prediction_indices)

    # Extract true and predicted labels from the results DataFrame for metrics
    y_true_eval = results['True NOVA Class'].values
    y_pred_eval = results['Predicted NOVA Class'].values

    # Define the expected NOVA classes (adjust if your classes are different)
    nova_classes = sorted(test_data['nova-class'].unique()) # Get actual unique classes present
    print(f"Unique NOVA classes found in test set: {nova_classes}")


    # Print evaluation metrics using the evaluated subset
    print_evaluation_metrics(y_true_eval, y_pred_eval, labels=nova_classes)

    # Print some example predictions from the results DataFrame
    print("\nSample Predictions (from evaluated data):")
    print(results.head(10))

    # Save detailed results
    try:
        results.to_csv(RESULTS_FILE, index=False)
        print(f"\nDetailed results for {len(results)} evaluated samples saved to {RESULTS_FILE}")
    except Exception as e:
        print(f"Error saving results to {RESULTS_FILE}: {e}")

    # Print some incorrect predictions for manual review
    print("\nIncorrect Predictions (Examples):")
    incorrect = results[~results['Correct Prediction']].sort_values('Food Description')
    print(incorrect[['Food Description', 'True NOVA Class', 'Predicted NOVA Class']].head(10))

    # Print prediction distribution
    print("\nPrediction Distribution (among evaluated samples):")
    print(pd.value_counts(y_pred_eval, normalize=True).sort_index() * 100)

if __name__ == "__main__":
    main()
# --- END OF FILE ---