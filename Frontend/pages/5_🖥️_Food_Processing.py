import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix
import base64
from io import BytesIO
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="NOVA Food Classification",
    layout="wide"
)

# Define PyTorch model class so it can be loaded from the joblib file
class FoodProcessingNN(nn.Module):
    """Neural network for food processing classification"""
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.3):
        super(FoodProcessingNN, self).__init__()
        
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

# Define the wrapper class needed for loading the model
class PyTorchModelWrapper:
    """Wrapper class to make PyTorch model compatible with sklearn interface"""
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler
    
    def predict(self, X):
        """Predict class labels"""
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.numpy()
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
        
        return probabilities.numpy()

# Define the nutrient columns needed for the model
def get_nutrient_columns():
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

# Function to calculate FPro score
def calculate_fpro(results):
    """
    Calculate the Food Processing Score (FPro) for each food item
    
    Args:
        results: DataFrame with p0_unprocessed and p3_ultra_processed probability columns
        
    Returns:
        DataFrame with added FPro column
    """
    results_with_fpro = results.copy()
    # FPro = (1 - p1 + p4) / 2, where p1 is unprocessed probability and p4 is ultra-processed probability
    # In our case, p0_unprocessed corresponds to p1 and p3_ultra_processed corresponds to p4
    results_with_fpro['FPro'] = (1 - results_with_fpro['p0_unprocessed'] + results_with_fpro['p3_ultra_processed']) / 2
    # Format FPro to 5 decimal places
    results_with_fpro['FPro'] = results_with_fpro['FPro'].round(5)
    return results_with_fpro

# Function to load the model
@st.cache_resource
def load_model(model_path):
    """Load the trained model"""
    return joblib.load(model_path)

# Function to predict NOVA class
def predict_nova_class(model, food_data):
    """
    Predict NOVA classification for food items
    
    Args:
        model: Trained model
        food_data: DataFrame with nutrient data
        
    Returns:
        DataFrame with predictions and probabilities
    """
    # Get required nutrient columns
    nutrient_cols = get_nutrient_columns()
    
    # Check if all required columns are present
    missing_cols = [col for col in nutrient_cols if col not in food_data.columns]
    if missing_cols:
        st.error(f"Missing required nutrient columns: {', '.join(missing_cols)}")
        st.stop()
    
    # Prepare features
    X = food_data[nutrient_cols].copy()
    X = X.fillna(-20)  # Fill missing values with -20 (log transformed zero)
    
    # Make predictions
    pred_class = model.predict(X)
    pred_proba = model.predict_proba(X)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'predicted_nova_class': pred_class,
        'p0_unprocessed': pred_proba[:, 0],
        'p1_processed_culinary': pred_proba[:, 1],
        'p2_processed': pred_proba[:, 2],
        'p3_ultra_processed': pred_proba[:, 3]
    })
    
    # Add food description if available
    if 'Main food description' in food_data.columns:
        results.insert(0, 'Main food description', food_data['Main food description'])
    elif 'Column1' in food_data.columns:  # Handle common alternate column name
        results.insert(0, 'Main food description', food_data['Column1'])
    
    # Calculate FPro score
    results = calculate_fpro(results)
    
    return results

# Function to plot probability distribution
def plot_probability_distribution(results):
    """Plot the probability distribution for each NOVA class"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Melt the DataFrame for easier plotting
    prob_cols = ['p0_unprocessed', 'p1_processed_culinary', 'p2_processed', 'p3_ultra_processed']
    plot_data = results[['Main food description'] + prob_cols].melt(
        id_vars='Main food description',
        var_name='NOVA Class',
        value_name='Probability'
    )
    
    # Rename classes for better readability
    class_names = {
        'p0_unprocessed': 'NOVA 0: Unprocessed',
        'p1_processed_culinary': 'NOVA 1: Processed Culinary',
        'p2_processed': 'NOVA 2: Processed',
        'p3_ultra_processed': 'NOVA 3: Ultra-processed'
    }
    plot_data['NOVA Class'] = plot_data['NOVA Class'].map(class_names)
    
    # Create the bar chart
    sns.barplot(x='Main food description', y='Probability', hue='NOVA Class', data=plot_data, ax=ax)
    ax.set_title('Probability Distribution by NOVA Class')
    ax.set_xlabel('Food Item')
    ax.set_ylabel('Probability')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title='NOVA Classification')
    
    plt.tight_layout()
    return fig

# Function to plot FPro scores
def plot_fpro_scores(results):
    """Create a bar chart of FPro scores"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort by FPro score for better visualization
    plot_data = results.sort_values('FPro', ascending=False)
    
    # Create color mapping based on FPro score
    colors = plt.cm.RdYlBu_r(plot_data['FPro'])
    
    # Create the bar chart
    bars = ax.bar(plot_data['Main food description'], plot_data['FPro'], color=colors)
    
    # Add a color bar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Food Processing Score (FPro)')
    
    ax.set_title('Food Processing Score (FPro) by Food Item')
    ax.set_xlabel('Food Item')
    ax.set_ylabel('FPro Score (0-1)')
    ax.set_ylim(0, 1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

# Main application function
def main():
    st.title("🍎 FoodProX - NOVA Food Classification & FPro Score")
        
    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    
    # Get list of available models
    model_dir = "Frontend/models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib')] if os.path.exists(model_dir) else []
    
    # Option to select existing model or upload a new one
    model_path = None
    
    # File uploader for model
    uploaded_model = st.sidebar.file_uploader("Upload a model file (.joblib)", type="joblib")
    
    if uploaded_model is not None:
        # Save the uploaded model
        model_save_path = os.path.join(model_dir, uploaded_model.name)
        with open(model_save_path, "wb") as f:
            f.write(uploaded_model.getbuffer())
        model_path = model_save_path
        st.sidebar.success(f"Model uploaded successfully: {uploaded_model.name}")
    elif model_files:
        # Select from existing models if available
        selected_model = st.sidebar.selectbox("Or select an existing model", [""] + model_files)
        if selected_model:
            model_path = os.path.join(model_dir, selected_model)
                    
    uploaded_file = st.file_uploader("Upload a CSV file with nutrient data", type="csv")
    if uploaded_file:
        try:
            data = pd.read_csv(uploaded_file)            
            # Display preview of the uploaded data
            st.subheader("Data Preview")
            st.dataframe(data.head())
            
            # Process the data if a model is selected
            if model_path:
                process_data(model_path, data)
            else:
                st.warning("Please select or upload a model file first")
        except Exception as e:
            st.error(f"Error processing the file: {e}")

# Function to process data and display results
def process_data(model_path, data):
    """Process data with the model and display results"""
    # Load the model
    model = load_model(model_path)
        
    # Make predictions and calculate FPro
    results = predict_nova_class(model, data)
    
    # Display results
    st.subheader("Prediction Results")
    
    # Add human-readable class names
    nova_class_names = {
        0: "NOVA 0: Unprocessed or minimally processed foods",
        1: "NOVA 1: Processed culinary ingredients",
        2: "NOVA 2: Processed foods",
        3: "NOVA 3: Ultra-processed foods"
    }
    
    display_results = results.copy()
    display_results['NOVA Class'] = display_results['predicted_nova_class'].astype(int).map(nova_class_names)
    
    # Show results in a table with FPro score
    st.dataframe(display_results[['Main food description', 'NOVA Class', 'FPro', 
                                'p0_unprocessed', 'p1_processed_culinary',
                                'p2_processed', 'p3_ultra_processed']])
        
    # Plot NOVA probability distribution
    st.subheader("NOVA Class Probability Distribution")
    max_display = min(10, len(results))  # Limit to avoid overcrowding
    fig = plot_probability_distribution(results.head(max_display))
    st.pyplot(fig)

    # Plot FPro scores
    st.subheader("Food Processing Score (FPro)")
    fig_fpro = plot_fpro_scores(results.head(max_display))
    st.pyplot(fig_fpro)
    
    
if __name__ == "__main__":
    main()