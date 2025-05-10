from plot import plot_nova_analysis
from mean import analyze_nutrients_by_nova_class
from randomizedata import split_and_save_dataset
import pandas as pd
from FoodProX import load_and_preprocess_data
from FoodProX import RandomForestClassifier
from FoodProX import predict_NOVA_class
from FoodProX import example_usage

# Load training data
df = load_and_preprocess_data('food_test_data.csv')

#Display the plot
results = analyze_nutrients_by_nova_class(df)
print(results)

# plot_nova_analysis(df)

# train_data, test_data = split_and_save_dataset(df)