import pandas as pd
import numpy as np

def analyze_nutrients_by_nova_class(df):
    # List of all nutrient columns (excluding 'Main food description' and 'nova-class')
    nutrient_columns = [
        'Protein', 'Total Fat', 'Carbohydrate', 'Alcohol', 'Water', 'Caffeine', 
        'Theobromine', 'Sugars, total', 'Fiber, total dietary', 'Calcium', 'Iron',
        'Magnesium', 'Phosphorus', 'Potassium', 'Sodium', 'Zinc', 'Copper', 
        'Selenium', 'Retinol', 'Carotene, beta', 'Vitamin E (alpha-tocopherol)',
        'Vitamin D (D2 + D3)', 'Cryptoxanthin, beta', 'Lycopene', 
        'Lutein + zeaxanthin', 'Vitamin C', 'Thiamin', 'Riboflavin', 'Niacin',
        'Vitamin B-6', 'Folate, total', 'Vitamin B-12', 'Choline, total',
        'Vitamin K (phylloquinone)', 'Folic acid', 'Folate, food', 
        'Vitamin E, added', 'Vitamin B-12, added', 'Cholesterol',
        'Fatty acids, total saturated', 'Fatty acids, total monounsaturated',
        'Fatty acids, total polyunsaturated'
    ]
    
    nutrient_means = df.groupby('nova-class')[nutrient_columns].mean()
    
    nutrient_means = np.exp(nutrient_means)
    
    nutrient_means = nutrient_means.round(7)
    
    class_counts = df['nova-class'].value_counts().sort_index()
    nutrient_means['Count_of_Items'] = class_counts
    
    return nutrient_means
