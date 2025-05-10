import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_nova_analysis(df):
    # Set the style    
    # Create a figure with subplots
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Distribution of foods across NOVA classes
    plt.subplot(2, 2, 1)
    sns.countplot(data=df, x='nova-class')
    plt.title('Distribution of Foods Across NOVA Classes')
    plt.xlabel('NOVA Class')
    plt.ylabel('Count of Food Items')

    # Adjust layout
    plt.tight_layout()
    plt.show()
