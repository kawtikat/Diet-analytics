import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import os

# Set the style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

def load_data(file_path):
    """Load the dataset and print basic information"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded with shape: {df.shape}")
    return df

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

def create_output_dir(dir_name="analysis_results"):
    """Create directory for saving analysis results"""
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return dir_name

def analyze_nova_distribution(df, output_dir):
    """Analyze and visualize the distribution of NOVA classes"""
    print("\n=== NOVA Class Distribution Analysis ===")
    
    # Count distribution
    nova_counts = df['nova-class'].value_counts().sort_index()
    print("NOVA Class Distribution:")
    for nova_class, count in nova_counts.items():
        print(f"NOVA {nova_class}: {count} items ({count/len(df):.1%})")
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x='nova-class', data=df, palette="viridis")
    
    # Add count labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha = 'center', va = 'bottom', 
                   xytext = (0, 5), textcoords = 'offset points')
    
    plt.title('Distribution of NOVA Classes', fontsize=15)
    plt.xlabel('NOVA Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"{output_dir}/nova_distribution.png", dpi=300)
    print(f"NOVA distribution plot saved to {output_dir}/nova_distribution.png")
    
    # Create percentage pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(nova_counts, labels=[f"NOVA {i}" for i in nova_counts.index], 
            autopct='%1.1f%%', startangle=90, shadow=True, 
            explode=[0.05] * len(nova_counts), colors=sns.color_palette("viridis", len(nova_counts)))
    plt.title('Percentage Distribution of NOVA Classes', fontsize=15)
    plt.tight_layout()
    
    # Save the pie chart
    plt.savefig(f"{output_dir}/nova_pie_chart.png", dpi=300)
    print(f"NOVA pie chart saved to {output_dir}/nova_pie_chart.png")

def analyze_food_descriptions(df, output_dir):
    """Analyze food descriptions by NOVA class"""
    print("\n=== Food Description Analysis by NOVA Class ===")
    
    # Get most common foods in each NOVA class
    top_n = 10
    
    for nova_class in sorted(df['nova-class'].unique()):
        class_foods = df[df['nova-class'] == nova_class]['Main food description'].value_counts().head(top_n)
        print(f"\nTop {top_n} foods in NOVA {nova_class}:")
        for food, count in class_foods.items():
            print(f"  - {food}: {count}")
    
    # Create a summary file of foods by NOVA class
    with open(f"{output_dir}/foods_by_nova_class.txt", "w") as f:
        for nova_class in sorted(df['nova-class'].unique()):
            class_foods = df[df['nova-class'] == nova_class]['Main food description'].value_counts()
            f.write(f"\n*** NOVA {nova_class} Foods ***\n")
            for food, count in class_foods.items():
                f.write(f"{food}: {count}\n")
    
    print(f"Food descriptions by NOVA class saved to {output_dir}/foods_by_nova_class.txt")

def analyze_nutrient_profiles(df, output_dir):
    """Analyze nutrient profiles across NOVA classes"""
    print("\n=== Nutrient Profile Analysis ===")
    
    # Get nutrient columns
    nutrient_cols = get_nutrient_columns()
    
    # Calculate mean nutrient values by NOVA class
    nutrient_means = df.groupby('nova-class')[nutrient_cols].mean()
    
    # Save nutrient profiles to CSV
    nutrient_means.to_csv(f"{output_dir}/nutrient_profiles_by_nova.csv")
    print(f"Nutrient profiles saved to {output_dir}/nutrient_profiles_by_nova.csv")
    
    # Create heatmap of nutrient differences
    plt.figure(figsize=(14, 10))
    # Standardize for better visualization
    nutrient_means_std = (nutrient_means - nutrient_means.mean()) / nutrient_means.std()
    
    # Create heatmap
    sns.heatmap(nutrient_means_std, cmap="RdBu_r", center=0, 
                annot=False, linewidths=.5)
    plt.title('Standardized Nutrient Profiles by NOVA Class', fontsize=15)
    plt.tight_layout()
    
    # Save heatmap
    plt.savefig(f"{output_dir}/nutrient_heatmap.png", dpi=300)
    print(f"Nutrient heatmap saved to {output_dir}/nutrient_heatmap.png")
    
    # Create plots for top differentiating nutrients
    # Calculate nutrient variance across NOVA groups
    nutrient_variance = nutrient_means.var(axis=0).sort_values(ascending=False)
    top_nutrients = nutrient_variance.head(10).index
    
    plt.figure(figsize=(12, 8))
    for i, nutrient in enumerate(top_nutrients):
        plt.subplot(3, 4, i+1)
        sns.boxplot(x='nova-class', y=nutrient, data=df)
        plt.title(nutrient, fontsize=10)
        plt.tight_layout()
    
    # Save boxplots
    plt.savefig(f"{output_dir}/top_nutrient_boxplots.png", dpi=300)
    print(f"Top nutrient boxplots saved to {output_dir}/top_nutrient_boxplots.png")

def perform_pca_analysis(df, output_dir):
    """Perform PCA to visualize nutrient patterns across NOVA classes"""
    print("\n=== PCA Analysis ===")
    
    # Get nutrient columns
    nutrient_cols = get_nutrient_columns()
    
    # Prepare data
    X = df[nutrient_cols].copy().fillna(-20)  # Log-transformed data
    y = df['nova-class']
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create PCA visualization
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', 
                         alpha=0.6, edgecolors='w', linewidth=0.5)
    
    plt.title('PCA of Nutrient Profiles Colored by NOVA Class', fontsize=15)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    
    # Add legend
    legend = plt.legend(*scatter.legend_elements(), 
                        title="NOVA Class", loc="best")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_visualization.png", dpi=300)
    print(f"PCA visualization saved to {output_dir}/pca_visualization.png")
    
    # Get feature importance in PCA
    pca_components = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(2)],
        index=nutrient_cols
    )
    
    # Save PCA components
    pca_components.to_csv(f"{output_dir}/pca_components.csv")
    print(f"PCA components saved to {output_dir}/pca_components.csv")
    
    # Visualize top contributing features
    plt.figure(figsize=(10, 8))
    
    # For PC1
    pc1_contributions = pca_components['PC1'].abs().sort_values(ascending=False).head(10)
    plt.subplot(1, 2, 1)
    pc1_contributions.plot(kind='barh')
    plt.title('Top 10 Features in PC1', fontsize=12)
    
    # For PC2
    pc2_contributions = pca_components['PC2'].abs().sort_values(ascending=False).head(10)
    plt.subplot(1, 2, 2)
    pc2_contributions.plot(kind='barh')
    plt.title('Top 10 Features in PC2', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_feature_contributions.png", dpi=300)
    print(f"PCA feature contributions saved to {output_dir}/pca_feature_contributions.png")

def evaluate_model_performance(df, output_dir):
    """Evaluate Random Forest model performance on the dataset"""
    print("\n=== Model Performance Evaluation ===")
    
    # Get nutrient columns
    nutrient_cols = get_nutrient_columns()
    
    # Prepare data
    X = df[nutrient_cols].copy().fillna(-20)  # Log-transformed data
    y = df['nova-class']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train model
    print("Training Random Forest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Classification report
    class_report = classification_report(y_test, y_pred)
    print("\nClassification Report:")
    print(class_report)
    
    # Save classification report
    with open(f"{output_dir}/classification_report.txt", "w") as f:
        f.write("Classification Report:\n")
        f.write(class_report)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=[f"NOVA {i}" for i in sorted(y.unique())],
               yticklabels=[f"NOVA {i}" for i in sorted(y.unique())])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix', fontsize=15)
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300)
    print(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")
    
    # Feature importance
    feature_imp = pd.DataFrame({
        'Feature': nutrient_cols,
        'Importance': clf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # Save feature importance
    feature_imp.to_csv(f"{output_dir}/feature_importance.csv", index=False)
    
    # Plot top features
    top_features = feature_imp.head(15)
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('Top 15 Features by Importance', fontsize=15)
    plt.tight_layout()
    
    plt.savefig(f"{output_dir}/feature_importance.png", dpi=300)
    print(f"Feature importance plot saved to {output_dir}/feature_importance.png")

def main():
    # Set up output directory
    output_dir = create_output_dir()
    
    # Load data
    df = load_data("food_train_data.csv")
    
    # Perform analyses
    analyze_nova_distribution(df, output_dir)
    analyze_food_descriptions(df, output_dir)
    analyze_nutrient_profiles(df, output_dir)
    perform_pca_analysis(df, output_dir)
    evaluate_model_performance(df, output_dir)
    
    print("\nAnalysis complete! All results saved to:", output_dir)

if __name__ == "__main__":
    main()