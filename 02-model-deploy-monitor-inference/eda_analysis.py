import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

class EDAAnalyzer:
    def __init__(self, data_path='data.csv', plots_dir='plots'):
        self.data_path = data_path
        self.plots_dir = plots_dir
        self.data = None
        
    def setup_plots_directory(self):
        if not os.path.exists(self.plots_dir):
            os.makedirs(self.plots_dir)
            
    def load_data(self):
        self.data = pd.read_csv(self.data_path)
        if 'User_ID' in self.data.columns:
            self.data = self.data.drop('User_ID', axis=1)
        print(f"Data loaded: {self.data.shape}")
        return self.data
        
    def basic_statistics(self):
        print("="*50)
        print("BASIC DATASET STATISTICS")
        print("="*50)
        
        print(f"Dataset shape: {self.data.shape}")
        print(f"Features: {list(self.data.columns)}")
        print(f"Data types:\n{self.data.dtypes}")
        
        print(f"\nMissing values:\n{self.data.isnull().sum()}")
        print(f"\nDuplicate rows: {self.data.duplicated().sum()}")
        
        print(f"\nNumerical features statistics:")
        print(self.data.describe())
        
        if 'Gender' in self.data.columns:
            print(f"\nGender distribution:\n{self.data['Gender'].value_counts()}")
            
    def plot_target_distribution(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].hist(self.data['Calories'], bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('Calories Distribution')
        axes[0, 0].set_xlabel('Calories')
        axes[0, 0].set_ylabel('Frequency')
        
        axes[0, 1].boxplot(self.data['Calories'])
        axes[0, 1].set_title('Calories Boxplot')
        axes[0, 1].set_ylabel('Calories')
        
        stats.probplot(self.data['Calories'], dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot for Calories')
        
        axes[1, 1].hist(np.log1p(self.data['Calories']), bins=50, alpha=0.7, color='green')
        axes[1, 1].set_title('Log-transformed Calories Distribution')
        axes[1, 1].set_xlabel('Log(Calories + 1)')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/target_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_feature_distributions(self):
        numerical_features = self.data.select_dtypes(include=[np.number]).columns
        numerical_features = [col for col in numerical_features if col != 'Calories']
        
        n_features = len(numerical_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, feature in enumerate(numerical_features):
            if self.data[feature].dtype == 'object':
                self.data[feature].value_counts().plot(kind='bar', ax=axes[i])
            else:
                axes[i].hist(self.data[feature], bins=30, alpha=0.7)
            axes[i].set_title(f'{feature} Distribution')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Frequency')
            
        for j in range(i + 1, len(axes)):
            axes[j].remove()
            
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_correlation_matrix(self):
        numerical_data = self.data.select_dtypes(include=[np.number])
        
        correlation_matrix = numerical_data.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_feature_target_relationships(self):
        numerical_features = self.data.select_dtypes(include=[np.number]).columns
        numerical_features = [col for col in numerical_features if col != 'Calories']
        
        n_features = len(numerical_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, feature in enumerate(numerical_features):
            axes[i].scatter(self.data[feature], self.data['Calories'], alpha=0.5)
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel('Calories')
            axes[i].set_title(f'{feature} vs Calories')
            
            z = np.polyfit(self.data[feature], self.data['Calories'], 1)
            p = np.poly1d(z)
            axes[i].plot(self.data[feature], p(self.data[feature]), "r--", alpha=0.8)
            
        for j in range(i + 1, len(axes)):
            axes[j].remove()
            
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/feature_target_relationships.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_categorical_analysis(self):
        if 'Gender' in self.data.columns:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            self.data['Gender'].value_counts().plot(kind='bar', ax=axes[0])
            axes[0].set_title('Gender Distribution')
            axes[0].set_xlabel('Gender')
            axes[0].set_ylabel('Count')
            
            self.data.boxplot(column='Calories', by='Gender', ax=axes[1])
            axes[1].set_title('Calories by Gender')
            axes[1].set_xlabel('Gender')
            axes[1].set_ylabel('Calories')
            
            gender_calories = self.data.groupby('Gender')['Calories'].mean()
            gender_calories.plot(kind='bar', ax=axes[2])
            axes[2].set_title('Average Calories by Gender')
            axes[2].set_xlabel('Gender')
            axes[2].set_ylabel('Average Calories')
            
            plt.tight_layout()
            plt.savefig(f'{self.plots_dir}/categorical_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
    def plot_outlier_analysis(self):
        numerical_features = self.data.select_dtypes(include=[np.number]).columns
        
        n_features = len(numerical_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 else axes
        
        for i, feature in enumerate(numerical_features):
            axes[i].boxplot(self.data[feature])
            axes[i].set_title(f'{feature} Outliers')
            axes[i].set_ylabel(feature)
            
        for j in range(i + 1, len(axes)):
            axes[j].remove()
            
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/outlier_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def calculate_feature_importance_correlation(self):
        numerical_data = self.data.select_dtypes(include=[np.number])
        target_correlations = numerical_data.corr()['Calories'].abs().sort_values(ascending=False)
        
        plt.figure(figsize=(10, 6))
        target_correlations[1:].plot(kind='bar')
        plt.title('Feature Importance (Correlation with Target)')
        plt.xlabel('Features')
        plt.ylabel('Absolute Correlation with Calories')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return target_correlations
        
    def generate_summary_statistics(self):
        summary = {
            'dataset_shape': self.data.shape,
            'missing_values': self.data.isnull().sum().to_dict(),
            'duplicates': int(self.data.duplicated().sum()),
            'numerical_summary': self.data.describe().to_dict(),
        }
        
        if 'Gender' in self.data.columns:
            summary['gender_distribution'] = self.data['Gender'].value_counts().to_dict()
            
        return summary
        
    def run_full_eda(self):
        print("Starting Exploratory Data Analysis...")
        
        self.setup_plots_directory()
        self.load_data()
        self.basic_statistics()
        
        print("\nGenerating visualization plots...")
        self.plot_target_distribution()
        self.plot_feature_distributions()
        self.plot_correlation_matrix()
        self.plot_feature_target_relationships()
        self.plot_categorical_analysis()
        self.plot_outlier_analysis()
        
        print("\nCalculating feature importance...")
        correlations = self.calculate_feature_importance_correlation()
        
        print("\nGenerating summary statistics...")
        summary = self.generate_summary_statistics()
        
        print(f"\nEDA completed. Plots saved to '{self.plots_dir}' directory.")
        print(f"Generated plots:")
        print("- target_analysis.png")
        print("- feature_distributions.png") 
        print("- correlation_matrix.png")
        print("- feature_target_relationships.png")
        print("- categorical_analysis.png")
        print("- outlier_analysis.png")
        print("- feature_importance.png")
        
        return summary, correlations

def main():
    eda = EDAAnalyzer()
    summary, correlations = eda.run_full_eda()
    return summary, correlations

if __name__ == "__main__":
    main()