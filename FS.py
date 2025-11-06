import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score
import pyswarms as ps

# Load the dataset
excel_file_path = '/content/drive/MyDrive/Colab-Notebooks/data.xlsx'
df = pd.read_excel(excel_file_path)

# Display the first few rows to understand the data structure
print(df.head())

# Separate the features (X) and target variable (y)
X = df.drop('Default_or_not', axis=1).values  # All columns except target
y = df['Default_or_not'].values  # Target column

# Check data shapes
print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

# Function to calculate Information Value (IV) for each feature
def calc_iv(X, y):
    iv_dict = {}
    for i in range(X.shape[1]):
        feature = X[:, i]
        feature_name = f'Feature {i+1}'
        df = pd.DataFrame({'X': feature, 'y': y})
        df['bin'] = pd.qcut(df['X'], q=5, labels=False)  # Discretize using quartiles
        grouped = df.groupby('bin').agg({'y': ['count', 'sum']}).reset_index()
        grouped.columns = ['bin', 'total', 'positive']
        grouped['negative'] = grouped['total'] - grouped['positive']
        grouped['woe'] = np.log(grouped['positive'] / grouped['negative'])
        iv = ((grouped['positive'] / grouped['total']) - (grouped['negative'] / grouped['total'])) * grouped['woe']
        iv_dict[feature_name] = iv.sum()
    return iv_dict

# Objective function for SMPSO
def objective_function(swarm, X, y):
    fitness = []
    for i in range(swarm.shape[0]):
        selected_features = np.where(swarm[i] == 1)[0]  # Selected feature indices
        if len(selected_features) == 0:
            fitness.append(np.inf)  # Penalize empty solutions
            continue
        X_selected = X[:, selected_features]
        
        # Train a classifier (KNN in this case) and calculate accuracy
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        
        # Calculate accuracy and F1 Score
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate Information Value (IV) of selected features
        iv_dict = calc_iv(X_selected, y)
        iv_score = sum(iv_dict.values())  # Sum of IV values for selected features
        
        # Combine the objectives: we minimize #features, maximize IV, and maximize accuracy/F1
        fitness_value = -accuracy + iv_score  # Negative because we minimize fitness in PSO
        fitness.append(fitness_value)
    return np.array(fitness)

# Main function to execute the feature selection and optimization
def feature_selection(X, y, num_features=5, max_iter=100, num_particles=10):
    # Standardizing features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Balancing the dataset using SMOTE
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    # Check if resampled data shape is correct
    print(f"Shape of resampled X: {X_resampled.shape}, Shape of resampled y: {y_resampled.shape}")
    
    # Set PSO parameters: n_particles, dimensions, bounds (binary: 0 or 1)
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}  # Parameters for PSO
    bounds = (np.zeros(X_resampled.shape[1]), np.ones(X_resampled.shape[1]))  # Binary features
    
    # Initialize the PSO optimizer
    optimizer = ps.single.GlobalBestPSO(n_particles=num_particles, dimensions=X_resampled.shape[1], options=options, bounds=bounds)
    
    # Run PSO optimization
    cost, pos = optimizer.optimize(objective_function, iters=max_iter, X=X_resampled, y=y_resampled)
    
    # Get the best subset of features
    best_features = np.where(pos == 1)[0]
    print(f"Best subset of features: {best_features}")
    
    # Train final model on best features
    X_best = X_resampled[:, best_features]
    X_train, X_test, y_train, y_test = train_test_split(X_best, y_resampled, test_size=0.3, random_state=42)
    
    # Retrain model on best features
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    # Print final evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Final Accuracy: {accuracy:.4f}, Final F1 Score: {f1:.4f}")
    
    return best_features, accuracy, f1

# Run the feature selection process
if __name__ == "__main__":
    best_features, accuracy, f1 = feature_selection(X, y, num_features=5)
    print(f"Selected Features: {best_features}")

