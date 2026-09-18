import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow import keras
from tensorflow.keras import layers
import os

os.makedirs('resultados', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("="*70)
print("P vs NP: ML Predicts TSP Hardness")
print("="*70)

# GENERADOR TSP SINTÉTICO
class TSPGenerator:
    def __init__(self, n_cities=50, topology='random', seed=42):
        self.n_cities = n_cities
        self.topology = topology
        np.random.seed(seed)
    
    def generate(self):
        if self.topology == 'random':
            return np.random.uniform(0, 100, (self.n_cities, 2))
        elif self.topology == 'clustered':
            n_clusters = max(2, self.n_cities // 10)
            centers = np.random.uniform(0, 100, (n_clusters, 2))
            cities = []
            for i in range(self.n_cities):
                center = centers[i % n_clusters]
                city = center + np.random.normal(0, 5, 2)
                cities.append(city)
            return np.array(cities)
        elif self.topology == 'grid':
            side = int(np.sqrt(self.n_cities))
            x = np.repeat(np.linspace(0, 100, side), side)[:self.n_cities]
            y = np.tile(np.linspace(0, 100, side), side)[:self.n_cities]
            return np.column_stack([x, y])

# EXTRACTOR DE FEATURES
class FeatureExtractor:
    def __init__(self, coordinates):
        self.coordinates = np.array(coordinates)
        self.n = len(self.coordinates)
        from scipy.spatial.distance import pdist, squareform
        distances = pdist(self.coordinates, metric='euclidean')
        self.distances = squareform(distances)
    
    def extract_all(self):
        features = {}
        features['n_cities'] = self.n
        
        dist_array = self.distances[np.triu_indices_from(self.distances, k=1)]
        features['mean_distance'] = np.mean(dist_array)
        features['std_distance'] = np.std(dist_array)
        features['max_distance'] = np.max(dist_array)
        
        from scipy import stats
        features['kurtosis'] = stats.kurtosis(dist_array)
        
        # Clustering
        from sklearn.neighbors import NearestNeighbors
        try:
            nbrs = NearestNeighbors(n_neighbors=min(5, self.n-1))
            nbrs.fit(self.coordinates)
            distances_nn, _ = nbrs.kneighbors(self.coordinates)
            features['clustering'] = np.mean(distances_nn[:, -1])
        except:
            features['clustering'] = 0.0
        
        # Hull ratio
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(self.coordinates)
            features['hull_ratio'] = len(hull.vertices) / self.n
        except:
            features['hull_ratio'] = 0.0
        
        # PCA dimension
        from sklearn.decomposition import PCA
        if self.n < 2:
            features['intrinsic_dim'] = 1
        else:
            pca = PCA()
            pca.fit(self.coordinates)
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            features['intrinsic_dim'] = np.argmax(cumsum >= 0.95) + 1
        
        # K-means silhouette
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        for k in [2, 3, 4, 5]:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(self.coordinates)
                score = silhouette_score(self.coordinates, labels)
                features[f'silhouette_k{k}'] = score
            except:
                features[f'silhouette_k{k}'] = 0.0
        
        # Roughness
        D_row = np.sum(self.distances > 0, axis=1)
        features['roughness'] = np.std(D_row)
        
        # NN ratio
        nn_distances = []
        for i in range(self.n):
            nearest = np.min(self.distances[i, self.distances[i] > 0])
            nn_distances.append(nearest)
        features['nn_ratio'] = np.mean(nn_distances) / np.mean(dist_array)
        
        # Density
        features['density'] = self.n / (100 * 100)
        
        return features

# SIMULADOR DE DUREZA
class HardnessSimulator:
    def __init__(self, features):
        self.features = features
    
    def predict_hardness(self):
        n = self.features['n_cities']
        clustering = self.features['clustering']
        kurtosis = self.features['kurtosis']
        
        complexity = (
            0.3 * (np.log(n) / 4) +
            0.4 * (clustering / 20) +
            0.3 * (abs(kurtosis) / 10)
        )
        
        if complexity < 0.3:
            return 0, np.random.uniform(0.1, 0.5)
        elif complexity < 0.5:
            return 1, np.random.uniform(1, 10)
        elif complexity < 0.7:
            return 2, np.random.uniform(10, 100)
        else:
            return 3, np.random.uniform(100, 300)

# GENERAR DATASET
print("\n[PASO 1/5] Generando dataset...")
data = []
topologies = ['random', 'clustered', 'grid']

for i in range(400):
    topology = topologies[i % 3]
    n_cities = np.random.randint(30, 150)
    
    generator = TSPGenerator(n_cities, topology)
    coordinates = generator.generate()
    
    extractor = FeatureExtractor(coordinates)
    features = extractor.extract_all()
    
    simulator = HardnessSimulator(features)
    hardness, time_taken = simulator.predict_hardness()
    
    features['hardness'] = hardness
    features['solver_time'] = time_taken
    features['topology'] = topology
    
    data.append(features)
    
    if (i + 1) % 100 == 0:
        print(f"  ✓ {i + 1}/400")

df = pd.DataFrame(data)
df.to_csv('data/dataset_tsp.csv', index=False)
print(f"✓ Dataset: {len(df)} instancias")

# PREPARAR DATOS
print("\n[PASO 2/5] Preparando datos...")
feature_cols = [col for col in df.columns if col not in 
                ['hardness', 'solver_time', 'topology']]
X = df[feature_cols].values
y = df['hardness'].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✓ Train: {len(X_train)} | Test: {len(X_test)}")

# ENTRENAR MODELOS
print("\n[PASO 3/5] Entrenando modelos...")

# kNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
print(f"  kNN Accuracy: {acc_knn:.4f}")

# Random Forest
forest = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
forest.fit(X_train, y_train)
y_pred_forest = forest.predict(X_test)
acc_forest = accuracy_score(y_test, y_pred_forest)
print(f"  Forest Accuracy: {acc_forest:.4f}")

# MLP
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(4, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_split=0.2, epochs=30, batch_size=32, 
          callbacks=[keras.callbacks.EarlyStopping(patience=5)], verbose=0)
y_pred_mlp = model.predict(X_test, verbose=0).argmax(axis=1)
acc_mlp = accuracy_score(y_test, y_pred_mlp)
print(f"  MLP Accuracy: {acc_mlp:.4f}")

# VISUALIZACIONES
print("\n[PASO 4/5] Creando gráficos...")

# 1. Distribución de dureza
fig, ax = plt.subplots(figsize=(10, 5))
hardness_dist = df['hardness'].value_counts().sort_index()
colors = ['green', 'yellow', 'orange', 'red']
ax.bar(hardness_dist.index, hardness_dist.values, color=colors, alpha=0.7)
ax.set_xlabel("Categoría de Dureza")
ax.set_ylabel("Cantidad")
ax.set_title("Distribución de Dureza en Dataset")
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(['Trivial', 'Fácil', 'Moderado', 'Duro'])
plt.tight_layout()
plt.savefig('resultados/01_hardness_distribution.png', dpi=300, bbox_inches='tight')
print("  ✓ 01_hardness_distribution.png")

# 2. Feature Importance
importance = forest.feature_importances_
top_idx = np.argsort(importance)[-10:]
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(np.array(feature_cols)[top_idx], importance[top_idx], color='steelblue')
ax.set_xlabel("Importancia")
ax.set_title("Top 10 Features (Random Forest)")
plt.tight_layout()
plt.savefig('resultados/02_feature_importance.png', dpi=300, bbox_inches='tight')
print("  ✓ 02_feature_importance.png")

# 3. Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, (name, y_pred) in enumerate([('kNN', y_pred_knn), ('Forest', y_pred_forest), ('MLP', y_pred_mlp)]):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
    axes[idx].set_title(f"{name}")
    axes[idx].set_ylabel("True")
    axes[idx].set_xlabel("Predicted")
plt.tight_layout()
plt.savefig('resultados/03_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("  ✓ 03_confusion_matrices.png")

# 4. Comparación de modelos
fig, ax = plt.subplots(figsize=(10, 5))
models = ['kNN', 'Forest', 'MLP']
accuracies = [acc_knn, acc_forest, acc_mlp]
colors_model = ['blue', 'green', 'red']
ax.bar(models, accuracies, color=colors_model, alpha=0.7)
ax.set_ylabel("Accuracy")
ax.set_title("Comparación de Modelos")
ax.set_ylim([0.7, 0.9])
for i, acc in enumerate(accuracies):
    ax.text(i, acc + 0.01, f"{acc:.2%}", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('resultados/04_model_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ 04_model_comparison.png")

# GUARDAR MODELOS
print("\n[PASO 5/5] Guardando modelos...")
import pickle
os.makedirs('models', exist_ok=True)
with open('models/model_forest.pkl', 'wb') as f:
    pickle.dump(forest, f)
with open('models/model_knn.pkl', 'wb') as f:
    pickle.dump(knn, f)
model.save('models/model_mlp.h5')
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_cols, f)
print("✓ Modelos guardados")

print("\n" + "="*70)
print("✅ PIPELINE COMPLETADO")
print("="*70)
print("\nResultados:")
print(f"  Dataset: data/dataset_tsp.csv")
print(f"  Gráficos: resultados/")
print(f"  Modelos: models/")
print(f"\nAccuracy:")
print(f"  kNN:    {acc_knn:.2%}")
print(f"  Forest: {acc_forest:.2%}")
print(f"  MLP:    {acc_mlp:.2%}")
print("\nProximo paso: Crear app Streamlit")