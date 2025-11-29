import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import pickle
import warnings
#Suppress future warnings for cleaner output
warnings.filterwarnings("ignore")

# 1. Load Data
# Assuming all necessary CSV files (rooms.csv, bookings.csv, branches.csv, guests.csv) are in the same directory.
try:
    rooms    = pd.read_csv("rooms.csv")
    bookings = pd.read_csv("bookings.csv")
    branches = pd.read_csv("branches.csv")
    guests   = pd.read_csv("guests.csv")
except FileNotFoundError as e:
    print(f"Error loading required file: {e}. Please ensure all CSV files are present.")
    exit()


print(f"Loaded: {len(rooms)} rooms, {len(bookings)} bookings, {len(guests)} guests, {len(branches)} branches")

# 2. Data Cleaning
# Convert check-in/out to datetime and calculate stay duration
bookings['check_in']  = pd.to_datetime(bookings['check_in'], errors='coerce')
bookings['check_out'] = pd.to_datetime(bookings['check_out'], errors='coerce')
bookings['stay_days'] = (bookings['check_out'] - bookings['check_in']).dt.days

# 3. Room-Level Stats (Aggregation)
stats = bookings.groupby('room_id').agg(
    booking_count=('booking_id', 'count'),
    total_revenue=('payment', 'sum'),
    avg_payment=('payment', 'mean'),
    avg_stay=('stay_days', 'mean')
).reset_index()

# Merge aggregated stats with room data
df = rooms.merge(stats, on='room_id', how='left')
df = df.merge(branches[['Branch_id', 'Branch_name']], on='Branch_id', how='left')

# Fill missing values (rooms with no bookings)
for col in ['booking_count','total_revenue','avg_payment','avg_stay']:
    df[col].fillna(0, inplace=True)
df['Branch_name'].fillna('Unknown', inplace=True)

# 4. Feature Engineering
# Create new derived metrics
df['revenue_per_booking'] = df['total_revenue'] / (df['booking_count'] + 1)
df['revenue_per_day']     = df['total_revenue'] / (df['avg_stay'] + 1)
# Create a binary flag for premium room types
df['is_premium']          = df['type'].isin(['Suite','Deluxe','Presidential']).astype(int)

# Apply Log transforms (essential for normal distribution and model performance)
df['log_price']       = np.log1p(df['price'])
df['log_revenue']     = np.log1p(df['total_revenue'])
df['log_bookings']    = np.log1p(df['booking_count'])
df['log_avg_stay']    = np.log1p(df['avg_stay'])
df['log_rev_per_day'] = np.log1p(df['revenue_per_day'])

# List of final features used for clustering
features = [
    'log_price', 'log_revenue', 'log_bookings',
    'log_avg_stay', 'log_rev_per_day', 'is_premium'
]

X = df[features].copy()

# 5. Remove Outliers using Isolation Forest
iso = IsolationForest(contamination='auto', n_estimators=300, random_state=42)
mask = iso.fit_predict(X) == 1
df_clean = df[mask].reset_index(drop=True)
X_clean  = X[mask].reset_index(drop=True)

print(f"\nRemoved Outliers → Remaining Rooms: {len(df_clean)}")

# 6. Scaling (Standardize the data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

# 7. Train/Test Split (for validation purposes)
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")

# 8. Best K search using Silhouette Score
best_score = -1
best_k = 3

print("\nSearching for best number of clusters (K)")

# We must use the 'X_train' subset for the search to simulate generalization
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k, n_init=100, random_state=42)
    labels_train = kmeans.fit_predict(X_train)
    score = silhouette_score(X_train, labels_train)

    print(f"   k={k} → Silhouette Score (TRAIN) = {score:.4f}")

    if score > best_score:
        best_score = score
        best_k = k

print(f"\nBEST RESULT → k={best_k} | Silhouette Score (TRAIN) = {best_score:.4f}")

# 9. Train final KMeans on all clean scaled data
kmeans_final = KMeans(n_clusters=best_k, n_init=100, random_state=42)
kmeans_final.fit(X_scaled) # Fit on the entire clean dataset (X_scaled)

# Assign cluster labels to the clean dataset
df_clean['cluster'] = kmeans_final.predict(X_scaled)

# 10. Train/Test Results (Validation)
train_labels = kmeans_final.predict(X_train)
test_labels  = kmeans_final.predict(X_test)
train_sil = silhouette_score(X_train, train_labels)
test_sil  = silhouette_score(X_test, test_labels)

print("\nTRAIN / TEST CLUSTER ACCURACY")
print(f"Train Silhouette Score = {train_sil:.4f}")
print(f"Test  Silhouette Score = {test_sil:.4f}")

# 11. Naming clusters (Mapping results back to original data for business understanding)
cluster_summary = df_clean.groupby('cluster').agg({
    'total_revenue': 'mean',
    'price': 'mean',
    'booking_count': 'mean',
    'room_id': 'count'
}).round(2).sort_values('total_revenue', ascending=False)

# Create a generic naming scheme based on ranking
cluster_names = [f"{i+1}st (Cluster {i+1})" for i in range(len(cluster_summary))]
cluster_summary['cluster_name'] = cluster_names[:len(cluster_summary)]

name_map = dict(zip(cluster_summary.index, cluster_summary['cluster_name']))
df_clean['cluster_name'] = df_clean['cluster'].map(name_map)

print("\nFinal Clusters (Ranked by Revenue):")
print(df_clean['cluster_name'].value_counts())

# 12. Principal Component Analysis (PCA) for Visualization
# IMPORTANT: Fit PCA on the scaled data (X_scaled)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# 13. Save Final CSV
output_cols = ['room_id','room_num','type','price','Branch_name','booking_count',
               'total_revenue','avg_stay','cluster','cluster_name']

df_clean[output_cols].to_csv("FINAL_ROOMS_CLUSTERS_HIGH_ACCURACY.csv", index=False)

print("\nCSV saved: FINAL_ROOMS_CLUSTERS_HIGH_ACCURACY.csv")

# 14. SAVE MODEL AS PKL (The FIX for the app problem!)
model_data = {
    # IsolaionForest object
    "iso": iso,
    # StandardScaler object (essential for scaling new data)
    "scaler": scaler,
    # KMeans object (essential for prediction)
    "kmeans": kmeans_final,
    # PCA object (essential for 2D visualization in Streamlit)
    "pca": pca,
    # List of features (essential for input column selection)
    "features": features,
    # Cluster name map (Added for app functionality)
    "name_map": name_map 
}

with open("hotel_clustering_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("\nModel saved as hotel_clustering_model.pkl")
print("Done: The trained model is ready to be used in app1.py")