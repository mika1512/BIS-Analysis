# %%
# Import required libraries
import pyodbc
from datetime import datetime
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
#from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import seaborn as sns
from sqlalchemy import inspect
from sqlalchemy import create_engine
import numpy as np

# %%
# Database connection details
DRIVER_NAME = 'SQL Server'
SERVER_NAME = 'DESKTOP-QU8EF69'  # Replace with your server name
DATABASE_NAME = 'CompanyX'       # Replace with your database name

# Trusted connection (Windows Authentication)
conn_str = f"mssql+pyodbc://{SERVER_NAME}/{DATABASE_NAME}?driver={DRIVER_NAME}&trusted_connection=yes"

# If using SQL Server authentication (provide username and password)
# conn_str = f"mssql+pyodbc://username:password@{SERVER_NAME}/{DATABASE_NAME}?driver={DRIVER_NAME}"

# Create the engine
engine = create_engine(conn_str)

# Test connection
try:
    with engine.connect() as connection:
        print("Connection successful!")
except Exception as e:
    print("Connection failed:", e)


# %%
# Step 2: Query data from the warehouse
query = """
WITH ProductMetrics AS (
    SELECT 
        P.ProductKey,
        P.ProductName,
        P.SubcategoryID,
        P.SubcategoryName,
        SUM(F.TotalQtySold) AS TotalQtySold,
        SUM(F.SalesRevenue) AS SalesRevenue,
        SUM(F.GrossProfit) AS GrossProfit,
        AVG(F.SalesGrowthRate) AS SalesGrowthRate
    FROM 
        PRODUCT DIM P
    INNER JOIN 
        PRODUCT MIX FACT F
        ON P.ProductKey = F.ProductKey
    GROUP BY 
        P.ProductKey, P.ProductName, P.SubcategoryID
)
SELECT 
    PM1.ProductKey AS ProductKey,
    PM1.ProductName AS ProductName,
    PM1.SubcategoryName AS SubcategoryName,
    PM1.SubcategoryID AS SubcategoryID,
    PM1.TotalQtySold,
    PM1.SalesRevenue,
    PM1.GrossProfit,
    PM1.SalesGrowthRate,
    PM2.ProductKey AS SameSubcategoryProductKey,
    PM2.ProductName AS SameSubcategoryProductName
FROM 
    ProductMetrics PM1
INNER JOIN 
    ProductMetrics PM2
    ON PM1.SubcategoryID = PM2.SubcategoryID
    AND PM1.ProductKey != PM2.ProductKey
ORDER BY 
    PM1.SubcategoryID, PM1.ProductKey;
"""
df = pd.read_sql(query, conn)


# %%
# Step 3: Preprocessing and cleaning
# Check for missing values
print(df.isnull().sum())
#data[metrics] = data[metrics].fillna(0)

# Fill or drop missing values
#NOTE: Missing values are manually handled

# data_scaled = scaler.fit_transform(data[metrics])
# df.fillna(method='ffill', inplace=True)  # Example: forward fill

# Remove outliers (outside 1.5 * IQR)
Q1 = df[numeric_features].quantile(0.25)
Q3 = df[numeric_features].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[numeric_features] < (Q1 - 1.5 * IQR)) | (df[numeric_features] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Normalize/standardize numerical data
scaler = StandardScaler()
numeric_features = ['TotalQtySold', 'SalesRevenue', 'GrossProfit']  
df_scaled = scaler.fit_transform(df[numeric_features])





# %%
# Define the range of potential cluster numbers
k_values = range(1, 11)  # Try clusters from 1 to 10
inertia = []  # Store the inertia (sum of squared distances) for each value of k

# Loop over each value of k and calculate inertia
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)  # Initialize KMeans
    kmeans.fit(df_scaled)  # Fit to scaled data
    inertia.append(kmeans.inertia_)  # Append the inertia to the list

# Plot the elbow graph
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.xticks(np.arange(1, 11, 1))
plt.grid(True)
plt.show()

# %%
# Step 4: Apply clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Define number of clusters
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Step 5: Visualize clustering results
sns.pairplot(df, hue='Cluster', vars=numeric_features)
plt.title("Clustering Results")
plt.show()

# Optional: Save the results back to SQL
#df.to_sql('ClusteredTable', con=conn, if_exists='replace', index=False)

# Close the connection
#conn.close()


# %%
"""
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Create a linkage matrix
linkage_matrix = linkage(df_scaled, method='ward')  # Use Ward's method for variance minimization

# Visualize the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix, truncate_mode='lastp', p=10)  # Adjust p for fewer leaf nodes
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.show()

# Assign clusters based on a distance threshold
df['HierarchicalCluster'] = fcluster(linkage_matrix, t=3, criterion='maxclust')  # Adjust `t` for number of clusters
"""

# %%

# Check if the CLUSTER_DIM table exists
inspector = inspect(engine)
table_names = inspector.get_table_names()

if 'CLUSTER_DIM' in table_names:
    print("CLUSTER_DIM table exists. Replacing the data...")
    if_exists_option = 'replace'
else:
    print("CLUSTER_DIM table does not exist. Creating it...")
    if_exists_option = 'replace'  # The first write will create the table



# Define the cluster metadata based on metrics
cluster_metadata = []

# Iterate through each cluster and calculate basic statistics for naming and describing
for cluster_id in df['Cluster'].unique():
    cluster_data = df[df['Cluster'] == cluster_id]
    
    # Extract statistics for description (example: SalesRevenue and TotalQtySold)
    avg_revenue = cluster_data['SalesRevenue'].mean()
    avg_qty = cluster_data['TotalQtySold'].mean()
    avg_profit = cluster_data['GrossProfit'].mean()
    current_timestamp = datetime.now()
    
    # Create dynamic names and descriptions
    cluster_name = f"Cluster_{cluster_id + 1}"
    cluster_description = (
        f"Cluster {cluster_id + 1}: "
        f"Average Revenue: {avg_revenue:.2f}, "
        f"Average Quantity Sold: {avg_qty:.2f}"
        f"Average Gross Profit: {avg_profit:.2f}"
    )
    
    cluster_metadata.append((cluster_id + 1, cluster_name, cluster_description, current_timestamp))

# Convert to DataFrame
cluster_dim_df = pd.DataFrame(cluster_metadata, columns=['ClusterKey', 'ClusterName', 'ClusterDescription', 'ModifiedDate'])

# Save to CLUSTER_DIM table (replace or create)
cluster_dim_df.to_sql('CLUSTER_DIM', con=engine, if_exists=if_exists_option, index=False)
print("Cluster metadata saved to CLUSTER_DIM.")

# %%
# %% Update PRODUCT_MIX_FACT table with ClusterKey
product_cluster_mapping = df[['ProductKey', 'Cluster']].copy()
product_cluster_mapping['ClusterKey'] = product_cluster_mapping['Cluster'] + 1

# Merge or update instead of replacing the entire table
existing_fact = pd.read_sql('SELECT * FROM PRODUCT_MIX_FACT', engine)
updated_fact = pd.merge(existing_fact, product_cluster_mapping, on='ProductKey', how='left')
updated_fact.to_sql('PRODUCT_MIX_FACT', con=engine, if_exists='replace', index=False)

print("Updated PRODUCT_MIX_FACT with ClusterKey.")


