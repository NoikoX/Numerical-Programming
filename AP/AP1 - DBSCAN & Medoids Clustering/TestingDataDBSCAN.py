# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN
#
# # Step 1: Read the CSV file
# df = pd.read_csv('avocado.csv')  # Replace 'your_file.csv' with your actual file path
#
# # Step 2: Select the two columns (replace 'column1' and 'column2' with actual column names)
#
# x = df['Total Volume']  # First column (X-axis)
# y = df['Large Bags']  # Second column (Y-axis)
#
# # Step 3: Plot the data
# plt.figure(figsize=(8, 6))  # Optional: Adjust the figure size
# plt.scatter(x, y, color='blue', marker='o')  # Scatter plot
# plt.xlabel('Column 1')  # Label for the x-axis
# plt.ylabel('Column 2')  # Label for the y-axis
# plt.title('2D Scatter Plot')  # Title of the plot
# plt.grid(True)  # Show grid lines
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Step 1: Read the CSV file
df = pd.read_csv('Datasets/avocado.csv')  # Replace with your actual file path

# Step 2: Select the two columns
X = df[['Total Volume', 'Small Bags']]  # Selecting both columns for DBSCAN

# Step 3: Apply DBSCAN without scaling
dbscan = DBSCAN(eps=500_000, min_samples=50)  # Adjust 'eps' to match the scale of your data
labels = dbscan.fit_predict(X)  # Getting cluster labels

# Step 4: Plot the clustered data
plt.figure(figsize=(8, 6))
plt.scatter(X['Total Volume'], X['Small Bags'], c=labels, cmap='viridis', marker='o')
plt.xlabel('Total Volume')
plt.ylabel('Small Bags')
plt.title('DBSCAN Clustering of Avocado Data (Without Scaling)')
plt.grid(True)
plt.colorbar(label='Cluster Label')  # Optional: Colorbar to indicate cluster labels
plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN
# from sklearn.neighbors import NearestNeighbors
#
# # Step 1: Read the CSV file
# df = pd.read_csv('avocado.csv')  # Load your data
# x = df['Total Volume']
# y = df['Small Bags']
#
# # Step 2: Prepare data for DBSCAN
# data = df[['Total Volume', 'Small Bags']].values
#
# # Step 3: Plot k-distance graph to find optimal epsilon
# # Using k=4 (as min_samples is often 4)
# neighbors = NearestNeighbors(n_neighbors=4)
# neighbors_fit = neighbors.fit(data)
# distances, indices = neighbors_fit.kneighbors(data)
#
# # Sort distances to find the elbow
# distances = sorted(distances[:, 3])
# plt.plot(distances)
# plt.title('k-distance Graph')
# plt.xlabel('Points')
# plt.ylabel('Distance')
# plt.show()
#
# # Step 4: Apply DBSCAN (choose optimal values after looking at the graph)
# dbscan = DBSCAN(eps=4000, min_samples=4)
# clusters = dbscan.fit_predict(data)
#
# # Step 5: Plot the resulting clusters
# plt.figure(figsize=(8, 6))
# plt.scatter(x, y, c=clusters, cmap='rainbow', marker='o')
# plt.xlabel('Total Volume')
# plt.ylabel('Small Bags')
# plt.title('DBSCAN Clustering')
# plt.grid(True)
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN
#
# # Load the dataset
# df = pd.read_csv('avocado.csv')
#
# # Select columns for clustering
# X = df[['Total Volume', 'Small Bags']]
#
# # Apply DBSCAN
# dbscan = DBSCAN(eps=1_000_000, min_samples=4)  # Adjust eps based on observation
# clusters = dbscan.fit_predict(X)
#
# # Plot the clusters
# plt.figure(figsize=(8, 6))
# plt.scatter(X['Total Volume'], X['Small Bags'], c=clusters, cmap='viridis')
# plt.xlabel('Total Volume')
# plt.ylabel('Small Bags')
# plt.title('DBSCAN Clustering')
# plt.grid(True)
# plt.show()

