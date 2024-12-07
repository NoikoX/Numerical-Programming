import pandas as pd
import matplotlib.pyplot as plt
from pyclustering.cluster.kmedoids import kmedoids
import random

# Step 1: Read the CSV file
df = pd.read_csv('Datasets/avocado.csv')

# Step 2: Select the two columns for clustering
data = df[['Total Volume', 'Large Bags']].values

# Step 3: Create initial medoids (choose random points)
# Choose 3 random initial medoids (you can adjust the number of clusters here)
num_clusters = 4  # Set the number of clusters
initial_medoids = random.sample(range(len(data)), num_clusters)

# Step 4: Perform k-medoids clustering
kmedoids_instance = kmedoids(data, initial_medoids)

# Step 5: Run the algorithm
kmedoids_instance.process()
clusters = kmedoids_instance.get_clusters()  # Get the clusters
final_medoids = kmedoids_instance.get_medoids()  # Get the final medoids

# Step 6: Plot the results
# Convert clusters to a list of points for plotting
for cluster in clusters:
    cluster_data = data[cluster]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1])

# Plot the final medoids
for medoid_index in final_medoids:
    plt.scatter(data[medoid_index][0], data[medoid_index][1], color='red', marker='x', s=200, label='Medoid')

plt.xlabel('Total Volume')
plt.ylabel('Large Bags')
plt.title('K-Medoids Clustering')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# TO BE CONTINUED