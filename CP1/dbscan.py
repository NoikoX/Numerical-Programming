import numpy as np
from collections import deque

def dbscan(points, eps, min_samples):
    """Clusters points using DBSCAN."""
    clusters = []
    visited = set()
    noise = set()

    def region_query(point):
        """Find neighbors within epsilon radius."""
        distances = np.linalg.norm(points - point, axis=1)
        return np.argwhere(distances < eps).flatten()

    def expand_cluster(point_idx, neighbors):
        """Expands cluster from a core point."""
        cluster = [point_idx]
        queue = deque(neighbors)

        while queue:
            neighbor_idx = queue.popleft()
            if neighbor_idx not in visited:
                visited.add(neighbor_idx)
                new_neighbors = region_query(points[neighbor_idx])
                if len(new_neighbors) >= min_samples:
                    queue.extend(new_neighbors)
            if neighbor_idx not in cluster:
                cluster.append(neighbor_idx)

        return cluster

    for idx, point in enumerate(points):
        if idx in visited:
            continue
        visited.add(idx)
        neighbors = region_query(point)

        if len(neighbors) < min_samples:
            noise.add(idx)
        else:
            cluster = expand_cluster(idx, neighbors)
            clusters.append(cluster)

    return [points[cluster] for cluster in clusters], points[list(noise)]
