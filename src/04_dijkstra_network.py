#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_dijkstra_network.py
Least-cost path modeling, cost matrix, IIC sensitivity, graph metrics, and Conefor export
Compatible with full city, north, south, and south_30 subnetworks
"""

import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd
import networkx as nx
import heapq
from tqdm import tqdm
from scipy.spatial.distance import cdist
from rasterio.transform import rowcol, xy
from shapely.geometry import LineString
import matplotlib.pyplot as plt
import os
import pickle

# =====================================================
# 1️⃣ CONFIGURATION - CHANGE THESE FOR EACH SCENARIO
# =====================================================
# For Full City:   REGION_NAME = "City",   POINTS_PATH = "data/start_point.shp"
# For North:       REGION_NAME = "North",  POINTS_PATH = "data/core_N.shp"
# For South:       REGION_NAME = "South",  POINTS_PATH = "data/core_S.shp"
# For South_30:    REGION_NAME = "South_30", POINTS_PATH = "data/core_S_30.shp"

RES_PATH = "data/Resistance.tif"
POINTS_PATH = "data/start_point.shp"
REGION_NAME = "City"

# =====================================================
# 2️⃣ LOAD DATA
# =====================================================

print("=" * 60)
print(f"Processing region: {REGION_NAME}")
print("=" * 60)

with rasterio.open(RES_PATH) as src:
    resistance = src.read(1).astype(np.float32)
    transform = src.transform
    crs = src.crs

print(f"Resistance shape: {resistance.shape}")

points = gpd.read_file(POINTS_PATH)

if points.crs != crs:
    points = points.to_crs(crs)

# Ensure Area_ha column exists
if "Area_ha" not in points.columns:
    print("ERROR: Area_ha column not found in points file.")
    print("Please ensure the points shapefile has an 'Area_ha' field.")
    exit(1)

n = len(points)
rows, cols = resistance.shape

print(f"Number of cores (nodes): {n}")

# =====================================================
# 3️⃣ CONVERT POINTS TO PIXEL INDICES
# =====================================================

def get_pixel(point):
    row, col = rowcol(transform, point.x, point.y)
    return row, col

points["pixel"] = points.geometry.apply(get_pixel)
pixels = np.array(points["pixel"].tolist())

# =====================================================
# 4️⃣ DIJKSTRA WITH PATH TRACKING
# =====================================================

def neighbors(r, c):
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r+dr, c+dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc

def cost_distance_with_path(start):
    dist = np.full((rows, cols), np.inf, dtype=np.float32)
    prev = {}
    sr, sc = start
    dist[sr, sc] = 0
    heap = [(0, sr, sc)]

    while heap:
        current_cost, r, c = heapq.heappop(heap)
        if current_cost > dist[r, c]:
            continue

        for nr, nc in neighbors(r, c):
            step_cost = (resistance[r, c] + resistance[nr, nc]) / 2
            new_cost = current_cost + step_cost

            if new_cost < dist[nr, nc]:
                dist[nr, nc] = new_cost
                prev[(nr, nc)] = (r, c)
                heapq.heappush(heap, (new_cost, nr, nc))

    return dist, prev

# =====================================================
# 5️⃣ BUILD COST MATRIX + LCP SHAPEFILES
# =====================================================

print("\nComputing cost matrix and LCPs...")
cost_matrix = np.zeros((n, n), dtype=np.float32)
lines = []

for i in tqdm(range(n), desc="Dijkstra iterations"):
    dist_surface, prev = cost_distance_with_path(pixels[i])

    for j in range(i + 1, n):
        r, c = pixels[j]
        cost = dist_surface[r, c]
        cost_matrix[i, j] = cost
        cost_matrix[j, i] = cost

        # reconstruct path
        path = []
        current = (r, c)
        while current in prev:
            path.append(current)
            current = prev[current]
        path.append(pixels[i])
        path.reverse()

        # convert to geographic coordinates
        coords = [xy(transform, rr, cc) for rr, cc in path]
        line = LineString(coords)

        lines.append({
            "from_id": i + 1,
            "to_id": j + 1,
            "cost": float(cost),
            "geometry": line
        })

# =====================================================
# 6️⃣ SAVE ALL LEAST COST PATHS
# =====================================================

os.makedirs("results", exist_ok=True)

lcp_gdf = gpd.GeoDataFrame(lines, crs=crs)
lcp_path = f"results/LCP_all_pairs_{REGION_NAME}.shp"
lcp_gdf.to_file(lcp_path)
print(f"LCP shapefile saved: {lcp_path}")

# =====================================================
# 7️⃣ EUCLIDEAN DISTANCE AND COST RATIO
# =====================================================

coords = np.array([(geom.x, geom.y) for geom in points.geometry])
euclidean_matrix = cdist(coords, coords)

cost_ratio = np.divide(cost_matrix,
                       euclidean_matrix,
                       out=np.zeros_like(cost_matrix),
                       where=euclidean_matrix != 0)

# =====================================================
# 8️⃣ DEFINE THRESHOLDS (P50, P60, P75, P90)
# =====================================================

CR_values = cost_ratio[cost_ratio > 0]

thresholds = {
    "P50": np.percentile(CR_values, 50),
    "P60": np.percentile(CR_values, 60),
    "P75": np.percentile(CR_values, 75),
    "P90": np.percentile(CR_values, 90)
}

print("\nThresholds:")
for label, thr in thresholds.items():
    print(f"  {label}: {thr:.6f}")

CR_threshold = thresholds["P75"]

# =====================================================
# 9️⃣ FILTER NETWORK FOR P75
# =====================================================

network_edges = []
for idx, row in lcp_gdf.iterrows():
    i = row["from_id"] - 1
    j = row["to_id"] - 1
    if cost_ratio[i, j] <= CR_threshold:
        network_edges.append(row)

network_gdf = gpd.GeoDataFrame(network_edges, crs=crs)
network_path = f"results/Network_P75_{REGION_NAME}.shp"
network_gdf.to_file(network_path)
print(f"P75 network shapefile saved: {network_path}")

# =====================================================
# 🔟 SAVE NODES SHAPEFILE
# =====================================================

points["ID"] = np.arange(1, n + 1)
nodes_path = f"results/Nodes_{REGION_NAME}.shp"
points.to_file(nodes_path)
print(f"Nodes shapefile saved: {nodes_path}")

# =====================================================
# 1️⃣1️⃣ IIC CALCULATION (BINARY FORMULATION)
# =====================================================

areas = points["Area_ha"].values
AL = np.sum(areas)

def calculate_iic(G, areas):
    iic_sum = 0.0
    nodes_list = list(G.nodes())
    for i in nodes_list:
        for j in nodes_list:
            if i != j:
                try:
                    nl = nx.shortest_path_length(G, i, j)
                    iic_sum += (areas[i] * areas[j]) / (1 + nl)
                except nx.NetworkXNoPath:
                    pass
    return iic_sum / (AL ** 2)

# =====================================================
# 1️⃣2️⃣ SENSITIVITY ANALYSIS LOOP
# =====================================================

print("\nSensitivity analysis...")
sensitivity_results = []

for label, thr in tqdm(thresholds.items(), desc="Thresholds"):
    G = nx.Graph()
    for i in range(n):
        G.add_node(i)

    for i in range(n):
        for j in range(i + 1, n):
            if cost_ratio[i, j] <= thr:
                G.add_edge(i, j, weight=cost_matrix[i, j])

    iic_val = calculate_iic(G, areas)
    components = nx.number_connected_components(G)
    density = nx.density(G)

    sensitivity_results.append({
        "Threshold": label,
        "Value": thr,
        "IIC": iic_val,
        "Components": components,
        "Density": density,
        "Edges": G.number_of_edges()
    })

results_df = pd.DataFrame(sensitivity_results)
results_csv = f"results/sensitivity_{REGION_NAME}.csv"
results_df.to_csv(results_csv, index=False)
print(f"Sensitivity results saved: {results_csv}")

# =====================================================
# 1️⃣3️⃣ PLOT IIC SENSITIVITY
# =====================================================

plt.figure(figsize=(8, 5))
plt.plot(results_df["Threshold"], results_df["IIC"], marker='o', linewidth=2, markersize=8)
plt.xlabel("Threshold", fontsize=12)
plt.ylabel("IIC", fontsize=12)
plt.title(f"IIC Sensitivity Analysis - {REGION_NAME}", fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plot_path = f"results/sensitivity_plot_{REGION_NAME}.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"Sensitivity plot saved: {plot_path}")

# =====================================================
# 1️⃣4️⃣ BUILD FINAL GRAPH FOR P75
# =====================================================

G_final = nx.Graph()
for i in range(n):
    G_final.add_node(i)

for i in range(n):
    for j in range(i + 1, n):
        if cost_ratio[i, j] <= CR_threshold:
            G_final.add_edge(i, j, weight=cost_matrix[i, j])

print(f"P75 network edges: {G_final.number_of_edges()}")

# =====================================================
# 1️⃣5️⃣ GRAPH THEORY METRICS
# =====================================================

print("\nComputing graph theory metrics...")
degree = nx.degree_centrality(G_final)
betweenness = nx.betweenness_centrality(G_final, weight='weight')
closeness = nx.closeness_centrality(G_final)
eigen = nx.eigenvector_centrality(G_final, max_iter=1000)
density = nx.density(G_final)
components = nx.number_connected_components(G_final)
efficiency = nx.global_efficiency(G_final)
mst = nx.minimum_spanning_tree(G_final, weight='weight')

metrics_df = pd.DataFrame({
    "Node": list(range(n)),
    "Degree": [degree[i] for i in range(n)],
    "Betweenness": [betweenness[i] for i in range(n)],
    "Closeness": [closeness[i] for i in range(n)],
    "Eigenvector": [eigen[i] for i in range(n)]
})
metrics_csv = f"results/graph_metrics_{REGION_NAME}.csv"
metrics_df.to_csv(metrics_csv, index=False)
print(f"Graph metrics saved: {metrics_csv}")

# =====================================================
# 1️⃣6️⃣ SAVE GRAPH AND COST MATRIX
# =====================================================

graph_path = f"results/graph_{REGION_NAME}.pkl"
with open(graph_path, "wb") as f:
    pickle.dump(G_final, f)
print(f"Graph saved: {graph_path}")

matrix_path = f"results/cost_matrix_{REGION_NAME}.npy"
np.save(matrix_path, cost_matrix)
print(f"Cost matrix saved: {matrix_path}")

areas_path = f"results/areas_{REGION_NAME}.npy"
np.save(areas_path, areas)
print(f"Areas saved: {areas_path}")

# =====================================================
# 1️⃣7️⃣ SAVE CONEFOR FILES (Scenario 2 - With Corridors)
# =====================================================

os.makedirs("conefor", exist_ok=True)

# nodes file
nodes_output = points[["ID", "Area_ha"]]
nodes_output.to_csv(f"conefor/nodes_{REGION_NAME}.txt",
                    sep=" ",
                    index=False,
                    header=False)
print(f"Conefor nodes file (with corridors): conefor/nodes_{REGION_NAME}.txt")

# connections file (with actual weights for Scenario 2)
connections = []
for i, j, data in G_final.edges(data=True):
    connections.append([i + 1, j + 1, data["weight"]])

if connections:
    connections_df = np.array(connections)
    np.savetxt(f"conefor/connections_{REGION_NAME}.txt",
               connections_df,
               fmt=["%d", "%d", "%.6f"])
    print(f"Conefor connections file (with corridors): conefor/connections_{REGION_NAME}.txt")
else:
    print("Warning: No edges found for Conefor connections file.")

# =====================================================
# 1️⃣8️⃣ CREATE CONEFOR FILES FOR SCENARIO 1 (Baseline - No Corridors)
# =====================================================

# For baseline (no corridors), we need a connections file with very large weights
# so that no connections are considered by Conefor

baseline_nodes_file = f"conefor/nodes_baseline_{REGION_NAME}.txt"
baseline_connections_file = f"conefor/connections_baseline_{REGION_NAME}.txt"

# Copy nodes file (same as above)
nodes_output.to_csv(baseline_nodes_file, sep=" ", index=False, header=False)
print(f"Conefor nodes file (baseline): {baseline_nodes_file}")

# Create baseline connections file with large dummy weights
baseline_connections = []
for i in range(n):
    for j in range(i + 1, n):
        baseline_connections.append([i + 1, j + 1, 999999.0])

if baseline_connections:
    baseline_connections_df = np.array(baseline_connections)
    np.savetxt(baseline_connections_file,
               baseline_connections_df,
               fmt=["%d", "%d", "%.6f"])
    print(f"Conefor connections file (baseline): {baseline_connections_file}")

# =====================================================
# 1️⃣9️⃣ CALCULATE dIIC (node removal analysis)
# =====================================================

print("\nCalculating dIIC (node removal analysis)...")
dIIC = {}

for node in tqdm(G_final.nodes(), desc="dIIC calculation"):
    G_temp = G_final.copy()
    G_temp.remove_node(node)
    temp_iic = calculate_iic(G_temp, areas)
    dIIC[node] = calculate_iic(G_final, areas) - temp_iic

dIIC_df = pd.DataFrame(list(dIIC.items()), columns=["Node", "dIIC"])
dIIC_csv = f"results/dIIC_{REGION_NAME}.csv"
dIIC_df.to_csv(dIIC_csv, index=False)
print(f"dIIC results saved: {dIIC_csv}")

# =====================================================
# 2️⃣0️⃣ PRINT SUMMARY
# =====================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
iic_value = calculate_iic(G_final, areas)
print(f"IICnum: {iic_value:.2f}")
print(f"EC(IIC): {iic_value * (AL**2):.2f}")
print(f"Network density: {nx.density(G_final):.4f}")
print(f"Connected components: {nx.number_connected_components(G_final)}")
print("=" * 60)
print("\nAll outputs successfully created.")
print("\n" + "=" * 60)
print("CONEFOR FILES SUMMARY")
print("=" * 60)
print(f"Scenario 2 (With Corridors):")
print(f"  - Nodes: conefor/nodes_{REGION_NAME}.txt")
print(f"  - Connections: conefor/connections_{REGION_NAME}.txt")
print(f"Scenario 1 (Baseline - No Corridors):")
print(f"  - Nodes: conefor/nodes_baseline_{REGION_NAME}.txt")
print(f"  - Connections: conefor/connections_baseline_{REGION_NAME}.txt")
print("=" * 60)