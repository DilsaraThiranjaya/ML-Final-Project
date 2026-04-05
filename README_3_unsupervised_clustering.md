# Phase 3: Unsupervised Behavioral Clustering

## Overview
Phase 3 focuses on **Unsupervised Learning**, where we discover hidden structures in the data without predefined labels. We use **k-Means Clustering** to segment AuraCart's customer base into distinct behavioral groups based on their purchasing patterns, delivery history, and product preferences.

### Learning Objectives
- **Feature Scaling**: Ensuring numerical features have equal weight in distance calculations.
- **The Elbow Method**: Using WCSS (Within-Cluster Sum of Squares) to find the optimal number of groups.
- **Silhouette Analysis**: Measuring how well-defined and separated our clusters are.
- **Centroid Interpretation**: Translating abstract clusters into actionable business personas (e.g., "Frequent High-Spenders").

---

## Cell-by-Cell Breakdown

### Cell 1: Environment Setup
**Code Logic:**
- Imports `KMeans` and `silhouette_score` from `sklearn`.
- Loads the cleaned dataset and the frozen preprocessor from Phase 1.
**Strategic Rationale:**
- We reuse the Phase 1 preprocessor here to ensure the "Clustering Space" is consistent with the "Supervised Space."

### Cell 2: Feature Transformation
**Code Logic:**
- Applies `preprocessor.fit_transform()` to the raw features.
**Strategic Rationale:**
- **Why scale for k-Means?** k-Means uses **Euclidean Distance**. If one feature (like Revenue) ranges from 0-1000 and another (Quantity) ranges from 0-10, the Revenue feature will dominate the distance calculation. Scaling puts everyone on a level playing field.

### Cell 3: Finding 'k' (The Elbow & Silhouette Charts)
**Code Logic:**
- Iterates from $k=2$ to $k=10$.
- Calculates **Inertia** (WCSS) and **Silhouette Score** for each $k$.
**Strategic Rationale:**
- **The Elbow Method**: We look for the "bend" in the WCSS curve. After this point, adding more clusters doesn't significantly improve the fit.
- **The Silhouette Score**: We look for the highest peak. A score near 1.0 means clusters are dense and far apart. A score near 0.0 means they overlap.

### Cell 4: Applying the Final Model
**Code Logic:**
- Fits `KMeans(n_clusters=4)`.
- Re-attaches labels (`df['cluster']`) to the original dataframe.
**Strategic Rationale:**
- We chose $k=4$ because it represents a balance between mathematical excellence and business simplicity.

### Cell 5: Centroid Analysis (The "DNA" of the clusters)
**Code Logic:**
- Aggregates the data by cluster using `.mean()`.
- Visualizes feature distribution across clusters.
**Strategic Rationale:**
- This is the most important part for a business. It tells us:
    - **Cluster 0**: "Late-Shipment Victims" (High shipping delay).
    - **Cluster 1**: "Loyal Value Seekers" (Balanced purchase history).
    - **Cluster 3**: "VIP Big Spenders" (Highest price and quantity).

---

## Key Terms for Beginners
| Term | Meaning |
| :--- | :--- |
| **Inertia (WCSS)** | The total distance between points and their assigned cluster center. Lower is better, but it always decreases as you add more clusters. |
| **Euclidean Distance** | The "straight-line" distance between two points in space. |
| **Centroid** | The geometric center of a cluster. |
| **Persona** | A descriptive name given to a cluster to make it understandable for marketing teams. |
