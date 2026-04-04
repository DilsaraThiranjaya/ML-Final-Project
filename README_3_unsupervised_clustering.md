# Phase 3: Unsupervised Behavioral Clustering - Technical Documentation

This document explains the procedural methodology engineered inside `3_unsupervised_clustering.ipynb`. It translates complex geometric concepts regarding k-Means clustering and centroid analysis into human-readable business logic for AuraCart's marketing and customer success teams.

---

### Cell 1: Environment Registration and Cluster Seed Initialization
**Code Focus**: Loading libraries and importing pandas/preprocessor artifacts.
**Reasoning**:
- The notebook ingests the sanitized `ecommerce_cleaned.csv` and frozen `base_preprocessor.joblib`. 
- `KMeans` from Scikit-Learn is the primary engine for grouping.
- `silhouette_score` is imported separately for quantitative validation of cluster cohesion and separation.

### Cell 2: Feature Transformation for Euclidean Space
**Code Focus**: `preprocessor.fit_transform()` on raw behavioral inputs.
**Reasoning**:
- k-Means uses Euclidean distance calculation to minimize variance. Features with large scales (like 'price' vs 'quantity') would dominate the model weights improperly. 
- We use the `fit_transform` method on our frozen pipeline to ensure all numerical inputs are standardized (z-score normalization) before the algorithm attempts discovery.

### Cell 3: Task 3.5.2: Finding the 'True' k via Statistical Metrics
**Code Focus**: For-loops iterating through ranges of Clusters with WCSS and Silhouette plots.
**Reasoning**:
- **Elbow Method (WCSS)**: We calculate 'Inertia' (Within-Cluster Sum of Squares) for different k-values. We plot this to find the "Elbow" point where adding more clusters no longer significantly reduces error.
- **Silhouette Score**: This identifies how "distinct" the clusters are. We look for local maxima in this score to find a k that minimizes overlap between customer groups.
- Visualizing both side-by-side ensures we aren't choosing a k that is mathematically sound but structurally over-segmented.

### Cell 4: Executing Final Geometric Partitioning
**Code Focus**: `KMeans(n_clusters=4, init='k-means++', n_init=10)`.
**Reasoning**:
- We select k=4 based on the peak performance seen in the elbow/silhouette plots.
- `k-means++` initialization is used to prevent the algorithm from getting stuck in local minima (poor starting points for centroids). 
- `fit_predict` generates the integer label (0, 1, 2, or 3) and attaches it directly to our dataframe for human-readable grouping.

### Cell 5: Task 3.5.3 & 3.5.4: Centroid Extraction & Business Mandate Translation
**Code Focus**: `df.groupby('cluster').mean()` and Seaborn Scatterplots.
**Reasoning**:
- **Centroid Analysis**: By calculating the average values per cluster for critical features (Price, Quantity, Transit Time), we define the "Average Persona" of each group.
- **Scatterplots**: We visualize the separation of clusters (Price vs shipping_duration) to confirm that the algorithm has found distinct structural divisions in the data.
- **Strategic Mapping**: 
  - **Cluster 0**: High Price/Low Volume represents "VIP/Luxury" targets.
  - **Cluster 1**: Low Price/High Volume represents "Wholesale/Discount" targets.
  - **Cluster 2**: High returns/Long transit indicates "Logistical Friction".
  - **Cluster 3**: New/Small orders indicates a "Nurturing Phase".
  - These behavioral discoveries allow AuraCart to execute hyper-personalized campaigns instead of uniform marketing.
