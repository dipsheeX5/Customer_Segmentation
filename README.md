# Customer_Segmentation
Customer segmentation using K-Means clustering on Mall Customers dataset. Includes EDA, feature scaling, Elbow + Silhouette methods to find optimal clusters, clear visualizations, and business insights for targeted marketing strategies.

##  Project Overview

This project performs **unsupervised customer segmentation** on mall customer data using the **K-Means clustering algorithm**. The goal is to group customers based on their **annual income** and **spending score** to help businesses create targeted marketing strategies (e.g., premium offers for high-spenders, discounts for budget-conscious customers).

It is a beginner-to-intermediate level machine learning project, ideal for demonstrating understanding of:
- Unsupervised learning
- Clustering evaluation techniques
- Feature scaling importance
- Data visualization & business interpretation


**Dataset**: Mall Customers (200 records) from Kaggle  
Link: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python


## Key Features

- Exploratory Data Analysis (EDA) with pairplots and boxplots
- Feature scaling using StandardScaler (critical for distance-based algorithms)
- Optimal number of clusters determined using **Elbow Method** 


**Silhouette Score**
- Clear visualization of clusters with centroids
- Business-oriented interpretation of each customer segment
- Results saved for further use


## Tech Stack

- Python 3.12
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Project Structure
CustomerSegmentation/
│
├── Mall_Customers.csv                  # Original dataset
├── Mall_Customers_with_Clusters.csv    # Data with cluster labels
├── Customer.ipynb                      # Complete Jupyter Notebook
└── README.md                           # This file

## How to Run

1. Clone the repository
   git clone https://github.com/your-username/customer-segmentation-kmeans.git

2. Install required packages (recommended: use virtual environment)Bash
  pip install pandas numpy matplotlib seaborn scikit-learn

3. Open and run the notebookBash
   jupyter notebook Customer.ipynb

4. Make sure the dataset file is in the same folder (or update the path in the notebook).


### Main Results & Insights

After running K-Means with **k=5** clusters (chosen via Elbow method & highest silhouette score):

| Cluster | Avg Age | Avg Income (k$) | Avg Spending Score | Count | Interpretation                                      |
|---------|---------|------------------|---------------------|-------|-----------------------------------------------------|
| 0       | 42.7    | 55.3             | 49.5                | 81    | Medium income & spending – general customers        |
| 1       | 32.7    | 86.5             | 82.1                | 39    | **High income + high spending** → Premium/Loyal     |
| 2       | 25.3    | 25.7             | 79.4                | 22    | Low income + high spending → Young impulse buyers   |
| 3       | 41.1    | 88.2             | 17.1                | 35    | **High income + low spending** → Target with promotions |
| 4       | 45.2    | 26.3             | 20.9                | 23    | Low income + low spending → Budget segment          |


# Business Recommendations:

Cluster 1 → Luxury products, loyalty programs
Cluster 3 → Discounts, personalized offers to increase spending
Cluster 2 → Trendy, affordable items for young customers

# What I Learned / Key Takeaways
Why feature scaling is mandatory for K-Means
How to choose the right number of clusters (Elbow + Silhouette)
Importance of visualizing clusters and interpreting them from a business perspective
How simple 2-feature clustering can reveal meaningful customer groups


# Possible Improvements / Future Work

Include 'Age' and encode 'Gender' for 3D clustering
Compare K-Means with Hierarchical Clustering or DBSCAN
Add PCA if more features are included
Build a simple Streamlit dashboard for interactive segmentation