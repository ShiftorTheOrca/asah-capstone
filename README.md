# Customer Segmentation using RFM Analysis and K-Means Clustering

> Capstone Project for ASAH led by Dicoding Program

## Table of Contents

- [Overview](#overview)
- [Business Understanding](#business-understanding)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Project Structure](#project-structure)
- [Installation & Requirements](#installation--requirements)
- [Usage](#usage)
- [Results](#results)
- [Key Findings](#key-findings)
- [Contributors](#contributors)
- [License](#license)

---

## Overview

This project implements **Customer Segmentation** using **RFM (Recency, Frequency, Monetary) Analysis** combined with **K-Means Clustering** algorithm. The goal is to segment customers based on their purchasing behavior to enable targeted marketing strategies and improve customer retention.

---

## Business Understanding

### Problem Statement

Businesses need to understand their customers better to:

- Identify high-value customers
- Detect customers at risk of churning
- Optimize marketing budget allocation
- Personalize customer experiences

### Objectives

1. Segment customers based on their transaction patterns
2. Identify characteristics of each customer segment
3. Provide actionable business recommendations for each segment

### Success Metrics

- **Silhouette Score**: Measure cluster quality (higher is better)
- **Business Interpretability**: Clusters should be meaningful and actionable

---

## Dataset

### Data Source

The dataset is sourced from **Kaggle**:

- [Online Retail UCI Dataset](https://www.kaggle.com/datasets/viridianachow/online-retail-uci-dataset)
- [Online Retail II UCI](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)

### Dataset Description

| Column      | Description                             |
| ----------- | --------------------------------------- |
| Invoice     | Invoice number (unique per transaction) |
| StockCode   | Product code                            |
| Description | Product description                     |
| Quantity    | Number of items purchased               |
| InvoiceDate | Date and time of transaction            |
| Price       | Unit price of product                   |
| CustomerID  | Unique customer identifier              |
| Country     | Customer's country                      |

### Dataset Statistics

- **Total Records**: ~1,000,000+ transactions
- **Time Period**: December 2009 - December 2011
- **Countries**: 40+ countries (primarily UK)

---

## Methodology

### 1. Data Loading

- Load datasets from Kaggle API
- Merge two datasets into a single DataFrame

### 2. Exploratory Data Analysis (EDA)

- Data type analysis
- Statistical summary
- Correlation matrix
- Distribution visualization

### 3. Pre-processing

#### Raw Data Cleaning

- Remove duplicate records
- Handle missing values
- Filter out cancelled transactions (Invoice starting with 'C')
- Remove negative Quantity and Price values
- Create TotalSales column (Quantity × Price)

#### RFM Feature Engineering

- **Recency**: Days since last purchase
- **Frequency**: Number of unique transactions
- **Monetary**: Total spending amount

#### RFM Pre-processing

1. **Outlier Removal**: IQR method (1.5 × IQR)
2. **Log Transformation**: `log1p()` to normalize skewed distributions
3. **Normalization**: MinMaxScaler (0-1 range)

### 4. Clustering

#### K-Means Clustering

- **Elbow Method**: Determine optimal number of clusters
- **Optimal K**: 4 clusters
- **Silhouette Score**: Evaluate cluster quality

#### PCA Visualization

- Reduce dimensions to 2D for visualization
- Display cluster centroids

### 5. Cluster Analysis & Labeling

Clusters are labeled based on RFM characteristics:

| Cluster | Recency | Frequency | Monetary | Label                            |
| ------- | ------- | --------- | -------- | -------------------------------- |
| 0       | Low     | High      | High     | **Loyal Customers**              |
| 1       | High    | Low       | Low      | **Churns / One Time Spenders**   |
| 2       | High    | High      | High     | **High Risk Churns**             |
| 3       | Low     | Low       | Low      | **New Customers / Low Spenders** |

---

## Project Structure

```
asah-capstone/
├── README.md                              # Project documentation
├── customer_segmentation_capstone.ipynb   # Main Jupyter notebook
├── kaggle.json                            # Kaggle API credentials (not included)
├── Online Retail.csv                      # Dataset 1 (downloaded)
├── online_retail_II.csv                   # Dataset 2 (downloaded)
└── online_retail_uci_clustering.csv       # Output: Segmented customer data
```

---

## Installation & Requirements

### Prerequisites

- Python 3.8+
- Kaggle account (for dataset download)

### Required Libraries

```python
numpy
pandas
matplotlib
seaborn
scikit-learn
yellowbrick
plotly
```

### Installation Steps

1. **Clone the repository**

```bash
git clone https://github.com/ShiftorTheOrca/asah-capstone.git
cd asah-capstone
```

2. **Install dependencies**

```bash
pip install numpy pandas matplotlib seaborn scikit-learn yellowbrick plotly
```

3. **Set up Kaggle API**

   - Go to [Kaggle Account Settings](https://www.kaggle.com/settings)
   - Create new API token (downloads `kaggle.json`)
   - Place `kaggle.json` in the project directory

4. **Run the notebook**
   - Open in Google Colab (recommended) or Jupyter Notebook
   - Execute cells sequentially

---

## Usage

### Running in Google Colab (Recommended)

1. Click the "Open in Colab" badge at the top
2. Upload your `kaggle.json` when prompted
3. Run all cells

### Running Locally

```bash
jupyter notebook customer_segmentation_capstone.ipynb
```

### Output

The notebook generates:

- `online_retail_uci_clustering.csv`: Customer data with Cluster and Label columns

---

## Results

### Clustering Performance

| Metric             | Value      |
| ------------------ | ---------- |
| Number of Clusters | 4          |
| Silhouette Score   | ~0.35-0.45 |

### Customer Distribution

| Cluster | Label                        | Count  | Percentage |
| ------- | ---------------------------- | ------ | ---------- |
| 0       | Loyal Customers              | ~1,040 | ~25%       |
| 1       | Churns / One Time Spenders   | ~1,850 | ~45%       |
| 2       | High Risk Churns             | ~605   | ~15%       |
| 3       | New Customers / Low Spenders | ~625   | ~15%       |

### Cluster Characteristics

| Cluster                      | Avg Recency (days) | Avg Frequency | Avg Monetary ($) |
| ---------------------------- | ------------------ | ------------- | ---------------- |
| Loyal Customers              | 32                 | 8.5           | 3,942            |
| Churns / One Time Spenders   | 424                | 1.3           | 473              |
| High Risk Churns             | 246                | 4.6           | 1,972            |
| New Customers / Low Spenders | 38                 | 2.1           | 1,212            |

---

## Key Findings

### 1. Loyal Customers (Cluster 0)

- Recently purchased (32 days avg)
- High purchase frequency (8.5 transactions)
- Highest monetary value ($3,942)
- **Priority**: Highest - maintain and reward

### 2. Churns / One Time Spenders (Cluster 1)

- Haven't purchased in a long time (424 days)
- Very low frequency (1.3 transactions)
- Low monetary value ($473)
- **Priority**: Low - win-back campaigns

### 3. High Risk Churns (Cluster 2)

- Moderately inactive (246 days)
- Good historical frequency (4.6 transactions)
- High monetary value ($1,972)
- **Priority**: High - immediate retention efforts

### 4. New Customers / Low Spenders (Cluster 3)

- Recent activity (38 days)
- Low frequency (2.1 transactions)
- Moderate spending ($1,212)
- **Priority**: Medium - nurture and develop

---

## Contributors

| Name             | Role           | GitHub                                               |
| ---------------- | -------------- | ---------------------------------------------------- |
| Leonardo Sanjaya | Data Scientist | [@ShiftorTheOrca](https://github.com/ShiftorTheOrca) |
| Nelson Ahli      | Data Scientist | [@nelsooooon](https://github.com/nelsooooon)         |
| Steven Gunawan   | Data Scientist | [@veeennn](https://github.com/veeennn)               |

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## References

- [RFM Analysis - Wikipedia](<https://en.wikipedia.org/wiki/RFM_(market_research)>)
- [K-Means Clustering - Scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail)
- [Customer Segmentation Best Practices](https://www.analyticsvidhya.com/blog/2020/10/customer-segmentation-using-rfm-analysis/)

---
