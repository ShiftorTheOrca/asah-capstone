"""# **Import Library**
"""

import numpy as np
import pandas as pd
import os
import shutil
import zipfile

from sklearn.preprocessing import LabelEncoder, RobustScaler

from sklearn.cluster import HDBSCAN

"""# **Data Loading**
"""

# Mengambil dataset dari Kaggle
if not os.path.exists('assets/Online Retail.csv'):
    os.system('kaggle datasets download -d viridianachow/online-retail-uci-dataset')

    with zipfile.ZipFile('online-retail-uci-dataset.zip', 'r') as zip_ref:
        zip_ref.extractall('assets')

df = pd.read_csv('assets/Online Retail.csv')


"""# **Pre-processing**
## Raw Data
"""

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Memperbaiki format data masing-masing kolom
categorical_cols = ['InvoiceNo', 'StockCode', 'Description', 'CustomerID', 'Country']
numerical_cols = ['Quantity', 'UnitPrice']

for column in categorical_cols:
    df[column] = df[column].astype(str)

for column in numerical_cols:
    df[column] = df[column].astype(float)

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='mixed')

# Menghapus transaksi sebelum tahun 2010 bulan 12 tanggal 9 jam 13:00 agar total Bulan seimbang
start_date = "2010-12-09 13:00:00"

df.drop(df[df['InvoiceDate'] < start_date].index, inplace=True)

# Menghapus transaksi di luar United Kingdom
df.drop(df[df['Country'] != 'United Kingdom'].index, inplace=True)

"""### Feature Engineering
"""

# Mengelompokkan kata-kata dalam deskripsi ke dalam dictionary
dict_category = {
    'bags': [
        'bag', 'shopper', 'tote', 'lunch', 'jumbo bag',
        'satchel', 'handbag', 'travel bag'
    ],

    'kitchen': [
        'mug', 'cup', 'teacup', 'plate', 'bowl', 'dish',
        'bottle', 'water bottle', 'pantry', 'jug', 'milk jug',
        'teapot', 'napkin', 'coaster', 'tray', 'cutlery',
        'bread bin', 'egg cup', 'salt', 'pepper', 'kitchen'
    ],

    'decoration': [
        'heart', 'hanging', 'ornament', 'frame', 'photo frame',
        'sign', 'door sign', 'wall art', 'decoration',
        'doorstop', 'cushion', 'mirror', 'vase', 'bunting',
        'union jack', 'sweet home', 'love'
    ],

    'lighting': [
        'light', 't light', 'tlight', 'lantern', 'candle',
        'lamp', 'led', 'night light', 'fairy light', 'tea light'
    ],

    'christmas': [
        'christmas', 'xmas', 'santa', 'snowman', 'reindeer',
        'advent', 'stocking', 'tree decoration', 'bauble'
    ],

    'storage': [
        'box', 'tin', 'jar', 'basket', 'case', 'cases',
        'container', 'storage', 'trinket', 'recipe box',
        'tissue box', 'book box', 'chest'
    ],

    'baking': [
        'cake', 'cakestand', 'cake stand', 'baking', 'cookie',
        'mould', 'fairy cake', 'cupcake', 'muffin', 'recipe'
    ],

    'stationery': [
        'card', 'birthday', 'notebook', 'paper', 'wrap',
        'ribbon', 'tag', 'pen', 'pencil', 'gift'
    ],

    'clocks': [
        'clock', 'alarm', 'wall clock', 'alarm clock', 'watch'
    ],

    'garden': [
        'garden', 'plant', 'pot', 'flower', 'seed',
        'herb', 'watering', 'bird'
    ],

    'kids': [
        'spaceboy', 'dolly', 'fairy', 'childrens', 'children',
        'kids', 'girl', 'boy', 'toy', 'doll', 'teddy', 'baby'
    ],

    'furniture': [
        'drawer', 'knob', 'handle', 'hook', 'hanger',
        'coat', 'door', 'cabinet'
    ]
}

def category_map(desc):
    desc = str(desc).lower()

    for category, keywords in dict_category.items():
        for item in keywords:
            if item in desc:
                return category
    return 'others'

# Membuat kolom Category
df['Category'] = df['Description'].apply(category_map)

# Mengelompokkan Bulan transaksi ke dalam kolom Seasonality
seasons = {
    'Spring': [3,4,5],
    'Summer': [6,7,8],
    'Fall': [9,10,11],
    'Winter': [12,1,2]
}

for season, months in seasons.items():
    df.loc[df['InvoiceDate'].dt.month.isin(months), 'Seasonality'] = season

# Menghitung Monetary dengan kolom Sales
df['Sales'] = df['Quantity'] * df['UnitPrice']

# Menghapus transaksi custonmer yang diretur
retur = (df.loc[df['Sales'] <= 0, 'CustomerID'] + '_' + df.loc[df['Sales'] <= 0, 'StockCode'])

df = df[~(df['CustomerID'] + '_' + df['StockCode']).isin(retur)]

"""## RFM"""
# Membuat kolom RFM
analysis_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

Recency = ('InvoiceDate', lambda x: (analysis_date - x.max()).days)
Frequency = ('InvoiceNo', 'nunique')
Monetary = ('Sales', 'sum')

# Membuat df RFM
df_rfm = df.groupby('CustomerID').agg(
    Recency=Recency,
    Frequency=Frequency,
    Monetary=Monetary
).reset_index()

columns = ['Recency', 'Frequency', 'Monetary']

# Menerapkan Log Transformation
df_rfm_log = df_rfm.copy()
df_rfm_log[columns] = np.log1p(df_rfm_log[columns])

# Menerapkan Standarization
df_rfm_scaled = df_rfm_log.copy()

scaler = RobustScaler()
df_rfm_scaled[columns] = scaler.fit_transform(df_rfm_scaled[columns])

# Melakukan Binning pada kolom Frequency
bin_columns = ['Recency', 'FrequencyBin', 'Monetary']
bins = [-np.inf, 0, 1, 2, np.inf]
bin_labels = ['Low', 'Medium', 'High', 'Very High']

df_rfm_binned = df_rfm_scaled.copy()
df_rfm_binned['FrequencyBin'] = pd.cut(df_rfm_binned['Frequency'], bins=bins, labels=bin_labels)

le = LabelEncoder()
df_rfm_binned['FrequencyBin'] = le.fit_transform(df_rfm_binned['FrequencyBin'])

"""# **Clustering**
"""## HDBSCAN"""

# Mencari best param model HDBSCAN
X_model = df_rfm_binned[bin_columns].values

# Menganalisa hasil clustering HDBSCAN
h = 5
hdb = HDBSCAN(min_cluster_size=h, min_samples=h)

hdb.fit(X_model)
labels = hdb.labels_

df_hdb = df_rfm.copy()
df_hdb['Cluster'] = labels

"""### Cluster Labeling"""

# Menghapus Cluster Outlier (-1)
df_hdb = df_hdb[df_hdb['Cluster'] != -1]

# Melakukan analisa cluster
cluster_analysis = df_hdb.groupby('Cluster').agg(
    Avg_Recency=('Recency', 'mean'),
    Avg_Frequency=('Frequency', 'mean'),
    Avg_Monetary=('Monetary', 'mean'),
    Count=('CustomerID', 'count'),
    Percentage=('CustomerID', lambda x: (x.count() / len(df_rfm_scaled)) * 100)
).reset_index().sort_values(by='Cluster')

# Memberikan label pada cluster
cluster_label = {
    0: 'Loyal Customer',
    1: 'Regular Customer',
    2: 'Churn/One Time Spender',
    3: 'High Risk Churn'
}

cluster_analysis['Label'] = cluster_analysis['Cluster'].map(cluster_label)

"""# **Post-processing**"""

# Menyatukan label ke dalam df RFM
cluster_to_label = cluster_analysis['Label'].to_dict()

df_hdb['Label'] = df_hdb['Cluster'].map(cluster_to_label)

# Menyatukan df RFM kembali ke df Raw Data
df_final = df.copy()
df_final = df_final.merge(
    df_hdb[['CustomerID', 'Cluster', 'Label']],
    on='CustomerID',
    how='inner'
)

# Menyimpan hasil Clustering
df_final.to_csv('assets/online_retail_uci_clustering.csv', index=False)