import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from threading import RLock

_lock = RLock()

@st.cache_data
def load_data():
    path = "./online_retail_uci_clustering.csv"
    data = pd.read_csv(path)
    return data

@st.cache_data
def plot_numerical_columns(df):
    with _lock:
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        num_vars = df[numerical_cols].shape[1]

        n_cols = 3
        n_rows = -(-num_vars // n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 4))
        axes = axes.flatten()

        for i, column in enumerate(df[numerical_cols].columns):
            sns.boxplot(y=df[column], ax=axes[i], color='blue')
            axes[i].set_title(column, fontsize=13, weight='bold')
            axes[i].set_xlabel('Value', fontsize=13)
            axes[i].set_ylabel('Frequency', fontsize=13)
            
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout()
        
    return fig

@st.cache_data
def plot_categorical_columns(df):
    with _lock:
        categorical_cols = df.select_dtypes(include=['object']).columns
        num_vars = df[categorical_cols].shape[1]

        n_cols = 1
        n_rows = -(-num_vars // n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 4))
        axes = axes.flatten()

        for i, column in enumerate(df[categorical_cols].columns):
            top_values = df[column].value_counts().head(12)
            
            sns.barplot(x=top_values.values, y=top_values.index, ax=axes[i], palette='plasma', legend=False)
            axes[i].set_title(column, fontsize=13, weight='bold')
            axes[i].set_xlabel('Count', fontsize=13)
            
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout()
    
    return fig

@st.cache_data
def plot_correlation_heatmap(df):
    with _lock:
        df_lencoder = df.copy()
        categorical_cols = df_lencoder.select_dtypes(include=['object']).columns

        for column in df_lencoder[categorical_cols].columns:
            label_encoder = LabelEncoder()
            df_lencoder[column] = label_encoder.fit_transform(df_lencoder[column].astype(str))

        correlation_matrix = df_lencoder.corr()
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))

        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
        ax.set_title('Correlation Matrix', fontsize=13, weight='bold')

        fig.tight_layout()
    
    return fig

@st.cache_data
def plot_most_spend_category_per_cluster(df):
    with _lock:
        sales_by_category = df.groupby(['CustomerID', 'Cluster', 'Category'])['Sales'].sum().unstack()
        most_spend_category = sales_by_category.idxmax(axis=1).reset_index()
        most_spend_category.columns = ['CustomerID', 'Cluster', 'Category']
        
        clusters = 4
        cluster_category = []
        cluster_category_analysis = []
        
        for i in range(clusters):
            cluster_category.append(most_spend_category[most_spend_category['Cluster'] == i])
            cluster_category_analysis.append(cluster_category[i].groupby('Category').agg(
                Count = ('CustomerID', 'count'),
                Percentage = ('CustomerID', lambda x: (x.count() / len(cluster_category[i])) * 100)
                ).reset_index())
            
        values = []
        labels = []

        for i in range(clusters):
            values.append(cluster_category_analysis[i]['Percentage'])
            labels.append(cluster_category_analysis[i]['Category'])

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            ax.pie(values[i], labels=labels[i], autopct='%1.1f%%', colors=sns.color_palette('pastel', len(labels[i])))
            ax.set_title(f'Most Spend per Category Segmentation by Cluster {i+1}', fontsize=13, weight='bold')

        fig.tight_layout()
        
    return fig

@st.cache_data
def plot_most_spend_seasonality_per_cluster(df):
    with _lock:
        sales_by_seasonality = df.groupby(['CustomerID', 'Cluster', 'Seasonality'])['Sales'].sum().unstack()
        most_spend_seasonality = sales_by_seasonality.idxmax(axis=1).reset_index()
        most_spend_seasonality.columns = ['CustomerID', 'Cluster', 'Seasonality']
        
        clusters = 4
        cluster_seasonality = []
        cluster_seasonality_analysis = []
        
        for i in range(clusters):
            cluster_seasonality.append(most_spend_seasonality[most_spend_seasonality['Cluster'] == i])
            cluster_seasonality_analysis.append(cluster_seasonality[i].groupby('Seasonality').agg(
                Count = ('CustomerID', 'count'),
                Percentage = ('CustomerID', lambda x: (x.count() / len(cluster_seasonality[i])) * 100)
                ).reset_index())
            
        values = []
        labels = []

        for i in range(clusters):
            values.append(cluster_seasonality_analysis[i]['Percentage'])
            labels.append(cluster_seasonality_analysis[i]['Seasonality'])

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            ax.pie(values[i], labels=labels[i], autopct='%1.1f%%', colors=sns.color_palette('pastel', len(labels[i])))
            ax.set_title(f'Most Spend per Seasonality Segmentation by Cluster {i+1}', fontsize=13, weight='bold')

        fig.tight_layout()
        
    return fig

# Streamlit App
st.set_page_config(layout="centered")
st.title("Online Retail UCI Clustering Data")

df = load_data()
st.dataframe(df)

# Menampilkan Bar Chart kolom numeric
fig = plot_numerical_columns(df)
st.pyplot(fig=fig, clear_figure=True, width='stretch')

# Menampilkan Bar Chart kolom kategorikal
fig = plot_categorical_columns(df)
st.pyplot(fig=fig, clear_figure=True, width='stretch')
# Menampilkan Heatmap korelasi antar data
fig = plot_correlation_heatmap(df)
st.pyplot(fig=fig, clear_figure=True, width='stretch')

# Menampilkan Pie Chart rasio Category tertinggi per Cluster
fig = plot_most_spend_category_per_cluster(df)
st.pyplot(fig=fig, clear_figure=True, width='stretch')

# Menampilkan Pie Chart rasio Seasonality tertinggi per Cluster
fig = plot_most_spend_seasonality_per_cluster(df)
st.pyplot(fig=fig, clear_figure=True, width='stretch')