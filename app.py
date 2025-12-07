import pandas as pd

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

import streamlit as st
from threading import RLock


_lock = RLock()
sns.set_style("whitegrid")

@st.cache_data
def load_data():
    path = "online_retail_uci_clustering.csv"
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

        n_cols = 2
        n_rows = -(-num_vars // n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 4))
        axes = axes.flatten()

        for i, column in enumerate(df[categorical_cols].columns):
            top_values = df[column].value_counts().head(12)
            
            sns.barplot(x=top_values.values, y=top_values.index, ax=axes[i], palette='plasma', hue=top_values.index, legend=False)
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

@st.cache_data
def calculate_rfm(df):
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    analysis_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

    Recency = ('InvoiceDate', lambda x: (analysis_date - x.max()).days)
    Frequency = ('InvoiceNo', 'nunique')
    Monetary = ('Sales', 'sum')
    Cluster = ('Cluster', 'first')
    Label = ('Label', 'first')

    df_rfm = df.groupby('CustomerID').agg(
        Recency=Recency,
        Frequency=Frequency,
        Monetary=Monetary,
        Cluster=Cluster,
        Label=Label
    ).reset_index()
        
    return df_rfm

@st.cache_data
def plot_3d_rfm(df_rfm):
    visual_3d = go.Scatter3d(
        x= df_rfm['Recency'],
        y= df_rfm['Frequency'],
        z= df_rfm['Monetary'],
        hovertemplate=(
            "Recency: %{x}<br>"
            "Frequency: %{y}<br>"
            "Monetary: %{z:.2f}"),
        mode='markers',
        marker=dict(
            color = df_rfm['Cluster'],
            size= 12,
            line=dict(
                color= df_rfm['Cluster'],
                width= 6
            ),
            opacity=0.8
        )
    )

    data = [visual_3d]
    layout = go.Layout(
        title= 'Visualisasi 3D',
        scene = dict(
                xaxis = dict(title  = 'Recency'),
                yaxis = dict(title  = 'Frequency'),
                zaxis = dict(title  = 'Monetary')
            )
    )

    fig = go.Figure(data=data, layout=layout)
        
    return fig

@st.cache_data
def analyse_cluster(df_rfm):
    cluster_analysis = df_rfm.groupby('Cluster').agg(
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Monetary=('Monetary', 'mean'),
        Count=('CustomerID', 'count'),
        Percentage=('CustomerID', lambda x: (x.count() / len(df_rfm)) * 100),
        Label=('Label', 'first')
    ).reset_index().sort_values(by='Cluster')
    
    return cluster_analysis

@st.cache_data
def plot_monthly_sales_per_cluster(df):
    df_months = df.copy()
    df_months['Months'] = pd.to_datetime(df_months['InvoiceDate']).dt.month
    monthly_sales_cluster = df_months.groupby(['Cluster', 'Months'])['Sales'].sum().reset_index()

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    sns.lineplot(x='Months', y='Sales', data=monthly_sales_cluster, palette='plasma', hue='Cluster', ax=ax)
    ax.set_title('Monthly Sales per Cluster', fontsize=13, weight='bold')
    ax.set_xticks(range(1, 13))

    fig.tight_layout()
    
    return fig

@st.cache_data
def plot_segmentation_and_monetary_per_cluster(df):
    cluster_analysis = analyse_cluster(df)
    values = []
    labels = []

    values.append(cluster_analysis['Percentage'])
    labels.append(cluster_analysis['Label'])
    values.append(cluster_analysis['Avg_Monetary'])
    labels.append(cluster_analysis['Label'])

    fig, axes = plt.subplots(1, 2, figsize=(12, 12))
    axes = axes.flatten()

    axes[0].pie(x=values[0], labels=labels[0], autopct='%1.1f%%', colors=sns.color_palette('pastel', len(labels[0])))
    axes[0].set_title('Customer Segmentation by Labels', fontsize=13, weight='bold')
    axes[1].pie(x=values[1], labels=labels[1], autopct='%1.1f%%', colors=sns.color_palette('pastel', len(labels[1])))
    axes[1].set_title('Sales Contribution per Labels', fontsize=13, weight='bold')

    fig.tight_layout()
    
    return fig

@st.cache_data
def plot_pairplot_rfm(df_rfm):
    columns = ['Recency', 'Frequency', 'Monetary']
    
    pairplot = sns.pairplot(data=df_rfm, vars=columns, hue='Cluster', palette='plasma')
    fig = pairplot.figure
    fig.suptitle('Pair Plot RFM per Cluster', fontsize=13, weight='bold')

    fig.tight_layout()
    
    return fig

# Streamlit Deployment
st.set_page_config(layout="wide", page_title="Peningkatan Efisiensi Strategi Pemasaran Berbasis Visualisasi Segmen Pelanggan")
st.html('''
    <style>
        [alt=Logo] {
            height: 3rem;
        }
    </style>
        ''')
st.logo("logo.png")

if 'page' not in st.session_state:
    st.session_state.page = "home"
    
df = load_data()
df_copy = df.copy()

with st.sidebar:
    st.write("## Menu")
    if st.button("Home", width='stretch'):
        st.session_state.page = "home"
    if st.button("Data Analysis", width='stretch'):
        st.session_state.page = "data_analysis"
    if st.button("RFM Analysis", width='stretch'):
        st.session_state.page = "rfm_analysis"

if st.session_state.page == "home":
    st.title("Peningkatan Efisiensi Strategi Pemasaran Berbasis Visualisasi Segmen Pelanggan", text_alignment='center')
    st.header("Dataset Information")
    st.write('''
        This is a transactional data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.The company mainly sells unique all-occasion gifts. Many customers of the company are wholesale
    ''')
    
    
if st.session_state.page == "data_analysis":
    st.title("Online Retail UCI Clustering Data")
    
    with st.container():
        cluster_filter = st.selectbox("Filter Cluster", options=['All', 'Loyal Customer', 'Regular Customer', 'Churn/One Time Spender', 'High Risk Churn'], width=300)
        
        if cluster_filter != 'All':
            df_copy = df_copy[df_copy['Label'] == cluster_filter]
            
        st.dataframe(df_copy)
        st.download_button(label="Download CSV", data=df.to_csv(index=False), file_name='online_retail_uci_clustering.csv')

    with st.container(width = 1200, horizontal_alignment='center'):
        # Menampilkan Bar Chart kolom numeric
        fig = plot_numerical_columns(df_copy)
        st.pyplot(fig=fig, clear_figure=True)
        expander = st.expander("See explanation")
        expander.write('''
        ''')

    with st.container(width = 1200, horizontal_alignment='center'):
        # Menampilkan Bar Chart kolom kategorikal
        fig = plot_categorical_columns(df_copy)
        st.pyplot(fig=fig, clear_figure=True)
        expander = st.expander("See explanation")
        expander.write('''
        ''')
    
    with st.container(width = 1200, horizontal_alignment='center'):
        # Menampilkan Heatmap korelasi antar data
        fig = plot_correlation_heatmap(df_copy)
        st.pyplot(fig=fig, clear_figure=True)
        expander = st.expander("See explanation")
        expander.write('''
        ''')

    with st.container(width = 1200, horizontal_alignment='center'):
        # Menampilkan Pie Chart rasio Category tertinggi per Cluster
        fig = plot_most_spend_category_per_cluster(df_copy)
        st.pyplot(fig=fig, clear_figure=True)
        expander = st.expander("See explanation")
        expander.write('''
        ''')

    with st.container(width = 1200, horizontal_alignment='center'):
        # Menampilkan Pie Chart rasio Seasonality tertinggi per Cluster
        fig = plot_most_spend_seasonality_per_cluster(df_copy)
        st.pyplot(fig=fig, clear_figure=True)
        expander = st.expander("See explanation")
        expander.write('''
        ''')
    
    with st.container(width = 1200, horizontal_alignment='center'):
        # Menampilkan Line Chart Monthly Sales per Cluster
        fig = plot_monthly_sales_per_cluster(df_copy)
        st.pyplot(fig=fig, clear_figure=True)
        expander = st.expander("See explanation")
        expander.write('''
        ''')
    
elif st.session_state.page == "rfm_analysis":
    df_rfm = calculate_rfm(df_copy)
    
    st.title("RFM Analysis")

    with st.container():
        cluster_filter_rfm = st.selectbox("Filter Cluster", options=['All', 'Loyal Customer', 'Regular Customer', 'Churn/One Time Spender', 'High Risk Churn'], width=300)
        if cluster_filter_rfm != 'All':
            df_rfm = df_rfm[df_rfm['Label'] == cluster_filter_rfm]
            
        st.dataframe(df_rfm)

    with st.container(width = 1200, horizontal_alignment='center'):
        # Menampilkan visualisasi 3D dari hasil Clustering
        fig = plot_3d_rfm(df_rfm)
        st.plotly_chart(figure_or_data=fig, height=900)
        expander = st.expander("See explanation")
        expander.write('''
        ''')

    with st.container(width = 1200, horizontal_alignment='center'):
        # Menampilkan Pie Chart segmentasi Customer per Cluster
        fig = plot_segmentation_and_monetary_per_cluster(df_rfm)
        st.pyplot(fig=fig, clear_figure=True)
        expander = st.expander("See explanation")
        expander.write('''
        ''')
    
    with st.container(width = 1200, horizontal_alignment='center'):
        # Menampilkan Pairplot RFM per Cluster
        fig = plot_pairplot_rfm(df_rfm)
        st.pyplot(fig=fig, clear_figure=True)
        expander = st.expander("See explanation")
        expander.write('''
        ''')