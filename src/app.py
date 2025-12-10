# Import Library
import time
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

import streamlit as st
from threading import RLock

# Main Functions
_lock = RLock()
sns.set_style("whitegrid")

@st.cache_data
def load_data():
    path = "assets/online_retail_uci_clustering.csv"
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

@st.cache_data
def recommendation_system(df):
    items = df[['StockCode', 'Description', 'Category']]
    items = items.drop_duplicates().reset_index().drop('index', axis=1)
    items['Description'] = items['Description'].apply(lambda desc: desc.strip())
    items['Tags'] = items['Description'] + " " + items['Category']
    items['Tags'] = items['Tags'].apply(lambda x:x.lower().strip())

    cv = CountVectorizer(stop_words='english')
    vector = cv.fit_transform(items['Tags']).toarray()
    similarity = cosine_similarity(vector)
    
    sales_by_stockcode = df.groupby(['CustomerID', 'StockCode'])['Sales'].sum()
    favorite_items = sales_by_stockcode.groupby('CustomerID').nlargest(5).droplevel(level=1).reset_index()

    customer_dict = favorite_items.groupby('CustomerID')['StockCode'].apply(list).to_dict()
    
    return items, similarity, customer_dict

@st.cache_data
def recommend_item_description(item_name_full, top_n=5):
    data = ""
    index = items[items['Description'] == item_name_full].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    
    data = f"Karena Anda membeli {items.iloc[index]['Description']}, mungkin Anda tertarik: "
    
    for i in distances[1:top_n+1]:
        data += f"\n- {items.iloc[i[0]].Description} (Similarity: {i[1]:.4f})"
        
    return data
        
@st.cache_data
def recommend_for_single_customer(id_customer, top_n=5):
    data = ""
    
    if id_customer not in customer_dict:
        data += f"CustomerID {id_customer} tidak ditemukan"
        return data

    data += f"CustomerID: {id_customer}"

    stock_codes = customer_dict[id_customer]
    all_similarities = []

    for stock_code in stock_codes:
        index = items[items['StockCode'] == stock_code].index[0]
        distances = sorted(list(enumerate(similarity[index])))
        all_similarities.extend(distances)

        data += f"\n- Anda membeli {items.iloc[index]['Description']}"

    similarity_dict = {}

    for idx, sim in all_similarities:
        if idx in similarity_dict:
            similarity_dict[idx].append(sim)
        else:
            similarity_dict[idx] = [sim]

    avg_similarities = []

    for idx, sims in similarity_dict.items():
        if items.iloc[idx]['StockCode'] not in stock_codes:
            avg_similarities.append((idx, np.mean(sims)))

    avg_similarities = sorted(avg_similarities,reverse=True, key = lambda x: x[1])
    recommendations = []

    for i in range(top_n):
        idx, sim_score = avg_similarities[i]
        recommendations.append({
            'StockCode': items.iloc[idx]['StockCode'],
            'Description': items.iloc[idx]['Description'],
            'Similarity': sim_score
        })

    data += "\n\nRekomendasi untuk customer ini:"

    for i, rec in enumerate(recommendations):
        data += f"\n{i + 1}. {rec['Description']} (Similarity: {rec['Similarity']:.4f})"
        
    return data

def stream_data(data):
    for word in data.split(" "):
        yield word + " "
        time.sleep(0.02)

# Streamlit Deployment App
df = load_data()
df_copy = df.copy()

st.set_page_config(layout="wide", page_title="Peningkatan Efisiensi Strategi Pemasaran Berbasis Visualisasi Segmen Pelanggan", initial_sidebar_state="expanded")
st.html('''
    <style>
        [alt=Logo] {
            height: 4rem;
            margin-top: 2rem;
        }
    </style>
        ''')
st.logo("assets/logo.png")

if 'page' not in st.session_state:
    st.session_state.page = "home"

with st.sidebar:
    st.write("## Menu")
    
    if st.button("Home", width='stretch'):
        st.session_state.page = "home"
        
    if st.button("Data Analysis", width='stretch'):
        st.session_state.page = "data_analysis"
        
    if st.button("RFM Analysis", width='stretch'):
        st.session_state.page = "rfm_analysis"
        
    if st.button("Recommendation System", width='stretch'):
        st.session_state.page = "recommendation_system"

if st.session_state.page == "home":
    st.title("Peningkatan Efisiensi Strategi Pemasaran Berbasis Visualisasi Segmen Pelanggan", text_alignment='center')
    
    st.divider()
    
    st.header("Deskripsi Aplikasi")
    st.write('''
        - Aplikasi ini dirancang untuk membantu bisnis memahami perilaku pelanggan mereka melalui segmentasi yang didasarkan pada data pembelian historis secara. Aplikasi ini menyediakan berbagai visualisasi interaktif yang memungkinkan pengguna untuk menjelajahi karakteristik masing-masing segmen pelanggan. 
        - Pengguna dapat melihat distribusi pembelian, frekuensi pembelian, dan nilai pembelian rata-rata untuk setiap segmen. Selain itu, aplikasi ini juga menampilkan analisis RFM (Recency, Frequency, Monetary) yang memberikan wawasan mendalam tentang loyalitas pelanggan dan potensi mereka.
    ''')
    
    st.header("Deskripsi Dataset")
    st.write('''
        - Dataset yang digunakan dalam aplikasi ini adalah "Online Retail UCI Clustering Data" yang berisi informasi transaksi pembelian dari sebuah toko retail online. Dataset ini mencakup berbagai atribut seperti InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country, Category, Seasonality, Sales, Cluster, dan Label.
        - Data ini telah diproses dan dianalisis menggunakan teknik clustering untuk mengelompokkan pelanggan berdasarkan pola pembelian mereka. Hasil clustering ini kemudian digunakan untuk membuat visualisasi yang membantu dalam memahami karakteristik masing-masing segmen pelanggan.
    ''')
    
    st.divider()
    
    st.subheader("Repositori GitHub")
    st.write("Link: [asah-capstone](https://github.com/ShiftorTheOrca/asah-capstone)")
    
    st.subheader("Contributors")
    st.write('''
        - [Leonardo Sanjaya | M320D5Y0992 - ShiforTheOrca](https://github.com/ShiftorTheOrca)
        - [Nelson Ahli | M320D5Y1490 - nelsooooon](https://github.com/nelsooooon)
        - [Steven Gunawan | M320D5Y1870 - veeennn](https://github.com/veeennn)
    ''')
    
elif st.session_state.page == "data_analysis":
    st.title("Online Retail UCI Clustering Data")
    
    with st.container():
        cluster_filter = st.selectbox("Filter Cluster", options=['All', 'Loyal Customer', 'Regular Customer', 'Churn/One Time Spender', 'High Risk Churn'], width=300)
        
        if cluster_filter != 'All':
            df_copy = df_copy[df_copy['Label'] == cluster_filter]
            
        st.dataframe(df_copy)
        st.download_button(label="Download CSV", data=df.to_csv(index=False), file_name='online_retail_uci_clustering.csv')
        
    tab_numerical, tab_categorical, tab_heatmap, tab_analysis = st.tabs(["Numerical Columns", "Categorical Columns", "Correlation Heatmap", "Feature Analysis"])    

    # Menampilkan Bar Chart kolom numeric
    with tab_numerical:
        st.markdown("### Box Plot Numerical Columns", text_alignment='center')
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            fig = plot_numerical_columns(df_copy)
            st.pyplot(fig=fig, clear_figure=True)
            expander = st.expander("See explanation")
            expander.write('''
                - **InvoiceNo**: Nomor unik untuk setiap transaksi yang dilakukan oleh pelanggan.
                
                - **CustomerID**: Kode unik untuk setiap pelanggan.
                
                - **Quantity**: Jumlah unit produk yang dibeli dalam setiap transaksi. Terdapat beberapa Customer yang melakukan pembelian dengan Quantity di atas rata-rata, hal ini dapat diindikasikan sebagai pembelian grosir.
                
                - **UnitPrice**: Harga per unit produk yang dibeli. Terdapat beberapa Customer yang melakukan pembelian dengan UnitPrice di atas rata-rata, hal ini dapat diindikasikan sebagai pembelian produk premium.
                
                - **Sales**: Total penjualan yang dihasilkan dari setiap transaksi (Quantity x UnitPrice). Terdapat beberapa Customer yang menghasilkan Sales di atas rata-rata, hal ini dapat diindikasikan sebagai high-value customer.
                
                - **Cluster**: Hasil segmentasi pelanggan berdasarkan analisis clustering.
            ''')

    # Menampilkan Bar Chart kolom kategorikal
    with tab_categorical:
        st.markdown("### Bar Chart Categorical Columns", text_alignment='center')
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            fig = plot_categorical_columns(df_copy)
            st.pyplot(fig=fig, clear_figure=True)
            expander = st.expander("See explanation")
            expander.write('''
                - **StockCode**: Kode unik untuk setiap produk yang dijual.
                
                - **Description**: Deskripsi singkat tentang produk yang dijual.
                
                - **InvoiceDate**: Tanggal dan waktu ketika transaksi dilakukan.
                
                - **Country**: Negara tempat pelanggan berada. Analisis dilakukan pada pelanggan dari United Kingdom saja.
                
                - **Category**: Kategori produk yang dibeli oleh pelanggan. Terdapat beberapa kategori produk yang paling banyak dibeli oleh pelanggan, seperti 'decoration', 'kitchen', dan 'bags'.
                
                - **Seasonality**: Musim atau periode waktu tertentu ketika pembelian dilakukan (misalnya, Winter, Fall, dsb.). Sebagian besar pembelian dilakukan pada musim Fall.
                
                - **Label**: Kategori segmentasi pelanggan berdasarkan hasil clustering (Loyal Customer, Regular Customer, Churn/One Time Spender, High Risk Churn).
            ''')
    
    # Menampilkan Heatmap korelasi antar data
    with tab_heatmap:
        st.markdown("### Heatmap", text_alignment='center')
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            fig = plot_correlation_heatmap(df_copy)
            st.pyplot(fig=fig, clear_figure=True)
            expander = st.expander("See explanation")
            expander.write('''
                - Terdapat korelasi **positif** yang kuat antara kolom Quantity dan Sales, yang menunjukkan bahwa semakin banyak unit produk yang dibeli, semakin tinggi total penjualan yang dihasilkan.
                
                - Terdapat korelasi **negatif** antara Seasonality dan InvoiceDate, yang mengindikasikan bahwa pembelian cenderung menurun seiring berjalannya waktu dalam dataset ini.
                
                - Kolom Cluster tidak menunjukkan korelasi yang signifikan dengan kolom numerik lainnya, yang mengindikasikan bahwa segmentasi pelanggan berdasarkan clustering tidak secara langsung terkait dengan variabel-variabel numerik dalam dataset ini.
            ''')

    with tab_analysis:
        st.markdown("### Feature Analysis", text_alignment='center')
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            # Menampilkan Pie Chart rasio Category tertinggi per Cluster
            fig = plot_most_spend_category_per_cluster(df_copy)
            st.pyplot(fig=fig, clear_figure=True)
            expander = st.expander("See explanation")
            expander.write('''
                - **Cluster 1 (Loyal Customer)**: Pelanggan dalam cluster ini cenderung menghabiskan sebagian besar pembelian mereka pada kategori 'decoration' (55.3%) dan 'bags' (23.3%).
                - **Cluster 2 (Regular Customer)**: Pelanggan dalam cluster ini cenderung menghabiskan sebagian besar pembelian mereka pada kategori 'decoration' (55.4%), 'bags' (15.3%), dan 'kitchen' (12.5%) sebagai kategori utama.
                - **Cluster 3 (Churn/One Time Spender)**: Pelanggan dalam cluster menunjukkan preferensi yang lebih beragam, dengan kategori 'decoration' (40.7%), 'kitchen' (12.9%), dan 'others' (11.2%).
                - **Cluster 4 (High Risk Churn)**: Pelanggan dalam cluster ini cenderung menghabiskan sebagian besar pembelian mereka pada kategori 'decoration' (50.5%), 'kitchen' (12.6%), dan 'bags' (10.8%) sebagai kategori utama.
            ''')
            
            # Menampilkan Line Chart Monthly Sales per Cluster
            fig = plot_monthly_sales_per_cluster(df_copy)
            st.pyplot(fig=fig, clear_figure=True)
            expander = st.expander("See explanation")
            expander.write('''
                - **Cluster 0 (Loyal Customer)**: Menunjukkan tren penjualan yang tinggi dan relatif stabil sepanjang tahun dengan puncak penjualan pada bulan September
                - **Cluster 1 (Regular Customer)**: Menunjukkan tren penjualan yang meningkat secara signifikan pada bulan November, yang mungkin terkait dengan musim liburan.
                - **Cluster 2 (Churn/One Time Spender)**: Menunjukkan tren penjualan sangat rendah dan fluktuatif sepanjang tahun dengan puncak penjualan pada bulan September, Oktober, dan November.
                - **Cluster 3 (High Risk Churn)**: Menunjukkan tren penjualan yang sedikit fluktuatif dan dibawah Regular Customer sepanjang tahun dengan puncak penjualan pada bulan November, yang mungkin terkait dengan musim liburan.
            ''')
    
            # Menampilkan Pie Chart rasio Seasonality tertinggi per Cluster
            fig = plot_most_spend_seasonality_per_cluster(df_copy)
            st.pyplot(fig=fig, clear_figure=True)
            expander = st.expander("See explanation")
            expander.write('''
                - **Cluster 1 (Loyal Customer)**: Pelanggan dalam cluster ini cenderung menghabiskan sebagian besar pembelian mereka pada musim Fall (56.3%) dan Spring (18.4%).
                - **Cluster 2 (Regular Customer)**: Pelanggan dalam cluster ini cenderung menghabiskan sebagian besar pembelian mereka pada musim Fall (49.9%), Summer (17.8%), dan Winter (17.2%).
                - **Cluster 3 (Churn/One Time Spender)**: Pelanggan dalam cluster ini cenderung menghabiskan sebagian besar pembelian mereka pada musim Fall (45.0%) dan Spring (21.4%).
                - **Cluster 4 (High Risk Churn)**: Pelanggan dalam cluster ini cenderung menghabiskan sebagian besar pembelian mereka pada musim Fall (46.5%), Spring dan Summer (19.2%).
            ''')
        
elif st.session_state.page == "rfm_analysis":
    df_rfm = calculate_rfm(df_copy)
    
    st.title("RFM Analysis")

    with st.container():
        cluster_filter_rfm = st.selectbox("Filter Cluster", options=['All', 'Loyal Customer', 'Regular Customer', 'Churn/One Time Spender', 'High Risk Churn'], width=300)
        if cluster_filter_rfm != 'All':
            df_rfm = df_rfm[df_rfm['Label'] == cluster_filter_rfm]
            
        st.dataframe(df_rfm)

    tab_rfm_3d, tab_cluster_analysis= st.tabs(["3D RFM Visualization", "Cluster Analysis"])
    
    # Menampilkan visualisasi 3D dari hasil Clustering
    with tab_rfm_3d:
        st.markdown("### 3D RFM Visualization", text_alignment='center')
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            fig = plot_3d_rfm(df_rfm)
            st.plotly_chart(figure_or_data=fig, height=900)
            expander = st.expander("See explanation")
            expander.write('''
                - Visualisasi 3D ini menampilkan segmentasi pelanggan berdasarkan metrik RFM (Recency, Frequency, Monetary). Masing-masing titik mewakili seorang pelanggan, dan warna titik menunjukkan cluster atau segmen pelanggan yang berbeda berdasarkan hasil analisis clustering.
                - Sumbu X mewakili Recency (berapa lama sejak pembelian terakhir), sumbu Y mewakili Frequency (seberapa sering pelanggan melakukan pembelian), dan sumbu Z mewakili Monetary (total nilai pembelian pelanggan).
            ''')

    with tab_cluster_analysis:
        # Menampilkan Pie Chart segmentasi Customer per Cluster
        st.markdown("### Cluster Analysis", text_alignment='center')
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            fig = plot_segmentation_and_monetary_per_cluster(df_rfm)
            st.pyplot(fig=fig, clear_figure=True)
            expander = st.expander("See explanation")
            expander.write('''
                - Pie chart pertama menunjukkan proporsi pelanggan dalam setiap segmen berdasarkan label cluster [Loyal Customer (2.7%), Regular Customer (12.3%), Churn/One Time Spender (54.1%), High Risk Churn (30.9%)]. Hal ini memberikan gambaran tentang distribusi pelanggan di berbagai segmen.
                - Pie chart kedua menunjukkan kontribusi penjualan dari masing-masing segmen pelanggan [Loyal Customer (75.9%), Regular Customer (16.2%), Churn/One Time Spender (1.9%), High Risk Churn (6.0%)]. Hal ini membantu dalam memahami segmen mana yang memberikan kontribusi terbesar terhadap total penjualan.
            ''')
    
        # Menampilkan Pairplot RFM per Cluster
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            fig = plot_pairplot_rfm(df_rfm)
            st.pyplot(fig=fig, clear_figure=True)
            expander = st.expander("See explanation")
            expander.write('''
                - Pairplot ini menampilkan hubungan antar metrik RFM (Recency, Frequency, Monetary) untuk setiap cluster pelanggan. Setiap titik mewakili seorang pelanggan, dan warna titik menunjukkan cluster atau segmen pelanggan yang berbeda.
                - Dari visualisasi ini, kita dapat mengamati pola distribusi pelanggan dalam setiap cluster berdasarkan metrik RFM mereka. Misalnya, pelanggan dalam cluster "Loyal Customer" cenderung memiliki nilai Frequency dan Monetary yang lebih tinggi dibandingkan dengan cluster lainnya.
                - Karakteristik masing-masing segmen pelanggan berdasarkan perilaku pembelian mereka dapat diidentifikasi, yang dapat membantu dalam merancang strategi pemasaran yang lebih efektif dan personalisasi penawaran kepada pelanggan.
            ''')
            
elif st.session_state.page == "recommendation_system":
    items, similarity, customer_dict = recommendation_system(df)
    
    st.title("Recommendation System")
    
    with st.container():
        st.dataframe(items)

    tab_description, tab_customer = st.tabs(["Recommend Item Description", "Recommend for Single Customer"])
    
    with tab_description:
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            st.subheader("Recommend Item Description")
            item_name_full = st.selectbox("Select, search, or randomize item Description", options=items['Description'])
            left, right = st.columns(2)
            
            with left:
                if st.button("Recommend Item", key="recommend_item", width='stretch'):
                    st.session_state.item_name_full = item_name_full
                    st.session_state.item_name_text = recommend_item_description(item_name_full=st.session_state.item_name_full, top_n=5)
                    
                    st.write_stream((stream_data(st.session_state.item_name_text)))
                    st.rerun()
                    
                if 'item_name_full' in st.session_state:
                    st.write(st.session_state.item_name_text)
                    
            with right:
                if st.button("Randomize Item", key="randomize_item", width='stretch'):
                    random_item = np.random.choice(items['Description'])
                    st.session_state.random_item = random_item
                    st.session_state.random_item_text = recommend_item_description(item_name_full=st.session_state.random_item, top_n=5)
                    
                    st.write_stream((stream_data(st.session_state.random_item_text)))
                    st.rerun()
                    
                if 'random_item' in st.session_state:
                    st.write(st.session_state.random_item_text)
            
    with tab_customer:
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            st.subheader("Recommend for Single Customer")
            id_customer = st.selectbox("Select, search, or randomize CustomerID", options=list(customer_dict.keys()))
            left, right = st.columns(2)
            
            with left:
                if st.button("Recommend for Customer", key="recommend_customer", width='stretch'):
                    st.session_state.id_customer = id_customer
                    st.session_state.id_customer_text = recommend_for_single_customer(id_customer=st.session_state.id_customer, top_n=5)
                    
                    st.write_stream(stream_data(st.session_state.id_customer_text))
                    st.rerun()
                    
                if 'id_customer' in st.session_state:
                    st.write(st.session_state.id_customer_text)
            
            with right:
                if st.button("Randomize CustomerID", key="randomize_customer", width='stretch'):
                    random_id = np.random.choice(list(customer_dict.keys()))
                    st.session_state.random_id = random_id
                    st.session_state.random_id_text = recommend_for_single_customer(id_customer=st.session_state.random_id, top_n=5)
                    
                    st.write_stream(stream_data(st.session_state.random_id_text))
                    st.rerun()
                    
                if 'random_id' in st.session_state:
                    st.write(st.session_state.random_id_text)