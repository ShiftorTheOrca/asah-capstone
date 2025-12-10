import numpy as np
import pandas as pd
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv('../assets/online_retail_uci_clustering.csv')

# Ambil CustomerID & jumlah rekomendasi
idcustomer = float(input("Customer ID: "))
top_n = int(input("Jumlah rekomendasi: "))

# Feature Engineering
items = df[['StockCode', 'Description', 'Category']]
items = items.drop_duplicates().reset_index().drop('index', axis=1)
items.duplicated().sum()

items['Description'] = items['Description'].apply(lambda desc: desc.strip())
items['Tags'] = items['Description'] + " " + items['Category']
items['Tags'] = items['Tags'].apply(lambda x:x.lower().strip())

# Vectorizer
cv = CountVectorizer(stop_words='english')
vector = cv.fit_transform(items['Tags']).toarray()
similarity = cosine_similarity(vector)

# Inference
sales_by_stockcode = df.groupby(['CustomerID', 'StockCode'])['Sales'].sum()
favorite_items = sales_by_stockcode.groupby('CustomerID').nlargest(5).droplevel(level=1).reset_index()
customer_dict = favorite_items.groupby('CustomerID')['StockCode'].apply(list).to_dict()

if idcustomer in customer_dict:
    stock_codes = customer_dict[idcustomer]
    all_similarities = []

    for stock_code in stock_codes:
        index = items[items['StockCode'] == stock_code].index[0]
        distances = sorted(list(enumerate(similarity[index])))
        all_similarities.extend(distances)

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
        
    for i, rec in enumerate(recommendations):
        print(f"{i + 1}. {rec['Description']} (Similarity: {rec['Similarity']:.4f})")