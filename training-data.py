import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

# Load the data source into a returned dataframe variable
def load_training_data(file_path):
    # Read data source into pandas dataframe (df)
    df = pd.read_csv(file_path, nrows=2000)
    # Print head of df
    print(df.head())
    return df

def process_data(df, numeric, categorical):
    X = df[numeric + categorical]

    preprocess = ColumnTransformer([
        ('num', StandardScaler(), numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ])

    Xt = preprocess.fit_transform(X)
    print(Xt[0:5])
    return Xt

# Train a machine learning model using the global dataframe and return its output
def train_model(Xt, df, numeric):
    # Try different K values
    K_CANDIDATES = range(2,6)
    inertias, silhouettes = [], []
    for k in K_CANDIDATES:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(Xt)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(Xt, km.labels_))

    plt.plot(K_CANDIDATES, inertias, marker='o')
    plt.title('Elbow Method'); plt.xlabel('K'); plt.ylabel('Inertia'); plt.show()

    plt.plot(K_CANDIDATES, silhouettes, marker='o')
    plt.title('Silhouette Score'); plt.xlabel('K'); plt.ylabel('Score'); plt.show()

    # Fit final model (example: K=3)
    best_k = 4
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(Xt)
    df['cluster'] = labels

    # PCA projection for visualisation
    pca = PCA(n_components=2)
    Xt_pca = pca.fit_transform(Xt)
    plt.scatter(Xt_pca[:,0], Xt_pca[:,1], c=labels, cmap='rainbow', s=10)
    plt.title(f'K-Means Clusters (K={best_k})')
    plt.xlabel('PCA 1'); plt.ylabel('PCA 2'); plt.show()

    # Inspect cluster profiles
    print(df.groupby('cluster')[numeric].mean().round(2))
    pass

if __name__ == "__main__":
    # Configure pandas
    pd.set_option('display.max_columns', None)
    # Specify data source
    file_path = "credit_risk_dataset.csv"
    # Load training data into model
    data = load_training_data(file_path)
    numeric = ["person_age", "person_income", "person_emp_length", "loan_amnt", "loan_int_rate", "loan_status", "loan_percent_income", "cb_person_cred_hist_length"]
    categorical = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]
    numeric = [c for c in numeric if c in data.columns]
    categorical = [c for c in categorical if c in data.columns]
    preprocess = process_data(data, numeric, categorical)
    # output = train_model(preprocess, data, numeric)