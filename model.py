import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Ladataan ostohistoria ja tuotteet
history_df = pd.read_csv("history.csv", sep=";")
products_df = pd.read_csv("products.csv", sep=";")

# Poimitaan kategorioiden nimet
category_columns = products_df.columns[1:]

# Luodaan kategoriavektori ostoshistoriasta
def get_aggregated_history_vector(history_df, category_columns):

    # Haetaan ostettujen tuotteiden kategoriavektorit
    purchased_vectors = history_df[category_columns]

    # Tarkistetaan että historiassa on jotain
    if purchased_vectors.empty:
        print("No category vectors found in the user's history.")
        return None

    # Summataan vektorit kumulatiiviseksi vektoriksi, ja selvitetään mitkä kategoriat ilmenevät useimmin
    aggregated_vector = purchased_vectors.sum().values.reshape(1, -1)
    return aggregated_vector

# Haetaan tuotteet jotka parhaiten vastaavat kumulatiivista vektoria
def get_top_recommendations(aggregated_vector, products_df, category_columns, top_n=3):
    # Poimitaan tuotteiden kategoriat
    product_vectors = products_df[category_columns].values

    # Lasketaan kosinisamankaltaisuus historian tuotteiden ja kaikkien tuotteiden välillä
    similarity_scores = cosine_similarity(aggregated_vector, product_vectors)[0]

    # Sisällytetään samankaltaisuus dataframeen
    products_df['Similarity'] = similarity_scores

    # Järjestetään tuotteet samankaltaisuuden mukaan
    recommendations = products_df.sort_values(by='Similarity', ascending=False).head(top_n)

    return recommendations[['Product', 'Similarity']]

aggregated_vector = get_aggregated_history_vector(history_df, category_columns)

if aggregated_vector is not None:
    top_3_recommendations = get_top_recommendations(aggregated_vector, products_df, category_columns)

    print("Top 3 recommended products with relevancy scores:")
    print(top_3_recommendations)