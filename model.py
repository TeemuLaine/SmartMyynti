import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load user purchase history and product catalog
history_df = pd.read_csv("history.csv", sep=";")  # Use the updated history file with categories
products_df = pd.read_csv("products.csv", sep=";")  # Use your original products CSV

# Extract category columns
category_columns = products_df.columns[1:]  # Assuming the first column is 'Product'

# Step 1: Create a cumulative category vector for the user's purchase history
def get_aggregated_history_vector(history_df, category_columns):
    history_products = history_df['Product'].tolist()

    # Get the category vectors for the purchased products directly from history_df
    purchased_vectors = history_df[category_columns]  # Use the category columns from the history

    # Check if there are any purchased vectors
    if purchased_vectors.empty:
        print("No category vectors found in the user's history.")
        return None

    # Sum the category vectors of purchased products to create an aggregated vector
    aggregated_vector = purchased_vectors.sum().values.reshape(1, -1)
    return aggregated_vector

# Step 2: Find products in products_df that match the aggregated history vector
def get_top_recommendations(aggregated_vector, products_df, category_columns, top_n=3):
    # Extract product vectors (excluding the product names)
    product_vectors = products_df[category_columns].values

    # Compute cosine similarity between the aggregated vector and all product vectors
    similarity_scores = cosine_similarity(aggregated_vector, product_vectors)[0]

    # Attach similarity scores to products
    products_df['Similarity'] = similarity_scores

    # Sort products by similarity and get the top N recommendations
    recommendations = products_df.sort_values(by='Similarity', ascending=False).head(top_n)

    return recommendations[['Product', 'Similarity']]

# Step 3: Load user's aggregated history vector and recommend top 3 products
aggregated_vector = get_aggregated_history_vector(history_df, category_columns)

# Check if we got a valid aggregated vector
if aggregated_vector is not None:
    top_3_recommendations = get_top_recommendations(aggregated_vector, products_df, category_columns)

    print("Top 3 recommended products with relevancy scores:")
    print(top_3_recommendations)