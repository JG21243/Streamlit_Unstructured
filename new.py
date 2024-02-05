from openai import OpenAI

client = OpenAI()
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine

# Function to generate embeddings using OpenAI's text-embedding-3-large model
def generate_embeddings(texts):
    """
    Generate embeddings for a list of texts using the OpenAI text-embedding-3-large model.

    Args:
        texts (list): A list of texts to generate embeddings for.

    Returns:
        list: A list of embeddings generated for each text.
    """
    embeddings = []
    for text in texts:
        response = client.embeddings.create(model="text-embedding-3-large",
        input=text)
        embeddings.append(response.data[0].embedding)
    return embeddings

# Function to calculate cosine similarity between two embeddings
def cosine_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)

# Function to plot a bar chart of similarity scores
def plot_similarity(text_pairs, scores):
    labels = [f"{i+1}. '{text_pairs[i][0]}' vs '{text_pairs[i][1]}'" for i in range(len(text_pairs))]
    plt.barh(labels, scores, color='skyblue')
    plt.xlabel('Cosine Similarity Score')
    plt.title('Text Similarity Comparison')
    plt.gca().invert_yaxis()  # Invert y-axis to have the first pair at the top
    plt.show()

# Example usage
text_pairs = [
    ("The quick brown fox jumps over the lazy dog", "A fast dark-colored fox leaps above a sluggish canine"),
    ("I love to play football", "Playing soccer is enjoyable for me"),
    ("The weather is sunny", "It is a rainy day")
]

# Generate embeddings for all texts
embeddings = generate_embeddings([text for pair in text_pairs for text in pair])

# Calculate similarity scores for each pair
scores = [cosine_similarity(embeddings[i*2], embeddings[i*2+1]) for i in range(len(text_pairs))]

# Plot the similarity scores
plot_similarity(text_pairs, scores)