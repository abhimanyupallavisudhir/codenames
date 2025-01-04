import os
import numpy as np
from itertools import combinations
import gensim.downloader as api
import gensim
from typing import List, Tuple, Optional, Union
from scipy.spatial import ConvexHull
from sklearn.metrics import pairwise_distances

class WordEmbedding:
    def __init__(self, model_name: str = 'word2vec-google-news-300'):
        self.model_name = model_name
        embeddings_dir = 'embeddings'
        embeddings_path = os.path.join(embeddings_dir, f"{model_name}.model")
        
        if os.path.exists(embeddings_path):
            print(f"Loading embeddings from {embeddings_path}...")
            self.model = gensim.models.KeyedVectors.load(embeddings_path)
        else:
            print(f"Downloading {model_name} embeddings...")
            self.model = api.load(model_name)
            print(f"Saving embeddings to {embeddings_path}...")
            os.makedirs(embeddings_dir, exist_ok=True)
            self.model.save(embeddings_path)

    def get_vector(self, word: str) -> np.ndarray:
        """Get word embedding vector for a given word."""
        try:
            return self.model[word]
        except KeyError:
            raise ValueError(f"Word '{word}' not found in vocabulary")

def smallest_enclosing_ball(points: np.ndarray) -> Tuple[np.ndarray, float]:
    """Calculate the center and radius of the smallest ball containing all points."""
    # Use the average of points as an approximation of the center
    center = np.mean(points, axis=0)
    
    # Calculate distances from center to all points
    distances = np.linalg.norm(points - center, axis=1)
    radius = np.max(distances)
    
    return center, radius

def return_perfect_ball(A: List[str], B: List[str], embedder: WordEmbedding) -> Optional[Tuple[np.ndarray, float, List[str]]]:
    """
    Calculate smallest ball containing all points from A that contains no points from B.
    Returns (center, radius, contained_points) if successful, None otherwise.
    """
    try:
        # Get embeddings for all words
        A_vectors = np.array([embedder.get_vector(word) for word in A])
        B_vectors = np.array([embedder.get_vector(word) for word in B])
        
        # Calculate smallest enclosing ball
        center, radius = smallest_enclosing_ball(A_vectors)
        
        # Check if any B points are inside the ball
        B_distances = np.linalg.norm(B_vectors - center, axis=1)
        if np.all(B_distances > radius):
            return center, radius, A
        
        return None
    except ValueError as e:
        print(f"Warning: {str(e)}")
        return None

def return_best_ball(A: List[str], B: List[str], embedder: WordEmbedding) -> Optional[Tuple[np.ndarray, float, List[str]]]:
    """
    Try to find smallest ball containing subset of A points that contains no B points.
    Tries progressively smaller subsets until solution is found.
    """
    for remove_count in range(len(A)):
        # Try all possible combinations of current size
        for subset in combinations(A, len(A) - remove_count):
            result = return_perfect_ball(list(subset), B, embedder)
            if result is not None:
                return result
    return None

def sample_nearby(center: np.ndarray, k: int, embedder: WordEmbedding) -> List[str]:
    """Find k nearest dictionary words to the given vector."""
    return embedder.model.similar_by_vector(center, topn=k)

def main(A: List[str], B: List[str], k: int = 5) -> Tuple[List[str], List[str]]:
    """
    Main function to process word lists and return nearby words.
    Returns (list of nearby words, list of A points contained in ball)
    """
    embedder = WordEmbedding()
    
    # Get best ball
    result = return_best_ball(A, B, embedder)
    if result is None:
        return [], []
    
    center, radius, contained_points = result
    
    # make sure it always finds at least k words *not in A*
    k_ = k + len(A)

    # Get nearby words
    # Get nearby words and format them (bold if not in A)
    nearby_words = []
    for word, _ in sample_nearby(center, k_, embedder):
        if word not in A: # bold legal clues
            nearby_words.append(f"**{word}**")
        else:
            nearby_words.append(word)
    
    return nearby_words, contained_points

if __name__ == "__main__":
    # Example usage
    A = ["horn", "ivory", "soup", "cast", "telescope", "rainbow", "fish", "princess"]
    B = ["foam", "shampoo", "plane", "polish", "oven", "notre_dame", "forest", "microscope", "chocolate"]
    
    nearby_words, contained_points = main(A, B)
    print(f"Found ball containing points: {contained_points}")
    print(f"Nearby words: {nearby_words}")
    # A = ["horn", "ivory", "soup", "cast", "telescope", "rainbow", "fish", "princess"]
    # B = ["foam", "shampoo", "plane", "polish", "oven", "notre_dame", "forest", "microscope", "chocolate"]
