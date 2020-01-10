import os
import torch
from sentence_transformers import SentenceTransformer


class TextVectorizer:

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SentenceTransformer(
            'bert-base-nli-mean-tokens').to(self.device)
        self.model.eval()

    def vectorize(self, sentences, batch_size=8, verbose=True):
        assert isinstance(sentences, list) or isinstance(sentences, tuple)

        # Calculate the vector representations.
        with torch.no_grad():
            embeddings = self.model.encode(
                sentences, batch_size=batch_size, show_progress_bar=verbose)

        return embeddings
