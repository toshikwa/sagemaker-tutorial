import copy
import faiss
from itertools import chain
import numpy as np
import pandas as pd
import dill
from sklearn.preprocessing import normalize

class ProductSearch:
    def __init__(self, reviews=None, sentences=None, embeddings=None, index_path=None):
        if index_path is not None:
            self.load(index_path)
            return
        assert reviews is not None, 'model_path or other data is required'
        self._reviews = reviews
        self._sentences = sentences
        # normalize for cosine distance
        self._embeddings = normalize(embeddings, axis=1).astype(np.float32)

        self._index = faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(self._embeddings)

    def search(self, embedding):
        # normalize for cosine distance
        normalized_embedding = np.atleast_2d(embedding)/np.linalg.norm(embedding)
        scores, indices = self._index.search(normalized_embedding, 5)
        scores, indices = scores[0], indices[0]
        review_ids = self._sentences.iloc[indices]['review_id']
        dicts = list(chain.from_iterable([self._reviews[self._reviews['review_id'] == review_id].to_dict('records') for review_id in review_ids]))
        for score, d in zip(scores, dicts):
            d['product_search_score'] = score
        return dicts

    def save(self, path):
        self._index = faiss.serialize_index(self._index)
        with open(path, 'wb') as f:
            dill.dump(self.__dict__, f)
        self._index = faiss.deserialize_index(self._index)

    def load(self, path):
        with open(path, 'rb') as f:
            tmp_dict = dill.load(f)

        tmp_dict['_index'] = faiss.deserialize_index(tmp_dict['_index'])
        self.__dict__.update(tmp_dict) 


if __name__ == "__main__":
    import os
    import pprint
    # prepare dataset
    src_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    datasetdir = os.path.join(src_root, 'dataset')
    reviews = pd.read_csv(os.path.join(datasetdir, '10000_review.csv'))
    sentences = pd.read_csv(os.path.join(datasetdir, '10000_sentence.csv'))
    embeddings = np.load(os.path.join(datasetdir, '10000_embedding.npy'))

    # construct model
    product_search = ProductSearch(reviews, sentences, embeddings)

    # example search
    search_results = product_search.search(embeddings[0])

    pprint.pprint(search_results)

    print("Saving model")
    save_path = os.path.join(src_root, 'source_dir', 'product_search.pkl')
    product_search.save(save_path)

    del product_search

    print("Loading saved model")
    product_search = ProductSearch(index_path=save_path)

    # example search
    search_results_loaded = product_search.search(embeddings[0])

    print("Search results compatible between saved and loaded models: ", search_results == search_results_loaded)