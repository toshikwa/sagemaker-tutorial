import os
import sys
import argparse
import subprocess
import json
import pickle
import numpy as np
import pandas as pd
import torch
from  torch import nn
from sentence_transformers import SentenceTransformer
from search import ProductSearch, convert_text_into_sentences


def model_fn(model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load trained vectorizer.
    with open(os.path.join(model_dir, 'modules.pickle'),'rb') as f:
        modules = pickle.load(f)
    vectorizer = SentenceTransformer(modules=modules).eval()

    # Load trained search engine.
    product_search = ProductSearch(
        index_path=os.path.join(model_dir, 'product_search.pickle'))

    return {
        'vectorizer': vectorizer.to(device),
        'product_search': product_search
    }


def input_fn(input_data, content_type):
    assert content_type == 'application/json'

    request = json.loads(input_data)
    return convert_text_into_sentences(request['query'])


def predict_fn(data, model):
    sentences = data
    vectorizer, product_search = model['vectorizer'], model['product_search']

    # Vectorize.
    with torch.no_grad():
        embeddings = np.array(vectorizer.encode(sentences), dtype=np.float32)

    # Search.
    prediction = product_search.search(embeddings)
    prediction = {f'pred{str(i)}': pred for i, pred in enumerate(prediction)}
    
    # Convert np.float32 into np.float64 to serialize.
    for key in prediction.keys():
        prediction[key]['product_search_score'] =\
            prediction[key]['product_search_score'].astype(np.float64)

    return prediction


def output_fn(prediction, accept):
    return json.dumps(prediction), accept


def train(args):
    # Move the pretrained model into model_dir.
    subprocess.call([
        'cp', 'modules.pickle',
        os.path.join(args.model_dir, 'modules.pickle')])

    # Load datasets.
    reviews = pd.read_csv(os.path.join(args.train, '10000_review.csv'))
    sentences = pd.read_csv(os.path.join(args.train, '10000_sentence.csv'))
    embeddings = np.load(os.path.join(args.train, '10000_embedding.npy'))

    # Construct the search engine.
    product_search = ProductSearch(reviews, sentences, embeddings)
    product_search.save(
        os.path.join(args.model_dir, 'product_search.pickle'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))

    train(parser.parse_args())
