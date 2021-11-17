import argparse
import json
import os
import pickle
import subprocess

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from search import ProductSearch, convert_text_into_sentences


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained vectorizer.
    with open(os.path.join(model_dir, "modules.pickle"), "rb") as f:
        modules = pickle.load(f)
    vectorizer = SentenceTransformer(modules=modules).eval().to(device)

    # Load the trained search engine.
    product_search = ProductSearch(index_path=os.path.join(model_dir, "product_search.pickle"))

    return {"vectorizer": vectorizer, "product_search": product_search}


def input_fn(input_data, content_type):
    assert content_type == "application/json"

    request = json.loads(input_data)
    return {"query": convert_text_into_sentences(request["query"]), "n_items": request["n_items"]}


def predict_fn(data, model):
    sentences, n_items = data["query"], data["n_items"]
    vectorizer, product_search = model["vectorizer"], model["product_search"]

    # Vectorize.
    with torch.no_grad():
        embeddings = np.array(vectorizer.encode(sentences), dtype=np.float32)

    # Search.
    prediction = product_search.search(embeddings, n_items=n_items)
    # Convert list into dict.
    prediction = {f"pred{str(i)}": pred for i, pred in enumerate(prediction)}

    return prediction


def output_fn(prediction, accept):
    return json.dumps(prediction), accept


def train(args):
    # Move the pretrained model into model_dir.
    subprocess.call(["cp", "modules.pickle", os.path.join(args.model_dir, "modules.pickle")])

    # Load datasets.
    reviews = pd.read_csv(os.path.join(args.train, "10000_review.csv"))
    sentences = pd.read_csv(os.path.join(args.train, "10000_sentence.csv"))
    embeddings = np.load(os.path.join(args.train, "10000_embedding.npy"))

    # Construct the search engine.
    product_search = ProductSearch(reviews, sentences, embeddings)
    product_search.save(os.path.join(args.model_dir, "product_search.pickle"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))

    train(parser.parse_args())
