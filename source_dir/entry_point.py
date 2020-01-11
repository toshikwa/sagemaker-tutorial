import os
import sys
import argparse
import subprocess
import json
import pickle
import sagemaker_containers
import numpy as np
import torch
from  torch import nn
from sentence_transformers import SentenceTransformer


def model_fn(model_dir):
    sys.path.append(model_dir)
    from node import node

    # Load trained vectorizer.
    with open(os.path.join(model_dir, 'modules.pickle'),'rb') as f:
        modules = pickle.load(f)
    vectorizer = SentenceTransformer(modules=modules).eval()

    # Load trained tree.
    with open(os.path.join(model_dir, 'tree.pkl'), 'rb') as f:
        tree = pickle.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return {
        'vectorizer': vectorizer.to(device),
        'tree': tree
    }


def input_fn(input_data, content_type):
    assert content_type == 'application/json'

    from node.utils import convert_text_into_sentences

    request = json.loads(input_data)
    return {
        'sentences': convert_text_into_sentences(request['query']),
        'n_items': request['n_items']
    }


def predict_fn(data, model):
    sentences, n_items = data['sentences'], data['n_items']
    vectorizer, tree = model['vectorizer'], model['tree']

    # Vectorize.
    with torch.no_grad():
        embeddings = np.array(vectorizer.encode(sentences), dtype=np.float32)

    # Search.
    prediction = tree.binary_search(embeddings[0], n_items)
    prediction = {f'pred{str(i)}': pred for i, pred in enumerate(prediction)}

    return prediction


def output_fn(prediction, accept):
    return json.dumps(prediction), accept


def train(args):
    # Move own modules into model_dir.
    subprocess.call(['cp', '-r', 'node', os.path.join(args.model_dir, 'node')])
    subprocess.call(['cp', 'modules.pickle', os.path.join(args.model_dir, 'modules.pickle')])

    # We use pretrained tree now.
    with open(os.path.join('tree.pkl'), 'rb') as f:
        tree = pickle.load(f)

    with open(os.path.join(args.model_dir, 'tree.pkl'), 'wb') as f:
        pickle.dump(tree, f)


if __name__ == '__main__':
    from node import node

    parser = argparse.ArgumentParser()

    env = sagemaker_containers.training_env()
    parser.add_argument('--hosts', type=list, default=env.hosts)
    parser.add_argument('--current-host', type=str, default=env.current_host)
    parser.add_argument('--model-dir', type=str, default=env.model_dir)
    parser.add_argument('--data-dir', type=str, default=env.channel_input_dirs.get('training'))
    parser.add_argument('--num-gpus', type=int, default=env.num_gpus)

    train(parser.parse_args())
