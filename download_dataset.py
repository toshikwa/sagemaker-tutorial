import os
import sys
import argparse
import subprocess
import pandas as pd
import numpy as np
import gzip
import time
from urllib import request
import boto3
from rcis import convert_text_into_sentences, TextVectorizer

# Dataserver URL
DATASERVER_URL = 'https://s3.amazonaws.com/amazon-reviews-pds/tsv'


def progress(block_count, block_size, total_size):
    percentage = 100.0 * block_count * block_size / total_size
    mega_byte = total_size / 1024**2 * percentage / 100
    sys.stdout.write(
        f'- {percentage:.2f} % ({mega_byte:.2f} MB)\r')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str,
                        default='amazon_reviews_us_Toys_v1_00.tsv')
    parser.add_argument('--data_dir', type=str, default='dataset/toy')
    parser.add_argument('--num_reviews', type=int, default='50000')
    args = parser.parse_args()

    # Download dataset from the server.
    print('Downloading dataset from the server...')
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    dataset_url = os.path.join(DATASERVER_URL, f'{args.dataset_name}.gz')
    save_path = os.path.join(args.data_dir, f'{args.dataset_name}.gz')
    request.urlretrieve(dataset_url, save_path, reporthook=progress)
    print('Finished.\n ')

    # Uncompress file.
    print('Uncompressing...')
    subprocess.call(['gzip', '-df', save_path])
    print('Finished.\n')

    # Load dataset.
    print('Loading dataset...')
    df = pd.read_csv(
        os.path.join(args.data_dir, args.dataset_name), delimiter='\t',
        error_bad_lines=False, warn_bad_lines=False)
    info_keys = [
        'review_id', 'product_id', 'product_title', 'star_rating',
        'review_headline', 'review_body']
    review_df = df[:args.num_reviews][info_keys]
    print('Finished.\n')

    # Vectorize reviews.
    print('Vectorizing reviews...')
    review_id_list = []
    sentence_list = []
    for row in review_df.itertuples():
        sentences = convert_text_into_sentences(str(row.review_body))
        for sent in sentences:
            review_id_list.append(row.review_id)
            sentence_list.append(sent)
    vectorizer = TextVectorizer()
    embeddings = vectorizer.vectorize(sentence_list, batch_size=128)
    print('Finished.\n')

    # Save own dataset.
    review_df.to_csv(
        os.path.join(args.data_dir, f'{args.num_reviews}_review.csv'),
        index=False)
    sentence_df = pd.DataFrame({
        'review_id': review_id_list,
        'sentence': sentence_list})
    sentence_df.to_csv(
        os.path.join(args.data_dir, f'{args.num_reviews}_sentence.csv'),
        index=False)
    np.save(
        os.path.join(args.data_dir, f'{args.num_reviews}_embedding.npy'),
        np.array(embeddings))
