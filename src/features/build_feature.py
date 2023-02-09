import argparse
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
from gensim.models import FastText

import yaml
from nltk import WordPunctTokenizer


sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
))

from src.utils.udf import mem_usage
from src.features.dataset import AvitoDataset


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)

fileDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')


def build_feature(df: pd.DataFrame, model_path: str, mode: str) -> tuple:
    text = df['title'] + ' ' + df['description']
    text = text.apply(lambda item: item.lower())

    tokenizer = WordPunctTokenizer()
    tokens = text.apply(lambda item: tokenizer.tokenize(item))

    if mode == 'train':
        fasttext_model = FastText(tokens, vector_size=100, min_n=3, max_n=5, window=3, epochs=3)
        fasttext_model.save(model_path + 'fasttext_100.model')
        logging.info('fasttext model saved')
    # else:
    #     fasttext_model = FastText.load(model_path + 'fasttext_100.model')
    #     logging.info('fasttext model loaded')

    # feats = tokens.apply(lambda sentence: np.array([fasttext_model.wv[item] for item in sentence]))

    return tokens, df['is_bad']


def download_frame(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    logging.info(
        f'{int(df.is_bad.sum())} positive class '
        f'of {len(df)} labels ({(df.is_bad.sum() / len(df) * 100).round(1)}%)'
    )
    logging.info(f'dataframe size: {mem_usage(df)}')

    df.reset_index(inplace=True, drop=True)
    logging.info(f'dataframe shape: {df.shape}')

    return df


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--mode', default='train', dest='mode', type=str)
    arg_parser.add_argument('--config', default='params.yaml', dest='config')
    args = arg_parser.parse_args()

    if args.mode not in ('train', 'val'):
        raise NotImplementedError

    with open(args.config) as conf_file:
        config = yaml.safe_load(conf_file)

    model_path = fileDir + config['models']
    data_raw_dir = fileDir + config['data']['raw']
    data_processed_dir = fileDir + config['data']['processed']

    data = download_frame(data_raw_dir + args.mode + '.csv')

    features, targets = build_feature(data, model_path, args.mode)
    dataset = AvitoDataset(features, targets)

    with open(data_processed_dir + f'{args.mode}_dataset.pkl', 'wb') as file:
        pickle.dump(dataset, file)

    logging.info(f'dataset saved in {data_processed_dir}')
