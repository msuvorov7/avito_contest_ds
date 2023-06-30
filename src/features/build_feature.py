import argparse
import logging
import os
import pickle
import sys
from collections import Counter

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


def build_feature(df: pd.DataFrame, model_path: str, data_processed_dir: str, mode: str) -> tuple:
    text = df['title'] + ' ' + df['description']

    tokenizer = WordPunctTokenizer()
    tokens = text.apply(lambda item: tokenizer.tokenize(item))
    tokens = tokens.apply(lambda item: [word.lower() for word in item])

    if mode == 'train':
        counter = Counter()
        for word in tokens:
            counter.update(word)

        counter = counter.most_common(55_000)
        counter = list(filter(lambda item: item[1] > 20, counter))

        vocabulary = ['<PAD>', '<UNK>']
        vocabulary += [key for key, _ in counter]

        ind_to_word = dict(enumerate(vocabulary))
        word_to_ind = {value: key for key, value in ind_to_word.items()}

        with open(data_processed_dir + 'vocab_to_ind.pkl', 'wb') as file:
            pickle.dump(word_to_ind, file)
        logging.info('vocab_to_ind saved')

        fasttext_model = FastText(tokens, vector_size=100, min_n=3, max_n=4, window=3, epochs=3)
        fasttext_model.save(model_path + 'fasttext_100.model')
        logging.info('fasttext model saved')

    else:
        with open(data_processed_dir + 'vocab_to_ind.pkl', 'rb') as file:
            word_to_ind = pickle.load(file)

    return tokens, df['is_bad'], word_to_ind


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

    features, targets, word_to_ind = build_feature(data, model_path, data_processed_dir, args.mode)
    dataset = AvitoDataset(features, targets, word_to_ind)

    with open(data_processed_dir + f'{args.mode}_dataset.pkl', 'wb') as file:
        pickle.dump(dataset, file)

    logging.info(f'dataset saved in {data_processed_dir}')
