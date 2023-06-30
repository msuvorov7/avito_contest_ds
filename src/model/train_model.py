import argparse
import logging
import os
import pickle
import sys

import pandas as pd
import torch
import yaml
from gensim.models import FastText
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
))

from src.model.model import CNNBaseline

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)

fileDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')


def collate_fn(batch) -> dict:
    """
    Обработчик батча перед входом в модель.
    Забивает предложения pad-токенами до длинны самого длинного
    предложения в батче
    :param batch: батч данных
    :return:
    """
    max_len = max(len(row['feature']) for row in batch)

    feature = torch.empty((len(batch), max_len), dtype=torch.long)
    target = torch.empty(len(batch), dtype=torch.long)

    for idx, row in enumerate(batch):
        to_pad = max_len - len(row['feature'])
        _feat = row['feature']
        feature[idx] = torch.cat((torch.tensor(_feat), torch.zeros(to_pad)))
        target[idx] = row['target']
    return {
        'feature': feature,
        'target': target,
    }


def validate(category: pd.Series, model, loader: DataLoader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(loader):

            text = batch['feature'].to('cpu')
            labels = batch['target'].view(-1).to('cpu')

            prediction = model(text)
            preds = F.softmax(prediction, dim=1)[:, 1]

            y_true += labels.cpu().detach().numpy().ravel().tolist()
            y_pred += preds.cpu().detach().numpy().ravel().tolist()

    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)

    roc = []
    for cat in category.unique():
        y_true_cat = y_true[category == cat]
        y_pred_cat = y_pred[category == cat]
        score = roc_auc_score(y_true_cat, y_pred_cat)
        roc.append(score)
        print(cat, score)

    logging.info(f'mean categoty roc_auc_score: {sum(roc) / len(roc)}')


def train(
    model: nn.Module,
    training_data_loader: DataLoader,
    validating_data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str,
    max_grad_norm: int = 2,
) -> (float, float, float):
    """
    Функция для обучения модели на одной этохе
    :param model: модель
    :param training_data_loader: набор для обучения
    :param validating_data_loader: набор для валидации
    :param criterion: функция потерь
    :param optimizer: оптимизатор функции потерь
    :param scheduler: планировщик скорости обучения
    :param device: обучение на gpu или cpu
    :param max_grad_norm: обрезка градиента
    :return:
    """
    train_loss = 0.0
    val_loss = 0.0

    model.train()
    for batch in tqdm(training_data_loader):
        text = batch['feature'].to(device)
        labels = batch['target'].view(-1).to(device)

        y_predict = model(text)
        loss = criterion(y_predict, labels)
        optimizer.zero_grad()
        train_loss += loss.item()
        loss.backward()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()

    train_loss /= len(training_data_loader)

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(validating_data_loader):

            text = batch['feature'].to(device)
            labels = batch['target'].view(-1).to(device)

            prediction = model(text)
            preds = F.softmax(prediction, dim=1)[:, 1]

            y_true += labels.cpu().detach().numpy().ravel().tolist()
            y_pred += preds.cpu().detach().numpy().ravel().tolist()

            loss = criterion(prediction, labels)

            val_loss += loss.item()

    val_loss /= len(validating_data_loader)

    val_roc = roc_auc_score(y_true, y_pred)
    return train_loss, val_loss, val_roc


def fit(model: nn.Module,
        training_data_loader: DataLoader,
        validating_data_loader: DataLoader,
        epochs: int,
        num_freeze_iter: int = 1
        ) -> (list, list):
    """
    Основной цикл обучения по эпохам
    :param model: модель
    :param training_data_loader: набор для обучения
    :param validating_data_loader: набор для валидации
    :param epochs: число эпох обучения
    :param num_freeze_iter: число эпох с заморозкой весов
    :return:
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=int(len(training_data_loader) + 1) * epochs)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_losses = []

    train_rocs = []
    val_rocs = []

    for epoch in range(epochs):
        if epoch >= num_freeze_iter:
            freeze_embeddings(model, True)
        else:
            freeze_embeddings(model, False)
        train_loss, val_loss, val_roc = train(model,
                                              training_data_loader,
                                              validating_data_loader,
                                              criterion,
                                              optimizer,
                                              scheduler,
                                              device)
        print()
        print('Epoch: {}, Training Loss: {}, Validation Loss: {}, ROC_AUC: {}'.format(epoch,
                                                                                      train_loss,
                                                                                      val_loss,
                                                                                      val_roc)
              )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        val_rocs.append(val_roc)

    return train_rocs, val_rocs


def freeze_embeddings(model: nn.Module, req_grad: bool = False):
    embeddings = model.embedding
    for c_p in embeddings.parameters():
        c_p.requires_grad = req_grad


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config', default='params.yaml', dest='config')
    arg_parser.add_argument('--epochs', default=1, type=int, dest='epochs')
    args = arg_parser.parse_args()

    with open(fileDir + args.config) as conf_file:
        config = yaml.safe_load(conf_file)

    model_path = fileDir + config['models']
    data_raw_dir = fileDir + config['data']['raw']
    data_processed_dir = fileDir + config['data']['processed']

    with open(f'{data_processed_dir}train_dataset.pkl', 'rb') as file:
        train_dataset = pickle.load(file)
    with open(f'{data_processed_dir}val_dataset.pkl', 'rb') as file:
        test_dataset = pickle.load(file)

    logging.info('datasets loaded')

    fasttext_model = FastText.load(model_path + 'fasttext_100.model')
    logging.info('fasttext model loaded')

    with open(data_processed_dir + 'vocab_to_ind.pkl', 'rb') as file:
        word_to_ind = pickle.load(file)
    logging.info('vocab_to_ind loaded')

    train_size = len(train_dataset)
    validation_size = int(0.3 * train_size)
    train_data, valid_data = random_split(train_dataset, [train_size - validation_size, validation_size],
                                          generator=torch.Generator().manual_seed(42)
                                          )

    train_loader = DataLoader(train_data, batch_size=32, collate_fn=collate_fn, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=32, collate_fn=collate_fn, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn, shuffle=False)

    model = CNNBaseline(
        vocab_size=len(word_to_ind),
        embedding_dim=100,
        in_channels=1,
        out_channels=128,
        output_dim=2,
        kernel_sizes=[3, 4, 5],
    )

    # перенос обученных эмбеддингов
    with torch.no_grad():
        for word, idx in word_to_ind.items():
            model.embedding.weight[idx] = torch.tensor(fasttext_model.wv[word])

    _, _ = fit(model, train_loader, valid_loader, args.epochs)

    torch.save(model, model_path + 'model.torch')

    categories = pd.read_csv(data_raw_dir + 'val.csv').category
    validate(categories, model, test_loader)
