import argparse
import os
from functools import reduce
import shutil
from time import sleep
import time

import torch
import random
from typing import Callable, List, Optional

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset, download_url
import os.path as osp
import json
import pandas as pd
from torch_geometric.loader import DataLoader, DataListLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, RGCNConv, GINEConv, MLP, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from tqdm import tqdm
import logging


class GPClassification(InMemoryDataset):
    def __init__(self, root: str,
                 dataset_name: str = "GPClassificationTest_01",
                 graph_file: str = "./data/raw/plain-graphs_of_size_1.json",
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):

        self.dataset_name = dataset_name
        self.graph_file = graph_file

        super().__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0])
        self.node_dict = torch.load(self.processed_paths[1])
        self.rel_dict = torch.load(self.processed_paths[2])

        self.num_relations = len(self.rel_dict.keys())

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ['graphs.pt', 'node_dict.pt', 'rel_dict.pt']

    def download(self):
        pass

    def process(self):
        data_list, node_dict, rel_dict = [], {}, {}

        graph_dict = json.load(open(osp.join(self.graph_file), 'r'))

        all_node_terms = set()
        all_rel_terms = set()
        for graph in graph_dict.values():
            for triple in graph:
                all_node_terms.add(triple[0])
                all_rel_terms.add(triple[1])
                all_node_terms.add(triple[2])

        self.num_nodes = len(all_node_terms)
        self.num_relations = len(all_rel_terms)

        node_dict = {node: i for i, node in enumerate(all_node_terms)}

        rel_dict = {edge: i for i, edge in enumerate(all_rel_terms)}

        x = torch.arange(0, self.num_nodes + 1)

        for graph_id, graph in graph_dict.items():
            edge_index = []
            edge_type = []
            for head, relation, tail in graph:
                edge_index.append([node_dict[head], node_dict[tail]])
                edge_type.append(rel_dict[relation])

            data_list.append(Data(x=x,
                                  edge_index=torch.tensor(edge_index).T,
                                  edge_type=torch.tensor(edge_type),
                                  id=int(graph_id)))

        torch.save(self.collate(data_list), self.processed_paths[0])

        torch.save(node_dict, self.processed_paths[1])
        torch.save(rel_dict, self.processed_paths[2])


def add_edges(dataset, reverse_edges=False, self_loops=False):
    num_relations = dataset.num_relations

    if not reverse_edges and not self_loops:
        return [d for d in dataset], num_relations

    if reverse_edges:
        dataset_new = []
        for g in dataset:
            row, col = g.edge_index
            row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)

            g_new = Data(x=g.x,
                         edge_index=torch.stack([row, col], dim=0),
                         edge_type=torch.cat([g.edge_type, g.edge_type + num_relations]),
                         id=int(g.id))
            dataset_new.append(g_new)
        num_relations *= 2
        dataset = dataset_new

    if self_loops:
        dataset_new = []
        for g in dataset:
            g_nodes = torch.unique(g.edge_index.flatten())

            g_new = Data(x=g.x,
                         edge_index=torch.cat((g.edge_index, torch.stack([g_nodes, g_nodes], dim=0)),
                                              dim=1),
                         edge_type=torch.cat([g.edge_type, torch.full_like(g_nodes, num_relations)]),
                         id=g.id)
            dataset_new.append(g_new)
        num_relations += 1

    return dataset_new, num_relations


def add_labels(dataset, label_file):
    dataset_new = []
    targets = pd.read_csv(label_file, header=None, names=['id', 'label'])
    targets.astype('int')

    for g in dataset:

        if g.id in targets['id'].tolist():
            g.y = torch.tensor(targets.loc[targets['id'] == g.id]['label'].item())
            dataset_new.append(g)

    return dataset_new


def apply_downsampling(dataset):
    dataset_new = []

    labels = [g.y.item() for g in dataset]
    samples_true = samples_false = min([labels.count(i) for i in range(2)])

    while samples_true > 0 or samples_false > 0:
        for g in dataset:
            if g.y.item() == 1 and samples_true > 0:
                dataset_new.append(g)
                samples_true -= 1
            elif g.y.item() == 0 and samples_false > 0:
                dataset_new.append(g)
                samples_false -= 1

    return dataset_new


class GCN(torch.nn.Module):
    def __init__(self, layer_type, hidden_channels, num_nodes, num_relations, freeze_embeddings=True):
        super(GCN, self).__init__()
        torch.manual_seed(12345)

        self.layer_type = layer_type
        self.num_relations = num_relations
        self.emb = torch.nn.Embedding(num_nodes, hidden_channels, _freeze=freeze_embeddings)
        self.lin = Linear(hidden_channels, NUM_CLASSES)
        self.hidden_channels = hidden_channels
        self.freeze_embeddings = freeze_embeddings

        if layer_type == 'GCN':
            self.conv1 = GCNConv(hidden_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)

        elif layer_type == 'RGCN':
            self.conv1 = RGCNConv(hidden_channels, hidden_channels, num_relations=self.num_relations)
            self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations=self.num_relations)
            self.conv3 = RGCNConv(hidden_channels, hidden_channels, num_relations=self.num_relations)

        elif layer_type == 'GIN':
            self.conv1 = GINConv(MLP([hidden_channels, hidden_channels]))
            self.conv2 = GINConv(MLP([hidden_channels, hidden_channels]))
            self.conv3 = GINConv(MLP([hidden_channels, hidden_channels]))

        elif layer_type == 'GINE':
            self.conv1 = GINEConv(MLP([hidden_channels, hidden_channels]), edge_dim=hidden_channels)
            self.conv2 = GINEConv(MLP([hidden_channels, hidden_channels]), edge_dim=hidden_channels)
            self.conv3 = GINEConv(MLP([hidden_channels, hidden_channels]), edge_dim=hidden_channels)
            self.emb_rel = torch.nn.Embedding(num_relations, hidden_channels)

    def forward(self, x, edge_index, edge_type, batch):
        edge_features = None
        if self.layer_type == 'RGCN':
            edge_features = edge_type
        elif self.layer_type == 'GINE':
            if self.freeze_embeddings:
                edge_features = torch.nn.functional.one_hot(edge_type, num_classes=self.hidden_channels).to(
                    torch.float)
            else:
                edge_features = self.emb_rel.weight[edge_type.long()]

        # 1. Obtain node embeddings
        x = self.conv1(self.emb.weight[x.long()], edge_index, edge_features)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_features)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_features)

        # 2. Readout layer
        # filter x and batch (by deleting certain indices) to only contain nodes
        # that occude in edge_index
        relevant_nodes = torch.unique(edge_index.flatten())
        x = x[relevant_nodes]
        batch = batch[relevant_nodes]

        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x



def evaluate(pattern_name='p-v1-t1-v1', graph_size=2, gp_enriched=False):
    logging.debug(f'Evaluating GP: {pattern_name}, graph size: {graph_size}, enriched: {gp_enriched}')

    graph_file = f'./data/raw/enriched-graphs_of_size_{graph_size}.json' if gp_enriched else f'./data/raw/plain-graphs_of_size_{graph_size}.json'
    dataset_name = graph_file.split('/')[-1].replace('.json', '')
    label_file = f'./data/raw/target-{pattern_name}-graph_size_{graph_size}.csv'

    dataset = GPClassification(f'./data/{dataset_name}/', dataset_name=dataset_name,
                               graph_file=graph_file)
    dataset, num_relations = add_edges(dataset, reverse_edges=True, self_loops=True)

    logging.debug(f'Number of relations: {num_relations}')

    x = dataset[0].x
    torch.manual_seed(12345)
    random.shuffle(dataset)

    dataset = add_labels(dataset, label_file)

    random.shuffle(dataset)

    dataset = apply_downsampling(dataset)

    random.shuffle(dataset)

    dataset_new = []
    for g in dataset:
        dataset_new.append(g.to(DEVICE))
    dataset = dataset_new

    train, val, test = torch.utils.data.random_split(dataset, [0.7, 0.15, 0.15])

    train_loader = DataLoader(train, batch_size=BS, shuffle=True)  # , sampler=sampler)
    val_loader = DataLoader(val, batch_size=BS, shuffle=False)
    test_loader = DataLoader(test, batch_size=BS, shuffle=False)

    model = GCN(layer_type=MODEL_TYPE, hidden_channels=HIDDEN_CHANNELS, num_nodes=x.size(0),
                num_relations=num_relations, freeze_embeddings=FREEZE_EMBEDDINGS).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0
    epochs_no_improve = 0
    results = {}

    for epoch in range(EPOCHS):
        train_model(model, criterion, optimizer, train_loader)

        valid_results = test_model(model, val_loader)

        logging.debug(f'Epoch: {epoch}, Val Acc: {valid_results["Acc"]}')

        if valid_results['Acc'] > best_val_acc:
            best_val_acc = valid_results['Acc']
            epochs_no_improve = 0

            train_results = test_model(model, train_loader)
            test_results = test_model(model, test_loader)

            results = {m: {'train': train_results[m],
                           'val': valid_results[m],
                           'test': test_results[m]} for m in ['F1', 'P', 'R', 'Acc']}
        else:
            epochs_no_improve += 1

        if epochs_no_improve == 20:
            break

    logging.debug(f'Results: {str(results)}')

    return results


def train_model(model, criterion, optimizer, train_loader):
    model.train()

    loss_epoch = []
    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x.long(), data.edge_index, data.edge_type,
                    data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss_epoch.append(loss.item())
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    logging.debug(f'Loss: {reduce(lambda x, y: x + y, loss_epoch) / len(loss_epoch)}')


def test_model(model, loader):
    model.eval()
    y_true = []
    y_pred = []

    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.edge_type, data.batch)
        pred = out.argmax(dim=1)  # Use the class with the highest probability.

        y_true += data.y.tolist()
        y_pred += pred.tolist()

    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)

    return {'F1': f1, 'Acc': acc, 'P': p, 'R': r}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='GP Classification',
        description='Model to classify graphs based on graph patterns.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Hyperparameters
    parser.add_argument('--epochs', default=200, type=int, help="Number of training epochs.")  # option that takes a value
    parser.add_argument('--model', default='GCN', choices=['GCN', 'RGCN', 'GINE', 'GIN'], help='Model type.')
    parser.add_argument('--hidden', default=32, type=int, help='Number of hidden/embedding channels.')
    parser.add_argument('--bs', default=64, type=int, help='Batch size.')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate.')
    parser.add_argument('--freeze_emb',
                        action='store_true',
                        help='Freeze the embeddings of the nodes in the model.')
    parser.add_argument('--ming', default=75,
                        help='Minimum number of positive / negative samples to evaluate a setting.',
                        type=int)

    # Reporting
    parser.add_argument('--run_name', default='run=01', help='Name of the run to store the results.')
    parser.add_argument('--debug', action='store_true', help='Set to debug mode (more verbose).')

    # Example CMD call to evaluate a specific setting:
    # python main.py --pattern p-v1-t1-v1 --graph_size 2 --gp_enriched
    parser.add_argument('--pattern', help='Pattern name to evaluate., e.g. p-v1-t1-v1.')
    parser.add_argument('--graph_size', help='Graph size to evaluate, e.g. 2.', type=int)
    parser.add_argument('--gp_enriched', action='store_true',
                        help='Set to evaluate enriched graphs, otherwise plain graphs are evaluated.')
    args = parser.parse_args()

    # Constants
    NUM_CLASSES = 2
    EPOCHS = args.epochs
    LR = args.lr
    MODEL_TYPE = args.model
    HIDDEN_CHANNELS = args.hidden
    BS = args.bs
    FREEZE_EMBEDDINGS = args.freeze_emb
    RUN_NAME = args.run_name
    MIN_G = args.ming
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set-up Logging
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG if args.debug else logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')

    logging.info(f'Running experiments on: {DEVICE}')
    logging.info(f'Hyperparameters Epochs: {EPOCHS}, Model: {MODEL_TYPE}, Hidden: {HIDDEN_CHANNELS}, LR: {LR}, Freeze: {FREEZE_EMBEDDINGS}')

    if args.pattern:
        logging.info('Running specific experiment according to the CMD arguments.')
        time_start = time.time()
        evaluate(pattern_name=args.pattern, graph_size=args.graph_size, gp_enriched=args.gp_enriched)
        logging.info(f'Training time: {time.time() - time_start}')
    else:
        logging.info('Running series of experiments.')
        logging.info(f'Settings Name: {RUN_NAME}, Model: {MODEL_TYPE} Freeze: {FREEZE_EMBEDDINGS}')

        results_file = f'./results/{RUN_NAME}_model={MODEL_TYPE}_freeze={FREEZE_EMBEDDINGS}_dim={HIDDEN_CHANNELS}_lr={LR}_ming={MIN_G}.json'
        if not os.path.exists(results_file):
            shutil.copyfile('data/raw/evaluation_results.json', results_file)

        logging.info(f'Results will be saved to: {results_file}')

        results_dict = json.load(open(results_file, 'r'))

        # calculate the overall numer of experiments
        num_experiments = 0
        for pattern_size, pattern in results_dict.items():
            for pattern_name, pattern_results in pattern.items():
                for graph_size, metrics in pattern_results.items():
                    if metrics['cnt_pos'] >= MIN_G:
                        num_experiments += 1

        pbar = tqdm(total=num_experiments)
        for pattern_size in results_dict.keys():
            for pattern_name in results_dict[pattern_size].keys():
                for graph_size in results_dict[pattern_size][pattern_name].keys():
                    if results_dict[pattern_size][pattern_name][graph_size]['cnt_pos'] <= MIN_G or \
                            results_dict[pattern_size][pattern_name][graph_size]['cnt_neg'] <= MIN_G:
                        continue
                    pbar.update(1)
                    for enriched in ['enriched', 'plain']:

                        pattern_size_int = int(pattern_size.replace('pattern_of_size_', ''))
                        gp_enriched = enriched == 'enriched'

                        graph_size_int = int(graph_size.replace('graph_size_', ''))

                        if results_dict[pattern_size][pattern_name][graph_size][enriched]['F1'][
                            'test'] == -1:
                            results = evaluate(pattern_name=pattern_name, graph_size=graph_size_int,
                                               gp_enriched=gp_enriched)
                            results_dict[pattern_size][pattern_name][graph_size][enriched] = results

                            with open(results_file, 'w') as f:
                                json.dump(results_dict, f)
        pbar.close()
