import logging
from logging import getLogger
from recbole.utils import init_logger, init_seed
from recbole.trainer import Trainer
from torch_geometric.utils import from_networkx

from Model import GraphLLM4Rec
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
import networkx as nx
import torch

logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':

    config = Config(model=GraphLLM4Rec, dataset='quantinar', config_file_list=['gnnmodel.yaml'])
    init_seed(config['seed'], config['reproducibility'])

    # logger initialization
    init_logger(config)
    logger = getLogger()

    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    gml_file = '../assets/pmfg_graph_new.graphml'
    G = nx.read_graphml(gml_file)
    #print("Graph loaded.")
    #print("Number of nodes:", G.number_of_nodes())
    #print("Number of edges:", G.number_of_edges())

    # 将 NetworkX 图转换为 PyTorch Geometric 的数据结构
    data = from_networkx(G)

    logger.info("DATAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    logger.info(data)
    if 'x' not in data:
        data.x = torch.rand(G.number_of_nodes(), config['embedding_size'])

    logger.info(data.edge_index)

    # model loading and initialization
    model = GraphLLM4Rec(config, train_data.dataset, data.to(config['device'])).to(config['device'])
    # model = LSTM4Rec(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = Trainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)

    # model evaluation
    test_result = trainer.evaluate(test_data)

    logger.info('best valid result: {}'.format(best_valid_result))
    logger.info('test result: {}'.format(test_result))
