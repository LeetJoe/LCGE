# Copyright (c) Facebook, Inc. and its affiliates.

import os
import argparse
from typing import Dict
import logging
from numpy.core.fromnumeric import _size_dispatcher
import torch
from torch import optim

from datasets_lcge import TemporalDataset
from optimizers_lcge import TKBCOptimizer, IKBCOptimizer
from models_lcge import LCGE
from regularizers_rule import N3, Lambda3
import datetime

import sys

from regularizers_rule import RuleSim
import json

# todo 暂保留，最后再删除
import datetime
log_file = open('./run.log', 'a')
log_file.write('\n\n\nEvaluation start: {}\n'.format(datetime.datetime.now()))

parser = argparse.ArgumentParser(
    description="Logic and Commonsense-Guided Temporal KGE"
)
parser.add_argument(
    '--dataset', type=str, default='',
    help="Dataset name"
)
parser.add_argument(
    '--data_path', default='./src_data/', type=str,
    help="Data path for load & save (experiment path)"
)
parser.add_argument(
    '--exp_folder', default='lcge_main', type=str,
    help="Experiment folder for saving data"
)
parser.add_argument(
    '--iter', default=0, type=int,
    help="Iteration number of EM"
)

args = parser.parse_args()

exp_path = os.path.join(args.data_path, args.exp_folder)
kge_path = os.path.join(str(exp_path), str(args.iter), 'kge')
rule_path = os.path.join(str(exp_path), 'rulelearning')

dataset = TemporalDataset(args.dataset, data_path=args.data_path)

print("Calculating scores of infer...\n")
sigmoid_factor = 10
batch_size = 500
model = torch.load(os.path.join(kge_path, 'checkpoint_best.pt'))
model.eval()
file_buff = ''

"""
# save infer scores
infer_size = len(dataset.data['infer'])
data_infer = dataset.data['infer']
data_infer_reverse = dataset.get_reversed('infer')

with open(os.path.join(kge_path, 'annotation.txt'), 'w') as fw:
    for i in range(0, infer_size, batch_size):
        i_end = min(i + batch_size, infer_size)
        infer = torch.from_numpy(data_infer[i:i_end].astype('int64')).cuda()
        infer_reverse = torch.from_numpy(data_infer_reverse[i:i_end].astype('int64')).cuda()
        targets_tem, targets_cs = model.score(infer)
        targets_avg = torch.sigmoid((targets_tem + targets_cs) / (2 * sigmoid_factor))
        targets_tem_reverse, targets_cs_reverse = model.score(infer_reverse)
        targets_reverse_avg = torch.sigmoid((targets_tem_reverse + targets_cs_reverse) / (2 * sigmoid_factor))

        for j in range(i, i_end):
            # fw.write('{}\t{}\t{}\t{}\tsp\t{:.8f}\n'.format(*data_infer[j], targets_avg[j-i].item()))
            # fw.write('{}\t{}\t{}\t{}\tpo\t{:.8f}\n'.format(*data_infer[j], targets_reverse_avg[j-i].item()))
            file_buff += '{}\t{}\t{}\t{}\tsp\t{:.8f}\n'.format(*data_infer[j], targets_avg[j-i].item())
            file_buff += '{}\t{}\t{}\t{}\tpo\t{:.8f}\n'.format(*data_infer[j], targets_reverse_avg[j-i].item())

        if file_buff != '':
            fw.write(file_buff)
            file_buff = ''
    fw.close()
"""

# save test scores
test_size = len(dataset.data['test'])
data_test = dataset.data['test']
data_test_reverse = dataset.get_reversed('test')

with open(os.path.join(kge_path, 'pred_kge.txt'), 'w') as fw:
    for i in range(0, test_size, batch_size):
        i_end = min(i + batch_size, test_size)
        test = torch.from_numpy(data_test[i:i_end].astype('int64')).cuda()
        test_reverse = torch.from_numpy(data_test_reverse[i:i_end].astype('int64')).cuda()
        queries = model.get_queries(test)
        queries_reverse = model.get_queries(test_reverse)
        rhs = model.get_rhs(0, dataset.n_entities)
        rhs_static = model.get_rhs_static(0, dataset.n_entities)
        scores_tem = torch.sigmoid((queries[0] @ rhs) / sigmoid_factor)
        scores_cs = torch.sigmoid((queries[1] @ rhs_static) / sigmoid_factor)
        scores_tem_reverse = torch.sigmoid((queries_reverse[0] @ rhs) / sigmoid_factor)
        scores_cs_reverse = torch.sigmoid((queries_reverse[1] @ rhs_static) / sigmoid_factor)
        ranks = model.get_ranking(test, dataset.to_skip['rhs'], batch_size=500)
        ranks_reverse = model.get_ranking(test_reverse, dataset.to_skip['lhs'], batch_size=500)

        for j in range(i, i_end):
            score_dict_tem = {k: scores_tem[j - i][k].item() for k in range(len(scores_tem[j - i]))}
            scorelist_tem = [str(a) + '*{:.6f}'.format(b) for (a, b) in
                        sorted(score_dict_tem.items(), key=lambda x: x[1], reverse=True)]
            score_dict_cs = {k: scores_cs[j - i][k].item() for k in range(len(scores_cs[j - i]))}
            scorelist_cs = [str(a) + '*{:.6f}'.format(b) for (a, b) in
                        sorted(score_dict_cs.items(), key=lambda x: x[1], reverse=True)]
            # fw.write('{}\t{}\t{}\t{}\tsp\t{}\n{}\n{}\n'.format(*data_test[i], int(ranks[j - i].item()), ';'.join(scorelist_tem), ';'.join(scorelist_cs)))
            file_buff += '{}\t{}\t{}\t{}\tsp\t{}\n{}\n{}\n'.format(*data_test[i], int(ranks[j - i].item()), ';'.join(scorelist_tem), ';'.join(scorelist_cs))

            score_dict_tem = {k: scores_tem_reverse[j - i][k].item() for k in range(len(scores_tem_reverse[j - i]))}
            scorelist_tem = [str(a) + '*{:.6f}'.format(b) for (a, b) in
                        sorted(score_dict_tem.items(), key=lambda x: x[1], reverse=True)]
            score_dict_cs = {k: scores_cs_reverse[j - i][k].item() for k in range(len(scores_cs_reverse[j - i]))}
            scorelist_cs = [str(a) + '*{:.6f}'.format(b) for (a, b) in
                        sorted(score_dict_cs.items(), key=lambda x: x[1], reverse=True)]

            # fw.write('{}\t{}\t{}\t{}\tpo\t{}\n{}\n'.format(*data_test_reverse[i], int(ranks_reverse[j - i].item()), ';'.join(scorelist_tem), ';'.join(scorelist_cs)))
            file_buff += '{}\t{}\t{}\t{}\tpo\t{}\n{}\n'.format(*data_test_reverse[i], int(ranks_reverse[j - i].item()), ';'.join(scorelist_tem), ';'.join(scorelist_cs))
        if file_buff != '':
            fw.write(file_buff)
            file_buff = ''
        break
    fw.close()

# todo ranks.txt, result_em.txt

print("LCGE evaluation done.\n")


# todo 暂保留，最后再删除
log_file.write('Evaluation end: {}\n'.format(datetime.datetime.now()))
log_file.close()
