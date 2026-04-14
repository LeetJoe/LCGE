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


# load mln predictions of test
def load_pred_mln(pred_file, relation_size):
    test_pred = {'rhs': {}, 'lhs': {}}
    score_max = 0.0
    score_min = 10.0
    with open(pred_file, 'r') as fr:
        while True:
            line_query = fr.readline().strip()
            line_cands = fr.readline().strip()

            # prediction file contains all test data so the candidates line may be blank
            if (not line_query) and (not line_cands):
                break

            query_data = line_query.split('\t')
            cands = json.loads(line_cands)
            cands = {int(k) : float(v) for k, v in cands.items()}
            cands = dict(sorted(cands.items(), key=lambda item: item[1], reverse=True))
            if len(cands) > 0:
                key_list = list(cands.keys())
                if cands[key_list[0]] > score_max:
                    score_max = cands[key_list[0]]
                if cands[key_list[-1]] < score_min:
                    score_min = cands[key_list[-1]]

            if query_data[4] == 'sp':
                missing = 'rhs'
                query_key = (int(query_data[0]), int(query_data[1]), int(query_data[2]), int(query_data[3]))
            else:
                missing = 'lhs'
                query_key = (int(query_data[2]), int(query_data[1]) + relation_size, int(query_data[0]), int(query_data[3]))


            test_pred[missing][query_key] = cands
        fr.close()

    for missing in ['rhs', 'lhs']:
        for query_key in test_pred[missing]:
            if len(test_pred[missing][query_key]) > 0:
                for k in test_pred[missing][query_key]:
                    test_pred[missing][query_key][k] = (test_pred[missing][query_key][k] - score_min) / (score_max - score_min)

    return test_pred


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
mln_path = os.path.join(str(exp_path), str(args.iter), 'mln')
rule_path = os.path.join(str(exp_path), 'rulelearning')

log_file = open(os.path.join(kge_path, 'run.log'), 'a')
log_file.write('[{}]Evaluation start: {}\n'.format(args.iter, datetime.datetime.now()))

dataset = TemporalDataset(args.dataset, data_path=args.data_path)

print("Calculating scores of infer...\n")
sigmoid_factor = 10
batch_size = 500
model = torch.load(os.path.join(kge_path, 'checkpoint_best.pt'))
model.eval()
file_buff = ''

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

# save test scores
test_size = len(dataset.data['test'])
data_test = dataset.data['test']
data_test_reverse = dataset.get_reversed('test')
mln_pred_file = os.path.join(mln_path, 'pred_test.txt')
mln_scores = load_pred_mln(mln_pred_file, dataset.get_shape()[1])
mr = {'rhs': 0.0, 'lhs': 0.0}
mr_em = {'rhs': 0.0, 'lhs': 0.0}
mrr = {'rhs': 0.0, 'lhs': 0.0}
mrr_em = {'rhs': 0.0, 'lhs': 0.0}
hits = {'rhs': {1: 0.0, 3: 0.0, 10: 0.0}, 'lhs': {1: 0.0, 3: 0.0, 10: 0.0}}
hits_em = {'rhs': {1: 0.0, 3: 0.0, 10: 0.0}, 'lhs': {1: 0.0, 3: 0.0, 10: 0.0}}

with open(os.path.join(str(exp_path), str(args.iter), 'ranks.txt'), 'w') as fw:
    fw.write('query\tdirection\tkge_rank\tem_rank\n')
    for i in range(0, test_size, batch_size):
        i_end = min(i + batch_size, test_size)
        test = torch.from_numpy(data_test[i:i_end].astype('int64')).cuda()
        test_reverse = torch.from_numpy(data_test_reverse[i:i_end].astype('int64')).cuda()
        ranks = model.get_ranking(test, dataset.to_skip['rhs'], batch_size=500)
        ranks_em = model.get_ranking_em(test, dataset.to_skip['rhs'], mln_scores['rhs'], batch_size=500)
        ranks_reverse = model.get_ranking(test_reverse, dataset.to_skip['lhs'], batch_size=500)
        ranks_reverse_em = model.get_ranking_em(test_reverse, dataset.to_skip['lhs'], mln_scores['lhs'], batch_size=500)

        for j in range(i, i_end):
            file_buff += '{}\tsp\t{}\t{}\n'.format(','.split(data_test[j]), int(ranks[j - i]), int(ranks_em[j - i]))
            file_buff += '{}\tpo\t{}\t{}\n'.format(','.split(data_test[j]), int(ranks_reverse[j - i]), int(ranks_reverse_em[j - i]))

            mr['rhs'] += int(ranks[j - i])
            mr['lhs'] += int(ranks_reverse[j - i])
            mr_em['rhs'] += int(ranks_em[j - i])
            mr_em['lhs'] += int(ranks_reverse_em[j - i])

            mrr['rhs'] += 1 / int(ranks[j - i])
            mrr['lhs'] += 1 / int(ranks_reverse[j - i])
            mrr_em['rhs'] += 1 / int(ranks_em[j - i])
            mrr_em['lhs'] += 1 / int(ranks_reverse_em[j - i])

            if int(ranks[j - i]) <= 1:
                hits['rhs'][1] += 1
                hits['rhs'][3] += 1
                hits['rhs'][10] += 1
            elif int(ranks[j - i]) <= 3:
                hits['rhs'][3] += 1
                hits['rhs'][10] += 1
            elif int(ranks[j - i]) <= 10:
                hits['rhs'][10] += 1

            if int(ranks_em[j - i]) <= 1:
                hits_em['rhs'][1] += 1
                hits_em['rhs'][3] += 1
                hits_em['rhs'][10] += 1
            elif int(ranks_em[j - i]) <= 3:
                hits_em['rhs'][3] += 1
                hits_em['rhs'][10] += 1
            elif int(ranks_em[j - i]) <= 10:
                hits_em['rhs'][10] += 1

            if int(ranks_reverse[j - i]) <= 1:
                hits['lhs'][1] += 1
                hits['lhs'][3] += 1
                hits['lhs'][10] += 1
            elif int(ranks_reverse[j - i]) <= 3:
                hits['lhs'][3] += 1
                hits['lhs'][10] += 1
            elif int(ranks_reverse[j - i]) <= 10:
                hits['lhs'][10] += 1

            if int(ranks_reverse_em[j - i]) <= 1:
                hits_em['lhs'][1] += 1
                hits_em['lhs'][3] += 1
                hits_em['lhs'][10] += 1
            elif int(ranks_reverse_em[j - i]) <= 3:
                hits_em['lhs'][3] += 1
                hits_em['lhs'][10] += 1
            elif int(ranks_reverse_em[j - i]) <= 10:
                hits_em['lhs'][10] += 1

        if file_buff != '':
            fw.write(file_buff)
            file_buff = ''
    fw.close()

for missing in ['rhs', 'lhs']:
    mr[missing] /= test_size
    mr_em[missing] /= test_size
    mrr[missing] /= test_size
    mrr_em[missing] /= test_size
    for banner in [1, 3, 10]:
        hits[missing][banner] /= test_size
        hits_em[missing][banner] /= test_size

with open(os.path.join(str(exp_path), str(args.iter), 'result_em.txt'), 'w') as fw:
    fw.write('kge:\n')
    fw.write('MR: {}, MRR: {}, Hit@1: {}, Hit@3: {}, Hit@10: {}\n'.format(
        (mr['rhs'] + mr['lhs'])/2, (mrr['rhs'] + mrr['lhs'])/2, (hits['rhs'][1] + hits['lhs'][1])/2,
        (hits['rhs'][3] + hits['lhs'][3])/2, (hits['rhs'][10] + hits['lhs'][10])/2))
    fw.write('\n')
    fw.write('kge+mln:\n')
    fw.write('MR: {}, MRR: {}, Hit@1: {}, Hit@3: {}, Hit@10: {}\n'.format(
        (mr_em['rhs'] + mr_em['lhs'])/2, (mrr_em['rhs'] + mrr_em['lhs'])/2, (hits_em['rhs'][1] + hits_em['lhs'][1])/2,
        (hits_em['rhs'][3] + hits_em['lhs'][3])/2, (hits_em['rhs'][10] + hits_em['lhs'][10])/2))
    fw.write('\n')

print("LCGE evaluation done.\n")

log_file.write('[{}]Evaluation end: {}\n'.format(args.iter, datetime.datetime.now()))
log_file.close()
