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


parser = argparse.ArgumentParser(
    description="Logic and Commonsense-Guided Temporal KGE"
)
parser.add_argument(
    '--dataset', type=str,
    help="Dataset name"
)
models = [
    'LCGE'
]
parser.add_argument(
    '--model', choices=models,
    help="Model in {}".format(models)
)
parser.add_argument(
    '--max_epochs', default=50, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid_freq', default=5, type=int,
    help="Number of epochs between each valid."
)
parser.add_argument(
    '--rank', default=100, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Batch size."
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--emb_reg', default=0., type=float,
    help="Embedding regularizer strength"
)
parser.add_argument(
    '--time_reg', default=0., type=float,
    help="Timestamp regularizer strength"
)
parser.add_argument(
    '--no_time_emb', default=False, action="store_true",
    help="Use a specific embedding for non temporal relations"
)
parser.add_argument(
    '--rule_reg', default=0., type=float,
    help="Rule regularizer strength"
)
parser.add_argument(
    '--weight_static', default=0., type=float,
    help="Weight of static score"
)
parser.add_argument(
    '--data_path', default='./src_data/', type=str,
    help="Data path for load & save (experiment path)"
)
parser.add_argument(
    '--rule_path', default='./src_data/rulelearning/', type=str,
    help="Use provided rules or not"
)


args = parser.parse_args()

rule_path = args.rule_path

dataset = TemporalDataset(args.dataset, data_path=args.data_path)

with open(rule_path + "/rule1_p1.json", 'r') as load_rule1_p1:
    rule1_p1 = json.load(load_rule1_p1)
with open(rule_path + "/rule1_p2.json", 'r') as load_rule1_p2:
    rule1_p2 = json.load(load_rule1_p2)

f = open(rule_path + "/rule2_p1.txt", 'r')
rule2_p1 = {}
for line in f:
    head, body1, body2, confi = line.strip().split("\t")
    head, body1, body2, confi = int(head), int(body1), int(body2), float(confi)
    if head not in rule2_p1:
        rule2_p1[head] = {}
    rule2_p1[head][(body1, body2)] = confi
f.close()

f = open(rule_path + "/rule2_p2.txt", 'r')
rule2_p2 = {}
for line in f:
    head, body1, body2, confi = line.strip().split("\t")
    head, body1, body2, confi = int(head), int(body1), int(body2), float(confi)
    if head not in rule2_p2:
        rule2_p2[head] = {}
    rule2_p2[head][(body1, body2)] = confi
f.close()

f = open(rule_path + "/rule2_p3.txt", 'r')
rule2_p3 = {}
for line in f:
    head, body1, body2, confi = line.strip().split("\t")
    head, body1, body2, confi = int(head), int(body1), int(body2), float(confi)
    if head not in rule2_p3:
        rule2_p3[head] = {}
    rule2_p3[head][(body1, body2)] = confi
f.close()

f = open(rule_path + "/rule2_p4.txt", 'r')
rule2_p4 = {}
for line in f:
    head, body1, body2, confi = line.strip().split("\t")
    head, body1, body2, confi = int(head), int(body1), int(body2), float(confi)
    if head not in rule2_p4:
        rule2_p4[head] = {}
    rule2_p4[head][(body1, body2)] = confi
f.close()

rules = (rule1_p1, rule1_p2, rule2_p1, rule2_p2, rule2_p3, rule2_p4)

sizes = dataset.get_shape()
print("sizes of dataset is:\t", sizes)
model = {
    'LCGE': LCGE(sizes, args.rank, rules, args.weight_static, no_time_emb=args.no_time_emb),
}[args.model]
model = model.cuda()


opt = optim.Adagrad(model.parameters(), lr=args.learning_rate)

emb_reg = N3(args.emb_reg)
time_reg = Lambda3(args.time_reg)
rule_reg = RuleSim(args.rule_reg)       # relation embedding reglu via rules

best_mrr = 0.
best_hit = 0.
early_stopping = 0

for epoch in range(args.max_epochs):
    examples = torch.from_numpy(
        dataset.get_train().astype('int64')
    )
    #print("\nexamples:\n", examples.size())

    model.train()
    if dataset.has_intervals():
        optimizer = IKBCOptimizer(
            model, emb_reg, time_reg, opt, dataset,
            batch_size=args.batch_size
        )
        optimizer.epoch(examples)

    else:
        optimizer = TKBCOptimizer(
            model, emb_reg, time_reg, rule_reg, opt,
            batch_size=args.batch_size
        )
        optimizer.epoch(examples)


    def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
        """
        aggregate metrics for missing lhs and rhs
        :param mrrs: d
        :param hits:
        :return:
        """
        m = (mrrs['lhs'] + mrrs['rhs']) / 2.
        h = (hits['lhs'] + hits['rhs']) / 2.
        return {'MRR': m, 'hits@[1,3,10]': h}

    if epoch < 0 or (epoch + 1) % args.valid_freq == 0:
        if dataset.has_intervals():
            valid, test, train = [
                dataset.eval(model, split, -1 if split != 'train' else 50000)
                for split in ['valid', 'test', 'train']
            ]
            print("valid: ", valid)
            print("test: ", test)
            print("train: ", train)

        else:
            valid, test, train = [
                avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
                for split in ['valid', 'test', 'train']
            ]
            print("epoch: ", epoch+1)
            print("valid: ", valid['MRR'])
            print("test: ", test['MRR'])
            print("train: ", train['MRR'])

            print("test hits@n:\t", test['hits@[1,3,10]'])
            if test['MRR'] > best_mrr:
                torch.save(model, './whole_model.pth')
                best_mrr = test['MRR']
                best_hit = test['hits@[1,3,10]']
                early_stopping = 0
            else:
                early_stopping += 1
            if early_stopping > 10:
                print("early stopping!")
                break

if args.max_epochs > 0:
    print("The best test mrr is:\t", best_mrr)
    print("The best test hits@1,3,10 are:\t", best_hit)

print("Saving scores of test & infer...\n")

sigmoid_factor = 10
model = torch.load('./whole_model.pth')
model.eval()

# save test scores
test_size = len(dataset.data['test'])
data_test = dataset.data['test']
data_test_reverse = dataset.get_reversed('test')
test = torch.from_numpy(data_test.astype('int64')).cuda()
test_reverse = torch.from_numpy(data_test_reverse.astype('int64')).cuda()
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

file_buff = ''
with open('./pred_kge.txt', 'w') as fw:
    for i in range(test_size):
        score_dict_tem = {j: scores_tem[i][j].item() for j in range(len(scores_tem[i]))}
        scorelist_tem = [str(a) + '*{:.6f}'.format(b) for (a, b) in
                    sorted(score_dict_tem.items(), key=lambda x: x[1], reverse=True)]
        score_dict_cs = {j: scores_cs[i][j].item() for j in range(len(scores_cs[i]))}
        scorelist_cs = [str(a) + '*{:.6f}'.format(b) for (a, b) in
                    sorted(score_dict_cs.items(), key=lambda x: x[1], reverse=True)]
        # fw.write('{}\t{}\t{}\t{}\tsp\t{}\n{}\n{}\n'.format(*data_test[i], int(ranks[i].item()), ';'.join(scorelist_tem), ';'.join(scorelist_cs)))
        file_buff += '{}\t{}\t{}\t{}\tsp\t{}\n{}\n{}\n'.format(*data_test[i], int(ranks[i].item()), ';'.join(scorelist_tem), ';'.join(scorelist_cs))

        score_dict_tem = {j: scores_tem_reverse[i][j].item() for j in range(len(scores_tem_reverse[i]))}
        scorelist_tem = [str(a) + '*{:.6f}'.format(b) for (a, b) in
                    sorted(score_dict_tem.items(), key=lambda x: x[1], reverse=True)]
        score_dict_cs = {j: scores_cs_reverse[i][j].item() for j in range(len(scores_cs_reverse[i]))}
        scorelist_cs = [str(a) + '*{:.6f}'.format(b) for (a, b) in
                    sorted(score_dict_cs.items(), key=lambda x: x[1], reverse=True)]

        # fw.write('{}\t{}\t{}\t{}\tpo\t{}\n{}\n'.format(*data_test_reverse[i], int(ranks_reverse[i].item()), ';'.join(scorelist_tem), ';'.join(scorelist_cs)))
        file_buff += '{}\t{}\t{}\t{}\tpo\t{}\n{}\n'.format(*data_test_reverse[i], int(ranks_reverse[i].item()), ';'.join(scorelist_tem), ';'.join(scorelist_cs))
        if (i + 1) % 100 == 0:
            fw.write(file_buff)
            file_buff = ''
    if file_buff != '':
        fw.write(file_buff)
        file_buff = ''
    fw.close()

# save infer scores
infer_size = len(dataset.data['infer'])
data_infer = dataset.data['infer']
data_infer_reverse = dataset.get_reversed('infer')
infer = torch.from_numpy(data_infer.astype('int64')).cuda()
infer_reverse = torch.from_numpy(data_infer_reverse.astype('int64')).cuda()
targets_tem, targets_cs = model.score(infer)
targets_tem = torch.sigmoid(targets_tem / sigmoid_factor)
targets_cs = torch.sigmoid(targets_cs / sigmoid_factor)
targets_tem_reverse, targets_cs_reverse = model.score(infer_reverse)
targets_tem_reverse = torch.sigmoid(targets_tem_reverse / sigmoid_factor)
targets_cs_reverse = torch.sigmoid(targets_cs_reverse / sigmoid_factor)

with open('./annotation.txt', 'w') as fw:
    for i in range(infer_size):
        # fw.write('{}\t{}\t{}\t{}\tsp\t{:.f8}\t{:.f8}\n'.format(*data_infer[i], targets_tem[i].item(), targets_cs[i].item()))
        # fw.write('{}\t{}\t{}\t{}\tpo\t{:.f8}\t{:.f8}\n'.format(*data_infer[i], targets_tem_reverse[i].item(), targets_cs_reverse[i].item()))

        file_buff += '{}\t{}\t{}\t{}\tsp\t{:.8f}\t{:.8f}\n'.format(*data_infer[i], targets_tem[i].item(), targets_cs[i].item())
        file_buff += '{}\t{}\t{}\t{}\tpo\t{:.8f}\t{:.8f}\n'.format(*data_infer[i], targets_tem_reverse[i].item(), targets_cs_reverse[i].item())

        if (i + 1) % 500 == 0:
            fw.write(file_buff)
            file_buff = ''
    if file_buff != '':
        fw.write(file_buff)
        file_buff = ''
    fw.close()

print("LCGE done.\n")
