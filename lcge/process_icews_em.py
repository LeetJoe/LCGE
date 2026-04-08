# Copyright (c) Facebook, Inc. and its affiliates.

import argparse
import os
import time as pkg_time
import subprocess
import errno
from pathlib import Path
import pickle
import json

import numpy as np

from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(
        description="Logic and Commonsense-Guided Temporal KGE"
    )
    parser.add_argument(
        '--data_path', default='./src_data', type=str,
        help="Experiment path of data"
    )
    parser.add_argument(
        '--iter', default=0, type=int,
        help="EM iteration number"
    )

    return parser.parse_args()

def static_graph(data_path: str, iter: int):
    triple = []
    for f in ['train', 'valid']:
        with open(os.path.join(data_path, f), 'r') as fr:
            for line in fr:
                e1, r, e2, _ = line.split("\t")
                triple.append(e1 + "\t" + r + "\t" + e2 + "\n")
            fr.close()

    with open(os.path.join(data_path, str(iter) + '/kge/rulelearning/triples.tsv'), 'w') as fw:
        count = 0
        for tri in triple:
            fw.write(tri)
            count += 1
        print("the amount of triples is:\t", count)
        fw.close()


def reform_static_rule(data_path: str, iter: int):
    rule_path = os.path.join(data_path, str(iter) + '/kge/rulelearning')
    rules = {'1': [], '2': []}

    with open(os.path.join(rule_path, "amie_rules_static.txt"), 'r') as f:
        for line in f:
            if line[0] != '?':
                continue
            r_b, r_h = line.split("=>")
            body_list = r_b.split("  ")
            head_list = r_h.split("  ")
            he1 = head_list[0].strip()
            hr = head_list[1]
            he2 = head_list[2].split("\t")[0]
            if len(body_list) == 4:
                e1, r1, e2, _ = body_list
                rules['1'].append(e1 + "\t" + r1 + "\t" + e2 + "\t=>\t" + he1 + "\t" + hr + "\t" + he2 + "\n")
            else:
                e1, r1, e2, e3, r2, e4, _ = body_list
                rules['2'].append(
                    e1 + "\t" + r1 + "\t" + e2 + "\t" + e3 + "\t" + r2 + "\t" + e4 + "\t=>\t" + he1 + "\t" + hr + "\t" + he2 + "\n")
        f.close()

    for rule_len in ['1', '2']:
        with open(os.path.join(rule_path, "ruleslen" + rule_len + "_static.txt"), 'w') as fw:
            for rule in rules[rule_len]:
                fw.write(rule)
            fw.close()


def learn_temporal_rule(data_path: str, iter: int):
    tgraph = {}
    for f in ['train', 'valid']:
        with open(os.path.join(data_path, f), 'r') as fr:
            for line in fr:
                e1, r, e2, time = line.strip().split("\t")
                time = time.split("-")
                y, m, d = time
                time = m + d
                time = int(time)
                if time not in tgraph:
                    tgraph[time] = {}
                if r not in tgraph[time]:
                    tgraph[time][r] = []
                tgraph[time][r].append([e1, e2])
            fr.close()

    # convert time into sorted dict
    sort_t = sorted(tgraph.keys())
    tg = {}
    for t in sort_t:
        tg[t] = tgraph[t]
    tgraph = {}
    n = 0
    for time in tg:
        tgraph[n] = tg[time]
        n += 1

    # load the static candidate len-1 rules
    rule_path = os.path.join(data_path, str(iter) + '/kge/rulelearning')
    f = open(os.path.join(rule_path, "ruleslen1_static.txt"))
    rule_1 = {}
    rule_1_rev = {}
    for line in f:
        body, head = line.strip().split("=>")
        he1, hr, he2 = head.strip().split("\t")
        be1, br, be2 = body.strip().split("\t")
        if hr not in rule_1:
            rule_1[hr] = {}
        if br not in rule_1_rev:
            rule_1_rev[br] = {}
        if br not in rule_1[hr]:
            rule_1[hr][br] = 0
            rule_1_rev[br][hr] = 0
        if he1 == be1:
            rule_1[hr][br] = 1
            rule_1_rev[br][hr] = 1
    f.close()

    # learn len-1 rules in the pattern of r1(x, y, t+T) <= r2(x, y, t)
    time_start = pkg_time.time()
    sd = {}  # support degree
    sd_list = {}  # support degree of entity pair list
    bn = {}  # body degree    standard confidence = sd / bn
    window = 20
    for head in rule_1:
        # rule head rel in each length-1 rule
        for body in rule_1[head]:
            # each rule body rel in rule[head]
            if body not in bn:
                bn[body] = 0.
            if body not in sd:
                sd[body] = {}
            if body not in sd_list:
                sd_list[body] = {}
            sd[body][head] = 0.
            for time_cur in range(len(tgraph) - window + 1):
                if body not in tgraph[time_cur]:
                    continue
                sd_list[body][head] = []
                bn[body] += len(tgraph[time_cur][body])
                for ent_pair in tgraph[time_cur][body]:
                    if rule_1[head][body] == 1:
                        a, b = ent_pair
                    else:
                        b, a = ent_pair
                    for time_next in range(time_cur + 1, time_cur + window):
                        if head not in tgraph[time_next]:
                            continue
                        for ent_pair in tgraph[time_next][head]:
                            if ent_pair[0] == a and ent_pair[1] == b and ent_pair not in sd_list[body][head]:
                                sd_list[body][head].append(ent_pair)
                sd[body][head] += len(sd_list[body][head])
    time_end = pkg_time.time()
    print('running time of len-1 temporal rule:\t', time_end - time_start)

    # evaluate SC of len-1 rules in the pattern of r1(x, y, t+T) <= r2(x, y, t)
    sc_rule1_p1 = {}
    cnt = 0
    for head in rule_1:
        if head not in sc_rule1_p1:
            sc_rule1_p1[head] = {}
        for body in rule_1[head]:
            sc = sd[body][head] / bn[body]
            if sc > 0.1:
                sc_rule1_p1[head][body] = [rule_1[head][body], sc]
                cnt += 1
        if not sc_rule1_p1[head]:
            sc_rule1_p1.pop(head)
    print("The total number of rules is:\t%d\n" % cnt)
    print("each rule and its confidence:\n", sc_rule1_p1)

    # learn static len-1 rules in the pattern of r1(x, y, t) <= r2(x, y, t)
    time_start = pkg_time.time()
    sd = {}  # support degree
    sd_list = {}  # support degree of entity pair list
    bn = {}  # body degree    standard confidence = sd / bn
    for body in rule_1_rev:
        # rule body rel in each length-1 rule
        if body not in bn:
            bn[body] = 0.
            sd[body] = {}
            sd_list[body] = {}
        for head in rule_1_rev[body]:
            # each head body rel in rule[body]
            sd[body][head] = 0.
            for time_cur in range(len(tgraph)):
                if body not in tgraph[time_cur]:
                    continue
                sd_list[body][head] = []
                bn[body] += len(tgraph[time_cur][body])
                for ent_pair in tgraph[time_cur][body]:
                    if rule_1_rev[body][head] == 1:
                        a, b = ent_pair
                    else:
                        b, a = ent_pair

                    if head not in tgraph[time_cur]:
                        continue
                    for ent_pair in tgraph[time_cur][head]:
                        if ent_pair[0] == a and ent_pair[1] == b and ent_pair not in sd_list[body][head]:
                            sd_list[body][head].append(ent_pair)
                sd[body][head] += len(sd_list[body][head])
    time_end = pkg_time.time()
    print('run a head rule time:\t', time_end - time_start)

    # evaluate the SC of static len-1 rules in the pattern of r1(x, y, t) <= r2(x, y, t)
    sc_rule1_p2 = {}
    cnt = 0
    for head in rule_1:
        if head not in sc_rule1_p2:
            sc_rule1_p2[head] = {}
        for body in rule_1[head]:
            sc = sd[body][head] / bn[body]
            if sc > 0.1:
                sc_rule1_p2[head][body] = [rule_1[head][body], sc]
                cnt += 1
        if not sc_rule1_p2[head]:
            sc_rule1_p2.pop(head)
    print("The total number of rules is:\t%d\n" % cnt)
    print("each rule and its confidence:\n", sc_rule1_p2)

    # load all the static candidate len-2 rules
    f = open(os.path.join(rule_path, "ruleslen2_static.txt"))
    rule_2 = {}
    rule_2_rev = {}
    for line in f:
        body, head = line.strip().split("=>")
        he1, hr, he2 = head.strip().split("\t")
        be1, br1, be2, be3, br2, be4 = body.strip().split("\t")
        br = (br1, br2)
        if hr not in rule_2:
            rule_2[hr] = {}
        if br not in rule_2_rev:
            rule_2_rev[br] = {}
        if br not in rule_2[hr]:
            rule_2[hr][br] = 0
            rule_2_rev[br][hr] = 0
        if he1 == be1 and he1 == be3:
            rule_2[hr][br] = 1  # (a,b)<=(a,b),(a,b)
            rule_2_rev[br][hr] = 1
        elif he1 == be2 and he1 == be4:
            rule_2[hr][br] = 2  # (a,b)<=(b,a),(b,a)
            rule_2_rev[br][hr] = 2
        elif he1 == be2 and he1 == be3:
            rule_2[hr][br] = 3  # (a,b)<=(b,a),(a,b)
            rule_2_rev[br][hr] = 3
        elif he1 == be1 and he1 == be4:
            rule_2[hr][br] = 4  # (a,b)<=(a,b),(b,a)
            rule_2_rev[br][hr] = 4
        elif he1 == be1 and he2 == be4:
            rule_2[hr][br] = 5  # (a,b)<=(a,e),(e,b)
            rule_2_rev[br][hr] = 5
        elif he1 == be2 and he2 == be4:
            rule_2[hr][br] = 6  # (a,b)<=(e,a),(e,b)
            rule_2_rev[br][hr] = 6
        elif he1 == be2 and he2 == be3:
            rule_2[hr][br] = 7  # (a,b)<=(e,a),(b,e)
            rule_2_rev[br][hr] = 7
        elif he1 == be4 and he2 == be1:
            rule_2[hr][br] = 8  # (a,b)<=(b,e),(e,a)
            rule_2_rev[br][hr] = 8
        elif he1 == be3 and he2 == be2:
            rule_2[hr][br] = 9  # (a,b)<=(e,b),(a,e)
            rule_2_rev[br][hr] = 9
        elif he1 == be4 and he2 == be2:
            rule_2[hr][br] = 10  # (a,b)<=(e,b),(e,a)
            rule_2_rev[br][hr] = 10
        elif he1 == be1 and he2 == be3:
            rule_2[hr][br] = 11  # (a,b)<=(a,e),(b,e)
            rule_2_rev[br][hr] = 11
        elif he1 == be3 and he2 == be1:
            rule_2[hr][br] = 12  # (a,b)<=(b,e),(a,e)
            rule_2_rev[br][hr] = 12
        else:
            print(line)
    f.close()

    # learn len-2 rules in the pattern of r1(h1, t1, t+T2) <= r2(h2, t2, t) ^ r3(h3, t3, t+T1), which associated with the 12 cases
    time_start = pkg_time.time()
    sd = {}  # support degree
    sd_list = {}  # support degree of entity pair list
    bn = {}  # body degree    standard confidence = sd / bn
    bn_list = {}  # body number of entity pair list
    window = 30
    for body in rule_2_rev:
        # print("rule body:\t", body)
        # rule body rel in each length-1 rule
        if body not in bn:
            bn[body] = 0.
            # bn_list[body] = []
            sd[body] = {}
            sd_list[body] = {}
        for head in rule_2_rev[body]:
            # each head body rel in rule[body]
            sd[body][head] = 0.
            # sd_list[body][head] = []
            for time_cur in range(len(tgraph) - 2 * window):
                if body[0] not in tgraph[time_cur]:
                    continue
                bn_list[body] = []
                sd_list[body][head] = []
                for ent_pair in tgraph[time_cur][body[0]]:
                    r1h, r1t = ent_pair
                    for time_next1 in range(time_cur + 1, time_cur + window):
                        if body[1] not in tgraph[time_next1]:
                            continue
                        # sd_list[body][head] = []
                        # if
                        # bn[body] += len(tgraph[time_cur][body])
                        for ent_pair in tgraph[time_next1][body[1]]:
                            r2h, r2t = ent_pair
                            if rule_2_rev[body][head] == 1 or rule_2_rev[body][head] == 2:
                                if r2h != r1h or r2t != r1t:
                                    continue
                            elif rule_2_rev[body][head] == 4 or rule_2_rev[body][head] == 3:
                                if r2h != r1t or r2t != r1h:
                                    continue
                            elif rule_2_rev[body][head] == 5 or rule_2_rev[body][head] == 8:
                                if r2h != r1t:
                                    continue
                            elif rule_2_rev[body][head] == 6 or rule_2_rev[body][head] == 10:
                                if r2h != r1h:
                                    continue
                            elif rule_2_rev[body][head] == 7 or rule_2_rev[body][head] == 9:
                                if r2t != r1h:
                                    continue
                            elif rule_2_rev[body][head] == 11 or rule_2_rev[body][head] == 12:
                                if r2t != r1t:
                                    continue
                            else:
                                print("extra pattern")
                            if [r1h, r2t] not in bn_list[body]:
                                bn_list[body].append([r1h, r2t])
                            for time_next2 in range(time_next1 + 1, time_next1 + window):
                                if head not in tgraph[time_next2]:
                                    continue
                                for ent_pair in tgraph[time_next2][head]:
                                    if ent_pair[0] == r1h and ent_pair[1] == r2t and ent_pair not in sd_list[body][
                                        head]:
                                        sd_list[body][head].append(ent_pair)
                sd[body][head] += len(sd_list[body][head])
                bn[body] += len(bn_list[body])
    time_end = pkg_time.time()
    print('run a head rule time:\t', time_end - time_start)

    # evaluate the SC of static len-2 rules in the pattern of r1(h1, t1, t+T2) <= r2(h2, t2, t) ^ r3(h3, t3, t+T1)
    sc_rule2_p1 = {}
    for head in rule_2:
        if head not in sc_rule2_p1:
            sc_rule2_p1[head] = {}
        for body in rule_2[head]:
            if bn[body] == 0.0:
                continue
            sc = sd[body][head] / bn[body]
            if sc > 0.1:
                sc_rule2_p1[head][body] = [rule_2[head][body], sc]
            if sc > 1.0:
                print(sc_rule2_p1[head][body])
        if not sc_rule2_p1[head]:
            sc_rule2_p1.pop(head)
    print("The total number of rules is:\t%d\n" % len(sc_rule2_p1))
    print("each rule and its confidence:\n", sc_rule2_p1)

    # learn len-2 rules in the pattern of r1(h1, t1, t+T) <= r2(h2, t2, t) ^ r3(h3, t3, t)
    time_start = pkg_time.time()
    sd = {}  # support degree
    sd_list = {}  # support degree of entity pair list
    bn = {}  # body degree    standard confidence = sd / bn
    bn_list = {}  # body number of entity pair list
    window = 30
    for body in rule_2_rev:
        # print("rule body:\t", body)
        # rule body rel in each length-1 rule
        if body not in bn:
            bn[body] = 0.
            # bn_list[body] = []
            sd[body] = {}
            sd_list[body] = {}
        for head in rule_2_rev[body]:
            # each head body rel in rule[body]
            sd[body][head] = 0.
            # sd_list[body][head] = []
            for time_cur in range(len(tgraph) - window):
                if body[0] not in tgraph[time_cur]:
                    continue
                bn_list[body] = []
                sd_list[body][head] = []
                for ent_pair in tgraph[time_cur][body[0]]:
                    r1h, r1t = ent_pair
                    if body[1] not in tgraph[time_cur]:
                        continue
                    for ent_pair in tgraph[time_cur][body[1]]:
                        r2h, r2t = ent_pair
                        if rule_2_rev[body][head] == 1 or rule_2_rev[body][head] == 2:
                            if r2h != r1h or r2t != r1t:
                                continue
                        elif rule_2_rev[body][head] == 4 or rule_2_rev[body][head] == 3:
                            if r2h != r1t or r2t != r1h:
                                continue
                        elif rule_2_rev[body][head] == 5 or rule_2_rev[body][head] == 8:
                            if r2h != r1t:
                                continue
                        elif rule_2_rev[body][head] == 6 or rule_2_rev[body][head] == 10:
                            if r2h != r1h:
                                continue
                        elif rule_2_rev[body][head] == 7 or rule_2_rev[body][head] == 9:
                            if r2t != r1h:
                                continue
                        elif rule_2_rev[body][head] == 11 or rule_2_rev[body][head] == 12:
                            if r2t != r1t:
                                continue
                        else:
                            print("extra pattern")
                        if [r1h, r2t] not in bn_list[body]:
                            bn_list[body].append([r1h, r2t])

                        for time_next in range(time_cur + 1, time_cur + window):
                            if head not in tgraph[time_next]:
                                continue
                            for ent_pair in tgraph[time_next][head]:
                                if ent_pair[0] == r1h and ent_pair[1] == r2t and ent_pair not in sd_list[body][head]:
                                    sd_list[body][head].append(ent_pair)
                sd[body][head] += len(sd_list[body][head])
                bn[body] += len(bn_list[body])
    time_end = pkg_time.time()
    print('run a head rule time:\t', time_end - time_start)

    # evaluate the SC of static len-2 rules in the pattern of r1(h1, t1, t+T) <= r2(h2, t2, t) ^ r3(h3, t3, t)
    sc_rule2_p2 = {}
    for head in rule_2:
        if head not in sc_rule2_p2:
            sc_rule2_p2[head] = {}
        for body in rule_2[head]:
            if bn[body] == 0.0:
                continue
            sc = sd[body][head] / bn[body]
            if sc > 0.1:
                sc_rule2_p2[head][body] = [rule_2[head][body], sc]
            if sc > 1.0:
                print(sc_rule2_p2[head][body])
        if not sc_rule2_p2[head]:
            sc_rule2_p2.pop(head)
    print("The total number of rules is:\t%d\n" % len(sc_rule2_p2))
    print("each rule and its confidence:\n", sc_rule2_p2)

    # learn len-2 rules in the pattern of r1(h1, t1, t+T) <= r2(h2, t2, t) ^ r3(h3, t3, t+T)
    import time
    time_start = pkg_time.time()
    sd = {}  # support degree
    sd_list = {}  # support degree of entity pair list
    bn = {}  # body degree    standard confidence = sd / bn
    bn_list = {}  # body number of entity pair list
    window = 30
    for body in rule_2_rev:
        # print("rule body:\t", body)
        # rule body rel in each length-1 rule
        if body not in bn:
            bn[body] = 0.
            # bn_list[body] = []
            sd[body] = {}
            sd_list[body] = {}
        for head in rule_2_rev[body]:
            # each head body rel in rule[body]
            sd[body][head] = 0.
            # sd_list[body][head] = []
            for time_cur in range(len(tgraph) - window):
                if body[0] not in tgraph[time_cur]:
                    continue
                bn_list[body] = []
                sd_list[body][head] = []
                for ent_pair in tgraph[time_cur][body[0]]:
                    r1h, r1t = ent_pair
                    if body[1] not in tgraph[time_cur]:
                        continue

                    for time_next in range(time_cur + 1, time_cur + window):
                        if body[1] not in tgraph[time_next]:
                            continue
                        for ent_pair in tgraph[time_next][body[1]]:
                            r2h, r2t = ent_pair
                            if rule_2_rev[body][head] == 1 or rule_2_rev[body][head] == 2:
                                if r2h != r1h or r2t != r1t:
                                    continue
                            elif rule_2_rev[body][head] == 4 or rule_2_rev[body][head] == 3:
                                if r2h != r1t or r2t != r1h:
                                    continue
                            elif rule_2_rev[body][head] == 5 or rule_2_rev[body][head] == 8:
                                if r2h != r1t:
                                    continue
                            elif rule_2_rev[body][head] == 6 or rule_2_rev[body][head] == 10:
                                if r2h != r1h:
                                    continue
                            elif rule_2_rev[body][head] == 7 or rule_2_rev[body][head] == 9:
                                if r2t != r1h:
                                    continue
                            elif rule_2_rev[body][head] == 11 or rule_2_rev[body][head] == 12:
                                if r2t != r1t:
                                    continue
                            else:
                                print("extra pattern")
                            if [r1h, r2t] not in bn_list[body]:
                                bn_list[body].append([r1h, r2t])

                            if head not in tgraph[time_next]:
                                continue
                            for ent_pair in tgraph[time_next][head]:
                                if ent_pair[0] == r1h and ent_pair[1] == r2t and ent_pair not in sd_list[body][head]:
                                    sd_list[body][head].append(ent_pair)
                sd[body][head] += len(sd_list[body][head])
                bn[body] += len(bn_list[body])
    time_end = pkg_time.time()
    print('run a head rule time:\t', time_end - time_start)

    # Evaluate the SC of len-2 rules in the pattern of r1(h1, t1, t+T) <= r2(h2, t2, t) ^ r3(h3, t3, t+T)
    sc_rule2_p3 = {}
    for head in rule_2:
        if head not in sc_rule2_p3:
            sc_rule2_p3[head] = {}
        for body in rule_2[head]:
            if bn[body] == 0.0:
                continue
            sc = sd[body][head] / bn[body]
            if sc > 0.1:
                sc_rule2_p3[head][body] = [rule_2[head][body], sc]
            if sc > 1.0:
                print(sc_rule2_p3[head][body])
        if not sc_rule2_p3[head]:
            sc_rule2_p3.pop(head)
    print("The total number of rules is:\t%d\n" % len(sc_rule2_p3))
    print("each rule and its confidence:\n", sc_rule2_p3)

    # learn len-2 rules in the pattern of r1(h1, t1, t) <= r2(h2, t2, t) ^ r3(h3, t3, t)
    time_start = pkg_time.time()
    sd = {}  # support degree
    sd_list = {}  # support degree of entity pair list
    bn = {}  # body degree    standard confidence = sd / bn
    bn_list = {}  # body number of entity pair list
    window = 30
    for body in rule_2_rev:
        # print("rule body:\t", body)
        # rule body rel in each length-1 rule
        if body not in bn:
            bn[body] = 0.
            # bn_list[body] = []
            sd[body] = {}
            sd_list[body] = {}
        for head in rule_2_rev[body]:
            # each head body rel in rule[body]
            sd[body][head] = 0.
            # sd_list[body][head] = []
            for time_cur in range(len(tgraph)):
                if body[0] not in tgraph[time_cur]:
                    continue
                bn_list[body] = []
                sd_list[body][head] = []
                for ent_pair in tgraph[time_cur][body[0]]:
                    r1h, r1t = ent_pair
                    if body[1] not in tgraph[time_cur]:
                        continue

                    if body[1] not in tgraph[time_cur]:
                        continue
                    for ent_pair in tgraph[time_cur][body[1]]:
                        r2h, r2t = ent_pair
                        if rule_2_rev[body][head] == 1 or rule_2_rev[body][head] == 2:
                            if r2h != r1h or r2t != r1t:
                                continue
                        elif rule_2_rev[body][head] == 4 or rule_2_rev[body][head] == 3:
                            if r2h != r1t or r2t != r1h:
                                continue
                        elif rule_2_rev[body][head] == 5 or rule_2_rev[body][head] == 8:
                            if r2h != r1t:
                                continue
                        elif rule_2_rev[body][head] == 6 or rule_2_rev[body][head] == 10:
                            if r2h != r1h:
                                continue
                        elif rule_2_rev[body][head] == 7 or rule_2_rev[body][head] == 9:
                            if r2t != r1h:
                                continue
                        elif rule_2_rev[body][head] == 11 or rule_2_rev[body][head] == 12:
                            if r2t != r1t:
                                continue
                        else:
                            print("extra pattern")
                        if [r1h, r2t] not in bn_list[body]:
                            bn_list[body].append([r1h, r2t])

                        if head not in tgraph[time_cur]:
                            continue
                        for ent_pair in tgraph[time_cur][head]:
                            if ent_pair[0] == r1h and ent_pair[1] == r2t and ent_pair not in sd_list[body][head]:
                                sd_list[body][head].append(ent_pair)
                sd[body][head] += len(sd_list[body][head])
                bn[body] += len(bn_list[body])
    time_end = pkg_time.time()
    print('run a head rule time:\t', time_end - time_start)

    # Evaluate the SC of len-2 rules in the pattern of r1(h1, t1, t) <= r2(h2, t2, t) ^ r3(h3, t3, t)
    sc_rule2_p4 = {}
    for head in rule_2:
        if head not in sc_rule2_p4:
            sc_rule2_p4[head] = {}
        for body in rule_2[head]:
            if bn[body] == 0.0:
                continue
            sc = sd[body][head] / bn[body]
            if sc > 0.1:
                sc_rule2_p4[head][body] = [rule_2[head][body], sc]
            if sc > 1.0:
                print(sc_rule2_p4[head][body])
        if not sc_rule2_p4[head]:
            sc_rule2_p4.pop(head)
    print("The total number of rules is:\t%d\n" % len(sc_rule2_p4))
    print("each rule and its confidence:\n", sc_rule2_p4)

    ff = open(os.path.join(data_path, 'rel_id'), 'r')
    rel_dict = {}
    for line in ff:
        rel, id = line.strip().split("\t")
        rel_dict[rel] = id
    ff.close()
    rule1_p1 = {}
    rule1_p1_t = {}
    for head in sc_rule1_p1:
        for body in sc_rule1_p1[head]:
            if head == body:
                break
            if int(rel_dict[head]) not in rule1_p1:
                rule1_p1[int(rel_dict[head])] = {}
                rule1_p1_t[int(rel_dict[head])] = {}
                rule1_p1[int(rel_dict[head]) + 230] = {}
                rule1_p1_t[int(rel_dict[head]) + 230] = {}
            patt_confi = sc_rule1_p1[head][body]
            if patt_confi[0] == 1:
                rule1_p1[int(rel_dict[head])][int(rel_dict[body])] = sc_rule1_p1[head][body][1]
                rule1_p1_t[int(rel_dict[head])][int(rel_dict[body])] = sc_rule1_p1[head][body][1]
                rule1_p1[int(rel_dict[head]) + 230][int(rel_dict[body]) + 230] = sc_rule1_p1[head][body][1]
                rule1_p1_t[int(rel_dict[head]) + 230][int(rel_dict[body]) + 230] = sc_rule1_p1[head][body][1]
            else:
                rule1_p1[int(rel_dict[head])][int(rel_dict[body]) + 230] = sc_rule1_p1[head][body][1]
                rule1_p1_t[int(rel_dict[head])][int(rel_dict[body]) + 230] = sc_rule1_p1[head][body][1]
                rule1_p1[int(rel_dict[head]) + 230][int(rel_dict[body])] = sc_rule1_p1[head][body][1]
                rule1_p1_t[int(rel_dict[head]) + 230][int(rel_dict[body])] = sc_rule1_p1[head][body][1]

    rule1_p2 = {}
    rule1_p2_t = {}
    for head in sc_rule1_p2:
        for body in sc_rule1_p2[head]:
            if head == body:
                break
            if int(rel_dict[head]) not in rule1_p2:
                rule1_p2[int(rel_dict[head])] = {}
                rule1_p2_t[int(rel_dict[head])] = {}
                rule1_p2[int(rel_dict[head]) + 230] = {}
                rule1_p2_t[int(rel_dict[head]) + 230] = {}
            patt_confi = sc_rule1_p2[head][body]
            if patt_confi[0] == 1:
                rule1_p2[int(rel_dict[head])][int(rel_dict[body])] = sc_rule1_p2[head][body][1]
                rule1_p2_t[int(rel_dict[head])][int(rel_dict[body])] = sc_rule1_p2[head][body][1]
                rule1_p2[int(rel_dict[head]) + 230][int(rel_dict[body]) + 230] = sc_rule1_p2[head][body][1]
                rule1_p2_t[int(rel_dict[head]) + 230][int(rel_dict[body]) + 230] = sc_rule1_p2[head][body][1]
            else:
                rule1_p2[int(rel_dict[head])][int(rel_dict[body]) + 230] = sc_rule1_p2[head][body][1]
                rule1_p2_t[int(rel_dict[head])][int(rel_dict[body]) + 230] = sc_rule1_p2[head][body][1]
                rule1_p2[int(rel_dict[head]) + 230][int(rel_dict[body])] = sc_rule1_p2[head][body][1]
                rule1_p2_t[int(rel_dict[head]) + 230][int(rel_dict[body])] = sc_rule1_p2[head][body][1]

    for head in rule1_p1:
        for body in rule1_p1[head]:
            if head in rule1_p2:
                if body in rule1_p2[head]:
                    if rule1_p1[head][body] > rule1_p2[head][body]:
                        rule1_p2_t[head].pop(body)
                    else:
                        rule1_p1_t[head].pop(body)

    for head in list(rule1_p1_t.keys()):
        if not rule1_p1_t[head]:
            del rule1_p1_t[head]
    for head in list(rule1_p2_t.keys()):
        if not rule1_p2_t[head]:
            rule1_p2_t.pop(head)

    with open(os.path.join(rule_path, 'rule1_p1.json'), 'w') as f:
        json.dump(rule1_p1_t, f)
    with open(os.path.join(rule_path, 'rule1_p2.json'), 'w') as f:
        json.dump(rule1_p2_t, f)

    # The 12 patterns of len-2 rules:
    rule2_p1 = {}
    rule2_p1_t = {}
    for head in sc_rule2_p1:
        for body in sc_rule2_p1[head]:
            if int(rel_dict[head]) not in rule2_p1:
                rule2_p1[int(rel_dict[head])] = {}
                rule2_p1_t[int(rel_dict[head])] = {}
                rule2_p1[int(rel_dict[head]) + 230] = {}
                rule2_p1_t[int(rel_dict[head]) + 230] = {}
            patt_confi = sc_rule2_p1[head][body]
            if patt_confi[0] == 1 or patt_confi[0] == 5 or patt_confi[0] == 9:
                rule2_p1[int(rel_dict[head])][(int(rel_dict[body[0]]), int(rel_dict[body[1]]))] = patt_confi[1]
                rule2_p1_t[int(rel_dict[head])][(int(rel_dict[body[0]]), int(rel_dict[body[1]]))] = patt_confi[1]
                rule2_p1[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]) + 230)] = \
                patt_confi[1]
                rule2_p1_t[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]) + 230)] = \
                patt_confi[1]
            if patt_confi[0] == 2 or patt_confi[0] == 7 or patt_confi[0] == 8:
                rule2_p1[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]), int(rel_dict[body[1]]))] = patt_confi[1]
                rule2_p1_t[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]), int(rel_dict[body[1]]))] = patt_confi[1]
                rule2_p1[int(rel_dict[head])][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]) + 230)] = \
                patt_confi[1]
                rule2_p1_t[int(rel_dict[head])][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]) + 230)] = \
                patt_confi[1]
            if patt_confi[0] == 3 or patt_confi[0] == 6 or patt_confi[0] == 12:
                rule2_p1[int(rel_dict[head])][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]))] = patt_confi[1]
                rule2_p1_t[int(rel_dict[head])][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]))] = patt_confi[1]
                rule2_p1[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]), int(rel_dict[body[1]]) + 230)] = \
                patt_confi[1]
                rule2_p1_t[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]), int(rel_dict[body[1]]) + 230)] = \
                patt_confi[1]
            if patt_confi[0] == 4 or patt_confi[0] == 10 or patt_confi[0] == 11:
                rule2_p1[int(rel_dict[head])][(int(rel_dict[body[0]]), int(rel_dict[body[1]]) + 230)] = patt_confi[1]
                rule2_p1_t[int(rel_dict[head])][(int(rel_dict[body[0]]), int(rel_dict[body[1]]) + 230)] = patt_confi[1]
                rule2_p1[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]))] = \
                patt_confi[1]
                rule2_p1_t[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]))] = \
                patt_confi[1]

    rule2_p2 = {}
    rule2_p2_t = {}
    for head in sc_rule2_p2:
        for body in sc_rule2_p2[head]:
            if int(rel_dict[head]) not in rule2_p2:
                rule2_p2[int(rel_dict[head])] = {}
                rule2_p2_t[int(rel_dict[head])] = {}
                rule2_p2[int(rel_dict[head]) + 230] = {}
                rule2_p2_t[int(rel_dict[head]) + 230] = {}
            patt_confi = sc_rule2_p2[head][body]
            if patt_confi[0] == 1 or patt_confi[0] == 5 or patt_confi[0] == 9:
                rule2_p2[int(rel_dict[head])][(int(rel_dict[body[0]]), int(rel_dict[body[1]]))] = patt_confi[1]
                rule2_p2_t[int(rel_dict[head])][(int(rel_dict[body[0]]), int(rel_dict[body[1]]))] = patt_confi[1]
                rule2_p2[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]) + 230)] = \
                patt_confi[1]
                rule2_p2_t[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]) + 230)] = \
                patt_confi[1]
            if patt_confi[0] == 2 or patt_confi[0] == 7 or patt_confi[0] == 8:
                rule2_p2[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]), int(rel_dict[body[1]]))] = patt_confi[1]
                rule2_p2_t[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]), int(rel_dict[body[1]]))] = patt_confi[1]
                rule2_p2[int(rel_dict[head])][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]) + 230)] = \
                patt_confi[1]
                rule2_p2_t[int(rel_dict[head])][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]) + 230)] = \
                patt_confi[1]
            if patt_confi[0] == 3 or patt_confi[0] == 6 or patt_confi[0] == 12:
                rule2_p1[int(rel_dict[head])][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]))] = patt_confi[1]
                rule2_p1_t[int(rel_dict[head])][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]))] = patt_confi[1]
                rule2_p1[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]), int(rel_dict[body[1]]) + 230)] = \
                patt_confi[1]
                rule2_p1_t[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]), int(rel_dict[body[1]]) + 230)] = \
                patt_confi[1]
            if patt_confi[0] == 4 or patt_confi[0] == 10 or patt_confi[0] == 11:
                rule2_p1[int(rel_dict[head])][(int(rel_dict[body[0]]), int(rel_dict[body[1]]) + 230)] = patt_confi[1]
                rule2_p1_t[int(rel_dict[head])][(int(rel_dict[body[0]]), int(rel_dict[body[1]]) + 230)] = patt_confi[1]
                rule2_p1[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]))] = \
                patt_confi[1]
                rule2_p1_t[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]))] = \
                patt_confi[1]

    rule2_p3 = {}
    rule2_p3_t = {}
    for head in sc_rule2_p3:
        for body in sc_rule2_p3[head]:
            if int(rel_dict[head]) not in rule2_p3:
                rule2_p3[int(rel_dict[head])] = {}
                rule2_p3_t[int(rel_dict[head])] = {}
                rule2_p3[int(rel_dict[head]) + 230] = {}
                rule2_p3_t[int(rel_dict[head]) + 230] = {}
            patt_confi = sc_rule2_p3[head][body]
            if patt_confi[0] == 1 or patt_confi[0] == 5 or patt_confi[0] == 9:
                rule2_p3[int(rel_dict[head])][(int(rel_dict[body[0]]), int(rel_dict[body[1]]))] = patt_confi[1]
                rule2_p3_t[int(rel_dict[head])][(int(rel_dict[body[0]]), int(rel_dict[body[1]]))] = patt_confi[1]
                rule2_p3[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]) + 230)] = \
                patt_confi[1]
                rule2_p3_t[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]) + 230)] = \
                patt_confi[1]
            if patt_confi[0] == 2 or patt_confi[0] == 7 or patt_confi[0] == 8:
                rule2_p3[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]), int(rel_dict[body[1]]))] = patt_confi[1]
                rule2_p3_t[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]), int(rel_dict[body[1]]))] = patt_confi[1]
                rule2_p3[int(rel_dict[head])][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]) + 230)] = \
                patt_confi[1]
                rule2_p3_t[int(rel_dict[head])][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]) + 230)] = \
                patt_confi[1]
            if patt_confi[0] == 3 or patt_confi[0] == 6 or patt_confi[0] == 12:
                rule2_p1[int(rel_dict[head])][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]))] = patt_confi[1]
                rule2_p1_t[int(rel_dict[head])][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]))] = patt_confi[1]
                rule2_p1[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]), int(rel_dict[body[1]]) + 230)] = \
                patt_confi[1]
                rule2_p1_t[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]), int(rel_dict[body[1]]) + 230)] = \
                patt_confi[1]
            if patt_confi[0] == 4 or patt_confi[0] == 10 or patt_confi[0] == 11:
                rule2_p1[int(rel_dict[head])][(int(rel_dict[body[0]]), int(rel_dict[body[1]]) + 230)] = patt_confi[1]
                rule2_p1_t[int(rel_dict[head])][(int(rel_dict[body[0]]), int(rel_dict[body[1]]) + 230)] = patt_confi[1]
                rule2_p1[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]))] = \
                patt_confi[1]
                rule2_p1_t[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]))] = \
                patt_confi[1]

    rule2_p4 = {}
    rule2_p4_t = {}
    for head in sc_rule2_p4:
        for body in sc_rule2_p4[head]:
            if int(rel_dict[head]) not in rule2_p4:
                rule2_p4[int(rel_dict[head])] = {}
                rule2_p4_t[int(rel_dict[head])] = {}
                rule2_p4[int(rel_dict[head]) + 230] = {}
                rule2_p4_t[int(rel_dict[head]) + 230] = {}
            patt_confi = sc_rule2_p4[head][body]
            if patt_confi[0] == 1 or patt_confi[0] == 5 or patt_confi[0] == 9:
                rule2_p4[int(rel_dict[head])][(int(rel_dict[body[0]]), int(rel_dict[body[1]]))] = patt_confi[1]
                rule2_p4_t[int(rel_dict[head])][(int(rel_dict[body[0]]), int(rel_dict[body[1]]))] = patt_confi[1]
                rule2_p4[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]) + 230)] = \
                patt_confi[1]
                rule2_p4_t[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]) + 230)] = \
                patt_confi[1]
            if patt_confi[0] == 2 or patt_confi[0] == 7 or patt_confi[0] == 8:
                rule2_p4[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]), int(rel_dict[body[1]]))] = patt_confi[1]
                rule2_p4_t[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]), int(rel_dict[body[1]]))] = patt_confi[1]
                rule2_p4[int(rel_dict[head])][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]) + 230)] = \
                patt_confi[1]
                rule2_p4_t[int(rel_dict[head])][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]) + 230)] = \
                patt_confi[1]
            if patt_confi[0] == 3 or patt_confi[0] == 6 or patt_confi[0] == 12:
                rule2_p1[int(rel_dict[head])][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]))] = patt_confi[1]
                rule2_p1_t[int(rel_dict[head])][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]))] = patt_confi[1]
                rule2_p1[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]), int(rel_dict[body[1]]) + 230)] = \
                patt_confi[1]
                rule2_p1_t[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]), int(rel_dict[body[1]]) + 230)] = \
                patt_confi[1]
            if patt_confi[0] == 4 or patt_confi[0] == 10 or patt_confi[0] == 11:
                rule2_p1[int(rel_dict[head])][(int(rel_dict[body[0]]), int(rel_dict[body[1]]) + 230)] = patt_confi[1]
                rule2_p1_t[int(rel_dict[head])][(int(rel_dict[body[0]]), int(rel_dict[body[1]]) + 230)] = patt_confi[1]
                rule2_p1[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]))] = \
                patt_confi[1]
                rule2_p1_t[int(rel_dict[head]) + 230][(int(rel_dict[body[0]]) + 230, int(rel_dict[body[1]]))] = \
                patt_confi[1]

    for head in list(rule2_p1.keys()):
        for body in list(rule2_p1[head].keys()):
            if head in rule2_p2:
                if body in rule2_p2[head]:
                    if rule2_p1[head][body] > rule2_p2[head][body]:
                        rule2_p2[head].pop(body)
                    else:
                        rule2_p1[head].pop(body)

    for head in list(rule2_p1.keys()):
        for body in list(rule2_p1[head].keys()):
            if head in rule2_p3:
                if body in rule2_p3[head]:
                    if rule2_p1[head][body] > rule2_p3[head][body]:
                        rule2_p3[head].pop(body)
                    else:
                        rule2_p1[head].pop(body)

    for head in list(rule2_p1.keys()):
        for body in list(rule2_p1[head].keys()):
            if head in rule2_p4:
                if body in rule2_p4[head]:
                    if rule2_p1[head][body] > rule2_p4[head][body]:
                        rule2_p4[head].pop(body)
                    else:
                        rule2_p1[head].pop(body)

    for head in list(rule2_p2.keys()):
        for body in list(rule2_p2[head].keys()):
            if head in rule2_p3:
                if body in rule2_p3[head]:
                    if rule2_p2[head][body] > rule2_p3[head][body]:
                        rule2_p3[head].pop(body)
                    else:
                        rule2_p2[head].pop(body)

    for head in list(rule2_p2.keys()):
        for body in list(rule2_p2[head].keys()):
            if head in rule2_p4:
                if body in rule2_p4[head]:
                    if rule2_p2[head][body] > rule2_p4[head][body]:
                        rule2_p4[head].pop(body)
                    else:
                        rule2_p2[head].pop(body)

    for head in list(rule2_p3.keys()):
        for body in list(rule2_p3[head].keys()):
            if head in rule2_p4:
                if body in rule2_p4[head]:
                    if rule2_p3[head][body] > rule2_p4[head][body]:
                        rule2_p4[head].pop(body)
                    else:
                        rule2_p3[head].pop(body)

    for head in list(rule2_p1.keys()):
        if not rule2_p1[head]:
            del rule2_p1[head]
    for head in list(rule2_p2.keys()):
        if not rule2_p2[head]:
            rule2_p2.pop(head)
    for head in list(rule2_p3.keys()):
        if not rule2_p3[head]:
            rule2_p3.pop(head)
    for head in list(rule2_p4.keys()):
        if not rule2_p4[head]:
            rule2_p4.pop(head)

    f = open(os.path.join(rule_path, 'rule2_p1.txt'), 'w')
    for head in rule2_p1:
        for body in rule2_p1[head]:
            f.write(str(head) + "\t" + str(body[0]) + "\t" + str(body[1]) + "\t" + str(rule2_p1[head][body]) + "\n")
    f.close()
    f = open(os.path.join(rule_path, 'rule2_p2.txt'), 'w')
    for head in rule2_p2:
        for body in rule2_p2[head]:
            f.write(str(head) + "\t" + str(body[0]) + "\t" + str(body[1]) + "\t" + str(rule2_p2[head][body]) + "\n")
    f.close()
    f = open(os.path.join(rule_path, 'rule2_p3.txt'), 'w')
    for head in rule2_p3:
        for body in rule2_p3[head]:
            f.write(str(head) + "\t" + str(body[0]) + "\t" + str(body[1]) + "\t" + str(rule2_p3[head][body]) + "\n")
    f.close()
    f = open(os.path.join(rule_path, 'rule2_p4.txt'), 'w')
    for head in rule2_p4:
        for body in rule2_p4[head]:
            f.write(str(head) + "\t" + str(body[0]) + "\t" + str(body[1]) + "\t" + str(rule2_p4[head][body]) + "\n")
    f.close()


def prepare_dataset_ori(data_path):
    """
    Given a path to a folder containing tab separated files :
     train, test, valid
    In the format :
    (lhs)\t(rel)\t(rhs)\t(timestamp)\n
    Maps each entity and relation to a unique id, create corresponding folder
    name in pkg/data, with mapped train/test/valid files.
    Also create to_skip_lhs / to_skip_rhs for filtered metrics and
    rel_id / ent_id for analysis.
    """
    files = ['train', 'valid', 'test']
    entities, relations, timestamps = set(), set(), set()
    for f in files:
        file_path = os.path.join(data_path, f)
        to_read = open(file_path, 'r')
        for line in to_read.readlines():
            lhs, rel, rhs, timestamp = line.strip().split('\t')
            entities.add(lhs)
            entities.add(rhs)
            relations.add(rel)
            timestamps.add(timestamp)
        to_read.close()

    entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
    relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}
    timestamps_to_id = {x: i for (i, x) in enumerate(sorted(timestamps))}

    print("{} entities, {} relations over {} timestamps".format(len(entities), len(relations), len(timestamps)))
    n_relations = len(relations)
    n_entities = len(entities)

    # write ent to id / rel to id
    for (dic, f) in zip([entities_to_id, relations_to_id, timestamps_to_id], ['ent_id', 'rel_id', 'ts_id']):
        ff = open(os.path.join(data_path, f), 'w+')
        for (x, i) in dic.items():
            ff.write("{}\t{}\n".format(x, i))
        ff.close()

    # map train/test/valid with the ids
    for f in files + ['infer']:
        file_path = os.path.join(data_path, f)
        to_read = open(file_path, 'r')
        examples = []
        for line in to_read.readlines():
            lhs, rel, rhs, ts = line.strip().split('\t')
            try:
                examples.append([entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs], timestamps_to_id[ts]])
            except ValueError:
                continue
        out = open(Path(data_path) / (f + '.pickle'), 'wb')
        pickle.dump(np.array(examples).astype('uint64'), out)
        out.close()

    print("creating filtering lists")

    # create filtering files
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for f in files:
        examples = pickle.load(open(Path(data_path) / (f + '.pickle'), 'rb'))
        for lhs, rel, rhs, ts in examples:
            to_skip['lhs'][(rhs, rel + n_relations, ts)].add(lhs)  # reciprocals
            to_skip['rhs'][(lhs, rel, ts)].add(rhs)

    to_skip_final = {'lhs': {}, 'rhs': {}}
    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    out = open(Path(data_path) / 'to_skip.pickle', 'wb')
    pickle.dump(to_skip_final, out)
    out.close()

    examples = pickle.load(open(Path(data_path) / 'train.pickle', 'rb'))
    counters = {
        'lhs': np.zeros(n_entities),
        'rhs': np.zeros(n_entities),
        'both': np.zeros(n_entities)
    }

    for lhs, rel, rhs, _ts in examples:
        counters['lhs'][lhs] += 1
        counters['rhs'][rhs] += 1
        counters['both'][lhs] += 1
        counters['both'][rhs] += 1
    for k, v in counters.items():
        counters[k] = v / np.sum(v)
    out = open(Path(data_path) / 'probas.pickle', 'wb')
    pickle.dump(counters, out)
    out.close()


def prepare_dataset(data_path):
    """
    Given a path to a folder containing tab separated files :
     train, test, valid
    In the format :
    (lhs)\t(rel)\t(rhs)\t(timestamp)\n
    Maps each entity and relation to a unique id, create corresponding folder
    name in pkg/data, with mapped train/test/valid files.
    Also create to_skip_lhs / to_skip_rhs for filtered metrics and
    rel_id / ent_id for analysis.
    """
    files = ['train', 'valid', 'test']
    entities, relations, timestamps = set(), set(), set()
    for f in files:
        file_path = os.path.join(data_path, f)
        to_read = open(file_path, 'r')
        for line in to_read.readlines():
            lhs, rel, rhs, timestamp = line.strip().split('\t')
            entities.add(lhs)
            entities.add(rhs)
            relations.add(rel)
            timestamps.add(timestamp)
        to_read.close()

    entities_to_id = {x: i for (i, x) in enumerate(sorted(entities))}
    relations_to_id = {x: i for (i, x) in enumerate(sorted(relations))}
    timestamps_to_id = {x: i for (i, x) in enumerate(sorted(timestamps))}

    print("{} entities, {} relations over {} timestamps".format(len(entities), len(relations), len(timestamps)))
    n_relations = len(relations)
    n_entities = len(entities)

    # write ent to id / rel to id
    for (dic, f) in zip([entities_to_id, relations_to_id, timestamps_to_id], ['ent_id', 'rel_id', 'ts_id']):
        ff = open(os.path.join(data_path, f), 'w+')
        for (x, i) in dic.items():
            ff.write("{}\t{}\n".format(x, i))
        ff.close()

    # map train/test/valid with the ids
    for f in files + ['infer']:
        file_path = os.path.join(data_path, f)
        to_read = open(file_path, 'r')
        examples = []
        for line in to_read.readlines():
            lhs, rel, rhs, ts = line.strip().split('\t')
            try:
                examples.append([entities_to_id[lhs], relations_to_id[rel], entities_to_id[rhs], timestamps_to_id[ts]])
            except ValueError:
                continue
        out = open(Path(data_path) / (f + '.pickle'), 'wb')
        pickle.dump(np.array(examples).astype('uint64'), out)
        out.close()

    print("creating filtering lists")

    # create filtering files
    to_skip = {'lhs': defaultdict(set), 'rhs': defaultdict(set)}
    for f in files:
        examples = pickle.load(open(Path(data_path) / (f + '.pickle'), 'rb'))
        for lhs, rel, rhs, ts in examples:
            to_skip['lhs'][(rhs, rel + n_relations, ts)].add(lhs)  # reciprocals
            to_skip['rhs'][(lhs, rel, ts)].add(rhs)

    to_skip_final = {'lhs': {}, 'rhs': {}}
    for kk, skip in to_skip.items():
        for k, v in skip.items():
            to_skip_final[kk][k] = sorted(list(v))

    out = open(Path(data_path) / 'to_skip.pickle', 'wb')
    pickle.dump(to_skip_final, out)
    out.close()

    examples = pickle.load(open(Path(data_path) / 'train.pickle', 'rb'))
    counters = {
        'lhs': np.zeros(n_entities),
        'rhs': np.zeros(n_entities),
        'both': np.zeros(n_entities)
    }

    for lhs, rel, rhs, _ts in examples:
        counters['lhs'][lhs] += 1
        counters['rhs'][rhs] += 1
        counters['both'][lhs] += 1
        counters['both'][rhs] += 1
    for k, v in counters.items():
        counters[k] = v / np.sum(v)
    out = open(Path(data_path) / 'probas.pickle', 'wb')
    pickle.dump(counters, out)
    out.close()


if __name__ == "__main__":
    args = parse_args()

    # merge found into train
    with open(os.path.join(args.data_path, 'train'), 'a') as fa:
        with open(os.path.join(args.data_path, 'found'), 'r') as fr:
            for line in fr.readlines():
                fa.write(line)
            fr.close()
        fa.close()


    """
    # generate static rules
    static_graph(args.data_path, args.iter)
    try:
        cmd_resule = subprocess.run(['java', '-jar', 'src_data/rulelearning/amie_plus.jar', os.path.join(args.data_path, str(args.iter) + '/kge/rulelearning/triples.tsv')], check=True, capture_output=True, text=True)
        print(cmd_resule.stdout)
        print(cmd_resule.stderr)
        with open(os.path.join(args.data_path, str(args.iter) + '/kge/rulelearning/amie_rules_static.txt'), 'w') as fw:
            fw.write(cmd_resule.stdout)
            fw.close()
    except Exception as e:
        print(e)
        exit(1)

    # reform rule
    reform_static_rule(args.data_path, args.iter)
    learn_temporal_rule(args.data_path, args.iter)
    """

    # prepare
    prepare_dataset(args.data_path)


