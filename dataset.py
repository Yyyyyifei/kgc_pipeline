import argparse
import json
import os
import random
import torch

from collections import defaultdict
from format_question import split_line

from tqdm import tqdm
from config.data_format import data_formatting

from torch.utils.data import Dataset

def indexify(train_path, valid_path, test_path):
    id_2_ent = {}
    ent_2_id = {}

    id_2_rel = {}
    rel_2_id = {}

    num_entity = 0
    num_relation = 0

    for file in (train_path, valid_path, test_path):
        with open(file, "r") as f:
            lines = f.readlines()

            for l in lines:
                # Fix: strip
                head, relation, tail = l.split()
                head, relation, tail = head.strip(), relation.strip(), tail.strip()
                if head not in ent_2_id:
                    id_2_ent[num_entity] = head
                    ent_2_id[head] = num_entity
                    num_entity += 1

                if tail not in ent_2_id:
                    id_2_ent[num_entity] = tail
                    ent_2_id[tail] = num_entity
                    num_entity += 1
                
                if relation not in rel_2_id:
                    id_2_rel[num_relation] = relation
                    rel_2_id[relation] = num_relation
                    num_relation += 1

    return id_2_ent, ent_2_id, id_2_rel, rel_2_id

def id2text(id2ent, ent2text, id2rel, rel2text):
    id2ent_text = {id : ent2text[id2ent[id]] for id in id2ent}
    id2rel_text = {id : rel2text[id2rel[id]] for id in id2rel}

    return id2ent_text, id2rel_text

def ent_to_text(datapath):
    ent2text = {}

    with open(os.path.join(datapath, "entity2text.txt"), "r") as f:
        for line in f.readlines():
            key, value = split_line(line)
            ent2text[key] = value

    return ent2text

def parse_file(file, entities, relations):
    all_tuples = []
    predict_tail = defaultdict(list)
    predict_head = defaultdict(list)

    with open(file, "r") as f:
        lines = f.readlines()

        for l in lines:
            head, relation, tail = l.split()
            index_head, index_relation, index_tail = entities[head.strip()], relations[relation.strip()], entities[tail.strip()] 

            all_tuples.append((index_head, index_relation, index_tail))
            
            predict_tail[(index_head, index_relation)].append(index_tail)
            predict_head[(index_relation, index_tail)].append(index_head)

    return all_tuples, predict_tail, predict_head

def get_data(datapath):
    # default datapath as "./data"
    ent_path = os.path.join(datapath, "id2ent.json")
    rel_path = os.path.join(datapath, "id2rel.json")

    id2symbol_ent_path = os.path.join(datapath, "id2symbol_ent.json")
    id2symbol_rel_path = os.path.join(datapath, "id2symbol_rel.json")

    train_path = os.path.join(datapath, "train.txt")
    valid_path = os.path.join(datapath, "valid.txt")
    test_path = os.path.join(datapath, "test.txt")

    if os.path.exists(ent_path):
        id2ent = json.load(open(ent_path, "r"))
        id2rel = json.load(open(rel_path, "r"))

        id2symbol_ent = json.load(open(id2symbol_ent_path, "r"))
        id2symbol_rel = json.load(open(id2symbol_rel_path, "r"))

        symbol_ent2id = {v : k for k,v in id2symbol_ent.items()}
        symbol_rel2id = {v : k for k,v in id2symbol_rel.items()}
    else:
        id2symbol_ent, symbol_ent2id, id2symbol_rel, symbol_rel2id = indexify(train_path, valid_path, test_path)
        
        with open(id2symbol_ent_path, "w") as f:
            json.dump(id2symbol_ent, f)
            f.close()
        
        with open(id2symbol_rel_path, "w") as f:
            json.dump(id2symbol_rel, f)
            f.close()

        relation2text = json.load(open(os.path.join(datapath, "alignment_clean.json"), "r"))
        ent2text = ent_to_text(datapath)
        
        id2ent, id2rel = id2text(id2symbol_ent, ent2text, id2symbol_rel, relation2text)

        with open(ent_path, "w") as f:
            json.dump(id2ent, f)
            f.close()
        
        with open(rel_path, "w") as f:
            json.dump(id2rel, f)
            f.close()
    
    all_tuples_train, predict_tail_train, predict_head_train = parse_file(train_path, symbol_ent2id, symbol_rel2id)
    all_tuples_valid, predict_tail_valid, predict_head_valid = parse_file(valid_path, symbol_ent2id, symbol_rel2id)
    all_tuples_test, predict_tail_test, predict_head_test = parse_file(test_path, symbol_ent2id, symbol_rel2id)

    return all_tuples_train, predict_tail_train, predict_head_train, \
                all_tuples_valid, predict_tail_valid, predict_head_valid, \
                    all_tuples_test, predict_tail_test, predict_head_test

class PredictTailDataset(Dataset):
    def __init__(self, predict_tail_train, predict_tail_valid, predict_tail_test, num_entity):
        self.len = len(predict_tail_test.keys())

        self.train = predict_tail_train
        self.valid = predict_tail_valid
        self.test = predict_tail_test

        self.num_entity = num_entity

        self.queries = [(q, a) for q, a in predict_tail_test.items()]
        self.easy_answer, self.hard_answer = self._get_answer(predict_tail_train, predict_tail_valid, predict_tail_test)
    
    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query, hard_ans = self.queries[idx]
        easy_ans = self.easy_answer[query]

        return query, hard_ans, easy_ans

    def _get_answer(self, predict_train, predict_valid, predict_test):
        easy_qa = defaultdict(list)
        hard_qa = defaultdict(list)

        for q in predict_test:
            easy_answer = predict_train.get(q, []) + predict_valid.get(q, [])
            hard_answer = predict_test.get(q, [])

            assert len(hard_answer) != 0, f"{q} has none valid test answer"

            easy_qa[q] = easy_answer
            hard_qa[q] = hard_answer
        
        return easy_qa, hard_qa

class PredictHeadDataset(Dataset):
    def __init__(self, predict_head_train, predict_head_valid, predict_head_test, num_entity):
        self.len = len(predict_head_test.keys())

        self.train = predict_head_train
        self.valid = predict_head_valid
        self.test = predict_head_test

        self.num_entity = num_entity

        self.queries = [(q, a) for q, a in predict_head_test.items()]
        self.easy_answer, self.hard_answer = self._get_answer(predict_head_train, predict_head_valid, predict_head_test)
    
    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        breakpoint()
        query, hard_ans = self.queries[idx]
        easy_ans = self.easy_answer[query]

        return query, hard_ans, easy_ans

    def _get_answer(self, predict_head_train, predict_head_valid, predict_head_test):
        easy_qa = defaultdict(list)
        hard_qa = defaultdict(list)

        for q in predict_head_test:
            easy_answer = predict_head_train.get(q, []) + predict_head_valid.get(q, [])
            hard_answer = predict_head_test.get(q, [])

            assert len(hard_answer) != 0, f"{q} has none valid test answer"

            easy_qa[q] = easy_answer
            hard_qa[q] = hard_answer
        
        return easy_qa, hard_qa
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse the number of queries.')
    parser.add_argument("-p", dest="data_path", type=str, default="./data/kgc_data", help="path where data is stored")
    
    args = parser.parse_args()

    path = args.data_path

    tuples_train, predict_tail_train, predict_head_train, tuples_valid, \
        predict_tail_valid, predict_head_valid, tuples_test, predict_tail_test, predict_head_test = get_data(path)

    ent_path = os.path.join(path, "id2ent.json")
    rel_path = os.path.join(path, "id2rel.json")

    id2symbol_ent_path = os.path.join(path, "id2symbol_ent.json")
    id2symbol_rel_path = os.path.join(path, "id2symbol_rel.json")
    
    id2ent = json.load(open(ent_path, "r"))
    id2rel = json.load(open(rel_path, "r"))

    id2symbol_ent = json.load(open(id2symbol_ent_path, "r"))
    id2symbol_rel = json.load(open(id2symbol_rel_path, "r"))

    dataset = NegativeDataset(tuples_train, predict_head_train, predict_head_valid, len(id2ent))
    dataset.generate_negative("/scratch/ssd004/scratch/zjt/LLaMA-Factory/data/train.json", 4, id2ent, id2symbol_rel)

    predict_head_valid.update(predict_head_train)
    predict_tail_valid.update(predict_tail_train)
    dataset = NegativeDataset(tuples_valid, predict_head_valid, predict_tail_valid, len(id2ent))
    dataset.generate_negative("/scratch/ssd004/scratch/zjt/LLaMA-Factory/data/valid.json", 4, id2ent, id2symbol_rel)

    # print(len(predict_tail_test) + len(predict_head_test))

    
