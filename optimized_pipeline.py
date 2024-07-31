
import torch
import pickle
from tqdm import tqdm
import json
import os
import logging
import argparse

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StaticCache
from huggingface_hub import login
from accelerate import Accelerator

from format_question import load_alignment_text, hop_query_to_text, load_data, load_info
from pipeline import Pipeline
from dataset import get_data, PredictHeadDataset, PredictTailDataset

login("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")

query_name_dict = {('e',('r',)): '1p', 
                    ('e', ('r', 'r')): '2p',
                    ('e', ('r', 'r', 'r')): '3p',
                    (('e', ('r',)), ('e', ('r',))): '2i',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                    ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                    (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                    (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                    ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                    (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                    (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                    (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
                }
name_query_dict = {value: key for key, value in query_name_dict.items()}
all_tasks = list(name_query_dict.keys())

def split_line(line):
    first_split = line.strip().find("\t")

    if first_split == -1:
        return None, None
    
    return line[:first_split], line[first_split+1:]

def construct_vocab_token(tokenizer, id2ent, ent2text):
    id_to_encoded_e = {}
    encoded_entities = {}
    id_to_ent = {}
    
    num_accumulated = 0
    for id in tqdm(id2ent):
        ent_name = ent2text[id2ent[id]].strip()

        # if ent_name[-1] != ".":
        #     ent_name += "."
        # ent_name = "[" + ent_name

        if ent_name in encoded_entities:
            continue

        encoded_input = tokenizer.encode(ent_name)
        special_token_mask = 1 - torch.tensor(tokenizer.get_special_tokens_mask(encoded_input, already_has_special_tokens=True))

        masked_token = [token for token, is_normal in zip(encoded_input, special_token_mask) if is_normal]
        
        encoded_entities[ent_name] = masked_token
        id_to_encoded_e[num_accumulated] = masked_token
        id_to_ent[num_accumulated] = ent_name
    
        num_accumulated += 1

    with open("encoded_entities.json", "w") as f:
        json.dump(encoded_entities, f)

    with open("entity_encoding.json", "w") as f:
        json.dump(id_to_encoded_e, f)

    return encoded_entities, id_to_encoded_e, id_to_ent

def construct_vocab_token_kgc(tokenizer, id2ent):
    id_to_encoded_e = {}
    encoded_entities = {}
    id_to_ent = {}
    
    for id in tqdm(id2ent):
        ent_name = id2ent[id].strip()

        encoded_input = tokenizer.encode(ent_name)
        special_token_mask = 1 - torch.tensor(tokenizer.get_special_tokens_mask(encoded_input, already_has_special_tokens=True))

        masked_token = [token for token, is_normal in zip(encoded_input, special_token_mask) if is_normal]
        
        if ent_name not in encoded_entities:
            encoded_entities[ent_name] = masked_token

        id_to_encoded_e[int(id)] = masked_token
        id_to_ent[int(id)] = ent_name

    with open("./data/kgc_data/encoded_entities.json", "w") as f:
        json.dump(encoded_entities, f)

    with open("./data/kgc_data/entity_encoding.json", "w") as f:
        json.dump(id_to_encoded_e, f)

    return encoded_entities, id_to_encoded_e, id_to_ent

def re_map(ent_to_id, id2ent, ent2text, answers):
    remapping = []

    for ans in answers:
        entname = ent2text[id2ent[ans]].strip()
        remapped_id = ent_to_id[entname]
        remapping.append(remapped_id)

    return remapping

def flatten_query(queries):
    all_queries = []
    for query_structure in queries:
        tmp_queries = list(queries[query_structure])
        all_queries.extend([(query, query_structure) for query in tmp_queries])
    return all_queries

def forward(hard_answer, easy_answer, q):
    hard_answer = {int(ele) for ele in hard_answer}
    easy_answer = {int(ele) for ele in easy_answer}

    probs, indices = pipeline.get_prob(q, easy_answer)

    num_valid = torch.sum(torch.isin(torch.tensor(list(hard_answer)).to(device), indices).long())

    all_probs = torch.zeros(len(id_to_encoded_e.keys())).to(device)
    all_probs[indices] = probs
    argsort = torch.argsort(all_probs, dim=0, descending=True)

    re_ranked_entities = [id_to_ent[i.item()]for i in argsort]

    logging.info(f"Predict Answers {re_ranked_entities[:num_results]}")

    ranking = argsort.clone().to(torch.float).to(device)
    ranking = ranking.scatter_(0, argsort, torch.arange(all_probs.size(0)).to(torch.float).to(device))

    cur_ranking = ranking[list(hard_answer)]
    cur_ranking, indices = torch.sort(cur_ranking)
    answer_list = torch.arange(num_hard).to(torch.float).to(device)
    
    cur_ranking = cur_ranking - answer_list + 1
    cur_ranking = cur_ranking[:num_valid]

    return torch.sum(1./cur_ranking).item()

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description='Parse the number of queries.')
    parser.add_argument('-q', dest='num_queries', type=int, default=900, help='Number of queries')
    parser.add_argument("-r", dest="num_results", type=int, default=20, help="Number of answer logged")
    parser.add_argument("-p", dest="data_path", type=str, default="./data/kgc_data", help="path where data is stored")
    parser.add_argument("-b", dest="batch_size", type=int, default=512, help="batch size of entities")
    parser.add_argument("-m", dest="model_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Specify which model to run with")
    parser.add_argument("--quant", action="store_true", help="Use 8-bit quantization")
    
    args = parser.parse_args()
    
    num_queries = args.num_queries
    num_results = args.num_results
    data_path = args.data_path
    batch_size = args.batch_size

    torch.cuda.empty_cache()

    if args.quant:
        model_id = "astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit"
        logging.info("Using quantization version")
        quantize_config = BaseQuantizeConfig(
                bits=4,
                group_size=128,
                desc_act=True,
                damp_percent=0.1,
                true_sequential=True,
                sym=True,
        )

        model = AutoGPTQForCausalLM.from_quantized(
            "astronomer/Llama-3-8B-Instruct-GPTQ-8-Bit", 
            device_map="auto"
        )
    else: 
        model_id = args.model_path
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            _attn_implementation="flash_attention_2",
        )

        # model._static_cache = StaticCache(
        #     config=model.config,
        #     max_batch_size=1,
        #     max_cache_len=4096,
        #     device=model.device,
        #     dtype=torch.float16,
        # )

        # model.generation_config.cache_implementation = "static"
        # model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

    accelerator = Accelerator()
    model = accelerator.prepare(model)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()

    vocab = tokenizer.get_vocab()
    vocab_rev = {v: k for k, v in vocab.items()}

    # id2ent, id2rel, ent2text = load_info(data_path)
    # encoded_entities, id_to_encoded_e, id_to_ent = construct_vocab_token(tokenizer, id2ent, ent2text)
    # alignment_data = load_alignment_text(os.path.join(data_path, "alignment_clean.json"))
    # test_queries, test_hard_answers, test_easy_answers = load_data(["1p"], data_path)

    # query_questions = {}
    # for i, query  in enumerate(test_queries[('e', ('r', ))]):
    #     query_q = hop_query_to_text(alignment_data, id2ent, id2rel, ent2text, query)
    #     if query_q:
    #         query_questions[query] = (query_q + " [Y] is ")

    #     if len(query_questions) == num_queries:
    #         break
    ent_path = os.path.join(data_path, "id2ent.json")
    rel_path = os.path.join(data_path, "id2symbol_rel.json")
    
    id2ent = json.load(open(ent_path, "r"))
    id2rel = json.load(open(rel_path, "r"))

    tuples_train, predict_tail_train, predict_head_train, tuples_valid, \
            predict_tail_valid, predict_head_valid, tuples_test, predict_tail_test, predict_head_test = get_data(data_path)
    
    encoded_entities, id_to_encoded_e, id_to_ent = construct_vocab_token_kgc(tokenizer, id2ent)

    query_questions = []
    for i, q in enumerate(tuples_test):
        if i > num_queries:
            break
        else:
            head = q[0]
            relation = q[1]
            tail = q[2]

            predict_head_q = f"Complete this triple. Head Entity: [X], Relation: {id2rel[relation]}, Tail entity: {id2ent[tail].strip()} \nWhat is [X]?"
            predict_tail_q = f"Complete this triple. Head entity: {id2ent[head].strip()}, Relation: {id2rel[relation]}, Tail Entity: [X] \nWhat is [X]?" 

            query_questions.append((q, predict_head_q, predict_tail_q))

    MRR = 0
    answers = 0

    pipeline = Pipeline(id_to_encoded_e, tokenizer, model, batch_size, 16, kv_cache=True)

    for query in tqdm(query_questions):
        t, p_head, p_tail = query

        # head prediction
        head_hard_answer = set(predict_head_test[(t[1], t[2])])
        head_easy_answer = set(predict_head_train[(t[1], t[2])] + predict_head_valid[(t[1], t[2])])
        num_hard = len(head_hard_answer)
        num_easy = len(head_easy_answer)

        logging.info(f"Answers are {[id2ent[i].strip() for i in list(head_hard_answer)]}")
        
        mrr = forward(head_hard_answer, head_easy_answer, p_head)
        answers += num_hard
        MRR += mrr

        logging.info(f"Predict Head {p_head} has MRR {mrr / num_hard}")
        logging.info(MRR / answers)

        tail_hard_answer = set(predict_tail_test[(t[0], t[1])])
        tail_easy_answer = set(predict_tail_test[(t[0], t[1])] + predict_tail_valid[(t[0], t[1])])
        num_hard = len(tail_hard_answer)
        num_easy = len(tail_easy_answer)

        logging.info(f"Answers are {[id2ent[i].strip() for i in list(tail_hard_answer)]}")

        mrr = forward(tail_hard_answer, tail_easy_answer, p_tail)
        logging.info(f"Predict Tail {p_tail} has MRR {mrr / num_hard}")

        answers += num_hard
        MRR += mrr

        logging.info(MRR / answers)

    mrr = MRR / answers

    print(mrr)
