import torch
import pickle
from tqdm import tqdm
import json
import os
import logging
import argparse

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from accelerate import Accelerator

from format_question import load_alignment_text, hop_query_to_text, load_data, load_info
from config.base_prompt import get_question_from_prompt

class Pipeline:
    def __init__(self, encoded_entities, tokenizer, model, batch_size, top_k, device="cuda", kv_cache=False):
        self.device = device
        self.tokenizer = tokenizer

        self.vocab = tokenizer.get_vocab()

        self.pad_token_id = self.vocab[tokenizer.pad_token]
        self.stop_token_id = torch.tensor(list({
            self.vocab["<|end_of_text|>"], 
            self.vocab["<|eot_id|>"], 
            self.vocab[tokenizer.eos_token], 
            self.vocab[tokenizer.bos_token], 
            self.vocab['<|start_header_id|>'], 
            self.vocab['.']
            })
        )
        self.num_entity = len(encoded_entities.keys())

        logging.info(f"{self.num_entity} entities in total")

        self.max_token_length = self._max_token_length(encoded_entities)

        self.tokenized_entites, self.token_mask = self._tokenize_entities(encoded_entities)

        self.batch_size = batch_size
        self.use_cache = kv_cache

        self.top_k = top_k

        self.model = model

    def _tokenize_base_prompt(self, question):
        messages = get_question_from_prompt(question)

        structured_q = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # update base question tokens
        input_ids = self.tokenizer(structured_q, return_tensors="pt", padding=False).to(self.device)

        return input_ids

    def _tokenize_entities(self, encoded_entities):
        max_len = self.max_token_length
        
        size = (self.num_entity, max_len)
        tokenized_entites = torch.full(size, -1).to(self.device)

        for i in range(self.num_entity):
            tokenized_entites[i, :len(encoded_entities[i])] = torch.tensor(encoded_entities[i]).to(self.device)

        token_mask = (tokenized_entites != -1)

        return tokenized_entites, token_mask

    def _max_token_length(self, encoded_entities):
        max_len = -1
        for i in range(self.num_entity):
            if len(encoded_entities[i]) > max_len:
                max_len = len(encoded_entities[i])
        
        return max_len

    def _get_start_tokens(self):
        start_tokens = self.tokenized_entites[:, 0].squeeze().to(self.device)

        start_set = torch.tensor(list(set(start_tokens.tolist()))).to(self.device)

        return start_tokens, start_set

    def batch_retrieve_prob(self, input_ids, past_key_values=None, cache_position=None):
        with torch.no_grad():
            if past_key_values is not None:
                output = self.model(**input_ids, past_key_values=past_key_values, cache_position=cache_position)
            else:   
                output = self.model(**input_ids)

            logits = output.logits

        return logits
    
    def get_batch_start_token_prob(self, questions):
        all_messages = [get_message_from_promp(q) for q in questions]
        
        structured_q = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # update base question tokens
        input_ids = self.tokenizer(structured_q, return_tensors="pt", padding=False).to(self.device)

        output = self.model(**input_ids)

        breakpoint()

    def get_start_token_prob(self, question):
        input_ids = self._tokenize_base_prompt(question)

        output = self.model(**input_ids)

        start_logits = output.logits[:, -1, :]
        start_prob = torch.nn.functional.softmax(start_logits[-1], dim=0)
        start_tokens, start_set = self._get_start_tokens()

        prob_start_set = start_prob[start_set].to(self.device)

        prob_init = start_prob[start_tokens].to(self.device)
        base_kv = output.past_key_values

        return input_ids, prob_init, base_kv, prob_start_set

    def _get_selected_prob_in_batch(self, base_tokens, base_kv, indices, full_question_tokens, top_prob, num_selected):
        if self.use_cache:
            past_key_values = tuple(
                    tuple(past_kv.repeat(self.batch_size, 1, 1, 1) for past_kv in layer_past)
                    for layer_past in base_kv
                )

        s = 0
        while s < num_selected:
            bs = min(s + self.batch_size, num_selected) - s
            if bs != self.batch_size and self.use_cache:
                past_key_values = tuple(
                    tuple(past_kv.repeat(bs, 1, 1, 1) for past_kv in layer_past)
                    for layer_past in base_kv
                )

            inputs = {}

            if self.use_cache:
                batched_fq_tokens = self.tokenized_entites[indices[s:min(s+self.batch_size, num_selected)], :]
            else:
                batched_fq_tokens = full_question_tokens[indices[s:min(s+self.batch_size, num_selected)], :]

            attention_mask = (batched_fq_tokens != -1)

            batched_fq_tokens[~attention_mask] = self.pad_token_id
            inputs["input_ids"] = batched_fq_tokens

            new_inputs_attention = attention_mask.to(int)
            inputs["attention_mask"] = torch.cat((base_tokens["attention_mask"].repeat(new_inputs_attention.size(0), 1), new_inputs_attention), dim=1)

            # apply kv cache
            if self.use_cache:
                start_pos = base_tokens["input_ids"].size(1)
                end_pos = start_pos + batched_fq_tokens.size(1)

                cache_position = torch.arange(start_pos, end_pos).to(self.device)

                logits = self.batch_retrieve_prob(inputs, past_key_values, cache_position)
            
            else:
                logits = self.batch_retrieve_prob(inputs)

            normalized = torch.nn.functional.softmax(logits, dim=-1)

            base_token_length = 0 if self.use_cache else base_tokens["input_ids"].size(1) - 1
            prob_batch = torch.ones(bs, device=self.device)

            valid_lengths = attention_mask.sum(dim=1)
            
            valid_lengths = attention_mask.sum(dim=1) - 1
            tokens = (self.tokenized_entites[indices[s:s+bs], 1:][self.token_mask[indices[s:s+bs], 1:]]).unsqueeze(-1)
            bs_range = torch.arange(bs).unsqueeze(-1).to(self.device)
            head = torch.repeat_interleave(bs_range, valid_lengths).unsqueeze(-1)

            cumsum_l = torch.cumsum(valid_lengths, 0).to(self.device)
            ind_l = torch.arange(cumsum_l[-1]).to(self.device)
            
            # build offsets
            offsets = torch.zeros_like(ind_l).to(self.device)
            offsets[cumsum_l[:-1]] = valid_lengths[:-1]
            offsets = torch.cumsum(offsets, 0)

            token_seq = (ind_l - offsets).unsqueeze(-1)
            target_pick = torch.cat((head, token_seq, tokens), dim=1)

            prob_batch = torch.ones(bs).to(self.device)
            
            n_flatten = normalized.flatten()
            token_length = normalized.size(1)
            v_size = normalized.size(-1)
            target_linear = target_pick[:, 0] * (token_length * v_size) + target_pick[:, 1] * v_size + target_pick[:, 2]

            prob_batch = prob_batch.scatter_reduce(0, target_pick[:, 0], n_flatten[target_linear], reduce="prod")

            batch_indicies = torch.arange(bs)
            final_probs = normalized[batch_indicies, valid_lengths, :]
            batch_stop_probs = final_probs[:, self.stop_token_id]
            prob_stop, _ = torch.max(batch_stop_probs, dim=1)

            prob_batch *= prob_stop

            top_prob[s:min(s + self.batch_size, num_selected)] *= prob_batch
            s += self.batch_size

        # breakpoint()

        return top_prob

    def get_prob(self, question, easy_answer):
        start_token, start_token_set = self._get_start_tokens()

        base_tokens, prob_init, base_kv, prob_start_set = self.get_start_token_prob(question)
        top_tokens = start_token_set[torch.sort(prob_start_set, descending=True)[1]][:self.top_k]

        mask = torch.isin(start_token, top_tokens)
        top_indices = torch.nonzero(mask).squeeze()

        mask_easy = torch.isin(top_indices, torch.tensor(list(easy_answer)).to(self.device))
        top_indices = top_indices[torch.nonzero(~mask_easy).squeeze()]

        top_prob = prob_init[top_indices]
        
        tail_mask = ~mask
        tailing_indices = torch.nonzero(tail_mask).squeeze()
        tailing_prob = prob_init[tailing_indices]

        # tracks probability for all entities
        prob = prob_init.clone()

        # prob_sorted, indices = torch.sort(prob_init, descending=True)
        # top_prob, top_indices = prob_sorted[:self.top_k].clone(), indices[:self.top_k].clone()
        # tailing_prob, tailing_indices = prob_sorted[self.top_k:].clone(), indices[self.top_k:].clone()

        start_token = base_tokens["input_ids"][base_tokens["attention_mask"] == 1]
        full_question_tokens = torch.cat((start_token.repeat(self.num_entity, 1), self.tokenized_entites), dim=1)

        logging.info(f"Picking top {top_prob.size(0)}")
        top_prob = self._get_selected_prob_in_batch(base_tokens, base_kv, top_indices, full_question_tokens, top_prob, top_prob.size(0))

        # prob[top_indices] = top_prob
        # threshold = torch.mean(top_prob).item()

        # logging.info(threshold)

        # # processing indicies that still possible to reach
        # tailing_valid_indicies = torch.where(tailing_prob > threshold)[0]
        # num_valid = tailing_valid_indicies.size(0)

        # if num_valid > 0:
        #     logging.info(f"Still need to check {num_valid} entities")
        #     tailing_original_indicies = tailing_indices[tailing_valid_indicies]

        #     tailing_valid_probs = tailing_prob[tailing_valid_indicies]
        #     retrieved_prob = self._get_selected_prob_in_batch(base_tokens, base_kv, tailing_indices, full_question_tokens, tailing_valid_probs, num_valid)

        #     prob[tailing_original_indicies] = retrieved_prob

        # # re_rank invalid indicies based on 
        # tailing_invalid_indices = tailing_indices[torch.where(tailing_prob <= threshold)[0]]
        
        # if num_valid > 0 and tailing_invalid_indices.size(0) > 0:
        #     tailing_invalid_probs= torch.sort(prob[tailing_invalid_indices])[0]
        #     sorted_retrieve_prob = torch.sort(retrieved_prob)[0]

        #     if tailing_invalid_probs[0] > sorted_retrieve_prob[-1]:
        #         logging.warning("Untraversed entitiy has greater init probability than explored")

        #     prob[tailing_invalid_indices] = 0

        return top_prob, top_indices