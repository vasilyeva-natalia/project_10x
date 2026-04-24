import gc
import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pprint import pprint
from typing import Callable
from datetime import datetime

import random
import time
from tqdm.autonotebook import tqdm as notebook_tqdm
import torch
from torch.utils.data import Dataset, DataLoader
torch.set_float32_matmul_precision('high')

from huggingface_hub import login
import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoConfig, logging

from pathlib import Path
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Dataset10x(Dataset):
    def __init__(self, report_df: pd.DataFrame, items_path: str = 'items') -> None:
        super().__init__()

        self.target_columns = ['FILING_DATE', 'CIK', 'ACC_NUM']
        for col in self.target_columns:
            assert col in report_df.columns, col

        self.name_columns = [self.target_columns[0], '10-K_edgar_data'] + self.target_columns[1:]
        self.item_names = ['item1', 'item1a', 'item7']
        self.items_path = items_path
        self.report_df = report_df

    def __len__(self) -> int:
        return len(self.report_df)

    def __getitem__(self, idx: int) -> dict[str, str]:
        report_dict = {}
        notnone = False
        report = self.report_df.iloc[idx].to_dict()
        report_name = '_'.join([str(report[x]) if x in report.keys() else x for x in self.name_columns])

        for item_name in self.item_names:
            item_pathname = os.path.join(self.items_path, f'{item_name}_files', f'{report_name}_{item_name}.txt')

            if not os.path.exists(item_pathname):
                item_pathname = os.path.join(self.items_path, f'{report_name}_{item_name}.txt')

            if os.path.exists(item_pathname):
                with open(item_pathname, 'r', encoding='utf-8') as file:
                    item_text = file.read()
                    report_dict[item_name] = item_text
                    notnone = True
        return (idx, report_dict) if notnone else (idx, None)


def collate_fn_filter_none(batch):
    filtered = [(idx, data) for idx, data in batch if data is not None]
    if not filtered:
        return None
    indices, data = zip(*filtered)
    return list(indices), list(data)

def collate_fn_item7(batch):
    indices = []
    item7_texts = []
    
    for idx, data in batch:
        if data is not None and "item7" in data:
            item7 = data["item7"]
            if len(item7) > 200:
                indices.append(idx)
                item7_texts.append(item7)
    return indices, item7_texts

def fill_prompt_batch(texts, prompt_func,
                      tokenizer, model,
                      top_k=1, top_p=None, text_length=1000, verbose=False):
    """
    - Generates tokens for a batch of texts
    - Uses model to obtain the logits (prob to fill mask)
    - Returns the top_k tokens and associated probs
    """
    clean_mem()

    prompt_texts = [prompt_func(text[:text_length]) for text in texts]

    # Encode prompts
    inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(device)
    if verbose:
        print(inputs['input_ids'].shape)
    # Forward pass
    with torch.no_grad():
        logits = model(**inputs).logits.cpu()
    # Get probs for last token only
    probss = torch.nn.functional.softmax(logits[:,-1,:], dim=-1)

    del inputs
    gc.collect()

    if top_p is not None:
        sorted_probs, sorted_indices = torch.sort(probss, descending=True, dim=-1)    # batch by V
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)                         # batch by V
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_keep = cumulative_probs <= top_p                           # top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_keep[..., 1:] = sorted_indices_to_keep[..., :-1].clone()
        sorted_indices_to_keep[..., 0] = True
        indices_to_keep = [sorted_indices[i][sorted_indices_to_keep[i]].tolist() for i in range(len(sorted_indices))]
        probs_to_keep = [sorted_probs[i][sorted_indices_to_keep[i]].tolist()  for i in range(len(sorted_indices))]
        answers = [dict(zip(l.strip().split(),probs_to_keep[i]))
                for i,l in enumerate(tokenizer.batch_decode(indices_to_keep))]

    else:
        # top k
        top_k_tokens = [torch.topk(probss[i], top_k, dim=0) for i in range(len(probss))]
        top_decoded = [list(map(lambda s: s.strip(),
                        tokenizer.batch_decode(top_k_tokens[i].indices))) for i in range(len(top_k_tokens))]
        top_prob = [top_k_tokens[i].values.tolist() for i in range(len(top_k_tokens))]

        answers = [dict(zip(top_decoded[i], top_prob[i])) for i in range(len(top_decoded))]

    return answers


def apply_strategy(texts, strategy, tokenizer, model,
                   text_length=5000, verbose=False):
    '''
    Applies prompt strategy

    - text: to be appended the prompt
    - strategy: contains the prompt and the verbalizer
    '''

    output = fill_prompt_batch(texts, strategy.prompt, tokenizer=tokenizer,
                               model=model, top_p=strategy.top_p, text_length=text_length)
    clean_mem()

    scores = []
    for item in output:
        score = dict()
        for cat, vals in strategy.verbalizer.items():
            # Strip spaces + turn lowercase, then match with verbalizer
            score[cat] = sum([v for k,v in item.items() if k.strip().lower() in vals])
            try:
                score['polarity'] = score['positive'] - score['negative']
            except Exception:
                pass
        scores.append(score)

    if verbose:
        print(scores)
    return scores

def gather_stats(strategy, results, tokenizer, model, data, verbose=False, text_length=5000, 
                save_path="results", save_interval=5000, resume=True, max_retries=3):
    
    if resume and results:
        max_processed_idx = max([df.index.max() for df in results if not df.empty])
        print(f"Resuming from index: {max_processed_idx}")
    else:
        max_processed_idx = -1
        print("Starting from beginning")
    
    processed_count = 0
    batch_count = 0
    
    for indices, batch in notebook_tqdm(data):
        if len(indices) > 0:
            if resume and max(indices) <= max_processed_idx:
                continue
            else:
                success = False
                retry_count = 0

                while not success and retry_count < max_retries:
                    try:
                        current_text_length = text_length // (2 ** retry_count) if retry_count > 0 else text_length

                        if retry_count > 0:
                            print(f"Retry {retry_count} for indices {indices} with text_length={current_text_length}")

                        ans = apply_strategy(batch, strategy=strategy, tokenizer=tokenizer,
                                            model=model, text_length=current_text_length)
                        clean_mem()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        df = pd.DataFrame(ans, index=indices)
                        results.append(df)
                        processed_count += len(indices)
                        batch_count += 1

                        if verbose:
                            print(f"Processed indices: {indices}")
                            print(f"Total processed: {processed_count}")

                        success = True

                    except torch.cuda.OutOfMemoryError as oom_error:
                        retry_count += 1
                        print(f"CUDA OOM error processing batch {indices} (attempt {retry_count}): {oom_error}")

                        clean_mem()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        if retry_count >= max_retries:
                            print(f"Failed to process batch {indices} after {max_retries} attempts")
                            
                            error_file = os.path.join(save_path, f"error_batch_{indices[0]}_{indices[-1]}.txt")
                            with open(error_file, 'w') as f:
                                f.write(f"Batch indices: {indices}\n")
                                f.write(f"Error: CUDA Out of Memory after {max_retries} retries\n")
                                f.write(f"Original text length: {text_length}\n")
                                f.write(f"Final attempted text length: {current_text_length}\n")
                            break

                    except Exception as e:
                        print(f"Error processing batch with indices {indices}: {e}")
                        error_file = os.path.join(save_path, f"error_batch_{indices[0]}_{indices[-1]}.txt")
                        with open(error_file, 'w') as f:
                            f.write(f"Batch indices: {indices}\n")
                            f.write(f"Error: {str(e)}\n")
                        break

                if processed_count >= save_interval:
                    try:
                        stats_df = pd.concat(results, axis=0)
                        stats_df.reset_index(inplace=True)

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        csv_file = os.path.join(save_path, f"results_{timestamp}_{processed_count}_reports.csv")
                        stats_df.to_csv(csv_file, index=False)
                        print(f"Saved CSV: {csv_file}")

                        processed_count = 0

                    except Exception as e:
                        print(f"Error saving intermediate results: {e}")

        else:
            continue
    
    if results:
        try:
            stats_df = pd.concat(results, axis=0)
            stats_df.reset_index(inplace=True)
            
            final_csv_file = os.path.join(save_path, "final_results.csv")
            stats_df.to_csv(final_csv_file, index=False)
            print(f"Saved final CSV: {final_csv_file}")
            
            if verbose:
                print(stats_df.info())
            
            return stats_df
        
        except Exception as e:
            print(f"Error saving final results: {e}")
            for i, df in enumerate(results):
                try:
                    backup_file = os.path.join(save_path, f"backup_result_{i}.pkl")
                    df.to_pickle(backup_file)
                except:
                    pass
            return pd.DataFrame()
    else:
        print("No results to process")
        return pd.DataFrame()

def get_model_ouput(prompt: str, model=model) -> dict[str, float]:
    input = tokenizer(prompt, return_tensors='pt', padding=True).to(device)
    # print(input)
    print(f"input has {input['input_ids'].shape} tokens")

    with torch.no_grad():
        logits = model(**input).logits.cpu()

    logits = logits[0,-1,:]
    probs = torch.nn.functional.softmax(logits, dim=-1)
    top_k_tokens = torch.topk(probs, 20, dim=0)
    top_ix = top_k_tokens.indices.tolist()
    top_prob = top_k_tokens.values.tolist()

    # return dict(zip(top_ix, top_prob))
    return dict(zip([tokenizer.decode([x]) for x in top_ix], top_prob))