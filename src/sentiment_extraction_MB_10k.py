import gc
import os
import sys
from dataclasses import dataclass
# from google.colab import userdata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pprint import pprint
from typing import Callable
from datetime import datetime

import random
# from tqdm import tqdm
import time
from tqdm.autonotebook import tqdm as notebook_tqdm
import torch
from torch.utils.data import Dataset, DataLoader
torch.set_float32_matmul_precision('high')

from huggingface_hub import login
import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoConfig, logging, AutoModelForMaskedLM

from langchain_text_splitters import RecursiveCharacterTextSplitter

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

def collate_fn_item7(batch):
    indices = []
    item7_texts = []
    
    for idx, data in batch:
        if data is not None and "item7" in data:
            item7 = data["item7"]
#             if len(item7) > 200:
            indices.append(idx)
            item7_texts.append(item7)
    return indices, item7_texts

def fill_prompt_batch(texts, ending, prompt_func, tokenizer, model,
                      top_k=1, top_p=None, text_length=1000, verbose=False, device='cuda'):
    """
    - Generates tokens for a batch of texts
    - Uses model to obtain the logits (prob to fill mask)
    - Returns the top_k tokens and associated probs
    """
    clean_mem()

    prompt_texts = [prompt_func(text[:text_length], ending) for text in texts]
    
    inputs = tokenizer(prompt_texts, return_tensors="pt", truncation=True,
                       padding='max_length').to(device)
    
    if verbose:
        print(inputs['input_ids'].shape)

    with torch.no_grad():
        logits = model(**inputs).logits.cpu()

#     probss = torch.nn.functional.softmax(logits[:,-1,:], dim=-1)
    pad_ix = (inputs["input_ids"]==tokenizer.mask_token_id).cpu()    # batch_size, vocab_size
    answer_logits = logits[pad_ix]
    answer_probs = torch.nn.functional.softmax(answer_logits, dim=-1)

    del inputs
    gc.collect()
    
    if top_p is not None and top_p>0 and top_p<=1.0:
        sorted_probs, sorted_indices = torch.sort(answer_probs, descending=True, dim=-1)    # batch_size by vocab_size
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_keep = cumulative_probs <= top_p                           # top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_keep[..., 1:] = sorted_indices_to_keep[..., :-1].clone()
        sorted_indices_to_keep[..., 0] = True
        indices_to_keep = [sorted_indices[i][sorted_indices_to_keep[i]].tolist() for i in range(len(sorted_indices))]
        probs_to_keep = [sorted_probs[i][sorted_indices_to_keep[i]].tolist()  for i in range(len(sorted_indices))]
        answers = [dict(zip(l.strip().split(),probs_to_keep[i]))
                  for i,l in enumerate(tokenizer.batch_decode(indices_to_keep))]
        return answers
    
    else:
        top_k_tokens = torch.topk(answer_probs, top_k, dim=1).indices.tolist()
        top_k_probs = torch.topk(answer_probs, top_k, dim=1).values.tolist()
        top_k_decoded = tokenizer.batch_decode(top_k_tokens)
        return [dict(zip(d.strip().split(" "), top_k_probs[i])) for i,d in enumerate(top_k_decoded)]
    
def apply_strategy(texts, ending, strategy, tokenizer, model, text_length=5000, verbose=False, device='cuda'):
    '''
    Applies prompt strategy

    - text: to be appended the prompt
    - strategy: contains the prompt and the verbalizer
    '''

    output = fill_prompt_batch(texts, ending, strategy.prompt, tokenizer=tokenizer,
                               model=model, top_p=strategy.top_p, text_length=text_length, device=device)
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

def get_model_output(prompt, model, device, k=20) -> dict[str, float]:

    inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                     padding='max_length').to(device)

    print(f"input has {inputs['input_ids'].shape} tokens")

    with torch.no_grad():
        logits = model(**inputs).logits.cpu()

    pad_ix = (inputs["input_ids"]==tokenizer.mask_token_id).cpu()    # batch_size, vocab_size
    answer_logits = logits[pad_ix]
    answer_probs = torch.nn.functional.softmax(answer_logits, dim=-1)
    
    top_k_tokens = torch.topk(answer_probs, k, dim=1)
    top_ix = top_k_tokens.indices.tolist()[0]
    top_prob = top_k_tokens.values.tolist()[0]

    return dict(zip([tokenizer.decode([x]) for x in top_ix], top_prob))

def gather_stats(strategy, results, tokenizer, model, data, ending, verbose=False, text_length=5000, 
                save_path="results", save_interval=5000, resume=True, max_retries=3, device='cuda'):
    
    if resume and results:
        max_processed_idx = max([df.index.max() for df in results if not df.empty])
        print(f"Resuming from index: {max_processed_idx}")
    else:
        max_processed_idx = -1
    
    processed_count = 0
    batch_count = 0
    
    for doc_indices, batch_chunks, chunks_per_doc in notebook_tqdm(data):
        if len(doc_indices) > 0:
            if resume and max(doc_indices) <= max_processed_idx:
                continue
            else:
                success = False
                retry_count = 0

                while not success and retry_count < max_retries:
                    try:
                        chunk_scores = apply_strategy(batch_chunks, ending, strategy=strategy, 
                                                     tokenizer=tokenizer, model=model, 
                                                     text_length=text_length, device=device)
                        clean_mem()
                        
                        # Агрегируем результаты по документам
                        doc_scores = []
                        start_idx = 0
                        for doc_idx, num_chunks in zip(np.unique(doc_indices), chunks_per_doc):
                            if num_chunks == 0:
                                continue
                            
                            # Получаем скоры для всех чанков этого документа
                            doc_chunk_scores = chunk_scores[start_idx:start_idx + num_chunks]
                            start_idx += num_chunks
                            
                            # Агрегируем скоры (например, среднее по всем чанкам)
                            aggregated_score = {}
                            for key in doc_chunk_scores[0].keys():
                                values = [score[key] for score in doc_chunk_scores if key in score]
                                aggregated_score[key] = sum(values) / len(values) if values else 0
                            
                            doc_scores.append((doc_idx, aggregated_score))
                        
                        # Сохраняем результаты
                        indices = [idx for idx, _ in doc_scores]
                        scores = [score for _, score in doc_scores]
                        
                        df = pd.DataFrame(scores, index=indices)
                        results.append(df)
                        processed_count += len(indices)
                        batch_count += 1

                        success = True

                    except torch.cuda.OutOfMemoryError as oom_error:
                        retry_count += 1
                        print(f"CUDA OOM error processing batch {doc_indices} (attempt {retry_count}): {oom_error}")
                        
                        clean_mem()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        if retry_count >= max_retries:
                            print(f"Failed to process batch after {max_retries} attempts")
                            break
                    
                    except Exception as e:
#                         print(f"Error processing batch with indices {doc_indices}: {e}")
                        break
                
                # Сохранение промежуточных результатов
                if processed_count >= save_interval:
                    try:
                        stats_df = pd.concat(results, axis=0)
                        stats_df.reset_index(inplace=True)
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        csv_file = os.path.join(save_path, f"results_{timestamp}_{processed_count}_reports.csv")
                        stats_df.to_csv(csv_file, index=False)
                        
                        processed_count = 0
                        
                    except Exception as e:
                        print(f"Error saving intermediate results: {e}")
        
        else:
            continue
    
    # Финальное сохранение
    if results:
        try:
            stats_df = pd.concat(results, axis=0)
            stats_df.reset_index(inplace=True)
            
            final_csv_file = os.path.join(save_path, "final_results.csv")
            stats_df.to_csv(final_csv_file, index=False)
            
            if verbose:
                print(stats_df.info())
            
            return stats_df
        
        except Exception as e:
            print(f"Error saving final results: {e}")
            return pd.DataFrame()
    else:
        print("No results to process")
        return pd.DataFrame()

def split_collator(batch):
    all_chunks = []
    doc_indices = []
    chunks_per_doc = []
    
    for idx, data in batch:
        if data is not None and "item7" in data:
            text = data['item7']
            chunks = splitter.split_text(text)
            all_chunks.extend(chunks)
            doc_indices.extend([idx] * len(chunks))
            chunks_per_doc.append(len(chunks))
    
    return doc_indices, all_chunks, chunks_per_doc



