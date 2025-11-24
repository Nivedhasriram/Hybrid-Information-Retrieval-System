# src/eval.py
import numpy as np
import math

def precision_at_k(retrieved_ids, relevant_set, k):
    retrieved_k = retrieved_ids[:k]
    rel = sum(1 for d in retrieved_k if d in relevant_set)
    return rel / k

def average_precision(retrieved_ids, relevant_set):
    hits = 0
    sum_prec = 0.0
    for i, d in enumerate(retrieved_ids, start=1):
        if d in relevant_set:
            hits += 1
            sum_prec += hits / i
    if hits == 0:
        return 0.0
    return sum_prec / len(relevant_set)

def dcg_at_k(retrieved_ids, relevant_set, k):
    dcg = 0.0
    for i, d in enumerate(retrieved_ids[:k], start=1):
        rel_i = 1.0 if d in relevant_set else 0.0
        if i == 1:
            dcg += rel_i
        else:
            dcg += rel_i / math.log2(i)
    return dcg

def ndcg_at_k(retrieved_ids, relevant_set, k):
    dcg = dcg_at_k(retrieved_ids, relevant_set, k)
    ideal = sorted([1]*len(relevant_set) + [0]*100, reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal, start=1):
        if i == 1:
            idcg += rel
        else:
            idcg += rel / math.log2(i)
    return dcg / idcg if idcg > 0 else 0.0
