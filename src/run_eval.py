# src/run_eval.py
from data_loader import load_cranfield
from retrieve import tfidf_search
from eval import precision_at_k, ndcg_at_k, average_precision

def eval_all(topk=10):
    corpus, queries, qrels = load_cranfield()
    p_list, ndcg_list, ap_list = [], [], []
    for _, row in queries.iterrows():
        qid = row['qid']
        qtxt = row['query']
        res = tfidf_search(qtxt, topk=topk)
        retrieved_ids = res['id'].tolist()
        relevant = set(qrels.get(qid, []))
        p = precision_at_k(retrieved_ids, relevant, k=topk)
        ndcg = ndcg_at_k(retrieved_ids, relevant, k=topk)
        ap = average_precision(retrieved_ids, relevant)
        p_list.append(p); ndcg_list.append(ndcg); ap_list.append(ap)
        print(qid, p, ndcg, ap)
    print("Mean P@k", sum(p_list)/len(p_list))
    print("Mean nDCG@k", sum(ndcg_list)/len(ndcg_list))
    print("MAP", sum(ap_list)/len(ap_list))

if __name__=="__main__":
    eval_all(10)
