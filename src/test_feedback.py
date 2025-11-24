# run interactive
from .data_loader import load_cranfield
from .feedback import apply_feedback_for_query
_, queries, qrels = load_cranfield()
q = queries.iloc[0]['query']
clicked = qrels[queries.iloc[0]['qid']]  
res = apply_feedback_for_query(q, clicked, topk=5)
print(res[['id','title','score','tf_score','sem_score']])
