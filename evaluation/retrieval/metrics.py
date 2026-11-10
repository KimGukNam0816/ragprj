def retrieval_scores(reference, contexts):
    gold={x['book_id'] for group in reference.get('evidence_groups',[]) for x in group}
    if not gold:
        return {'hit_at_4':None,'recall_at_4':None,'retrieval_status':'not_applicable'}
    found={x['book_id'] for x in contexts}
    return {'hit_at_4':int(bool(gold&found)),'recall_at_4':len(gold&found)/len(gold),'retrieval_status':'ok'}


