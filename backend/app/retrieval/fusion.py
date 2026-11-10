def rrf(lists, constant=60):
    scores, items = {}, {}
    for rows in lists:
        seen = set()
        for rank, row in enumerate(rows, 1):
            key = row['chunk_id']
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0) + 1 / (constant + rank)
            items[key] = row
    return [{**items[k], 'score': scores[k]} for k in sorted(scores, key=lambda k: (-scores[k], k))]


