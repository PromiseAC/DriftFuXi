"""Stats for official per-user sasrec_format.csv; no execution of CSV cells."""
import argparse
import ast
import csv
import json
from pathlib import Path
import numpy as np

def sequence(cell):
    value = ast.literal_eval(cell)
    return list(value) if isinstance(value,(list,tuple)) else [value]

def summarize(path):
    # KuaiRec long-history CSV cells exceed Python CSV's 128 KiB default.
    csv.field_size_limit(1 << 30)
    lengths, items, users = [],set(),set()
    unique_pairs = 0
    with path.open() as f:
        for row in csv.DictReader(f):
            user = row['user_id']
            assert user not in users, 'Expected one row per user'
            users.add(user)
            ids = sequence(row['sequence_item_ids'])
            timestamps = sequence(row['sequence_timestamps'])
            ratings = sequence(row['sequence_ratings'])
            assert len(ids)==len(timestamps)==len(ratings) and len(ids)>0
            assert all(a<=b for a,b in zip(timestamps,timestamps[1:])), 'Unsorted timestamps'
            lengths.append(len(ids)); items.update(ids); unique_pairs += len(set(ids))
    assert users and items
    return {'file':str(path),'users':len(users),'items_observed':len(items),'interactions':sum(lengths),'unique_user_item_pairs':unique_pairs,'sequence_length':{'avg':float(np.mean(lengths)),'median':float(np.median(lengths)),'p90':float(np.percentile(lengths,90)),'p95':float(np.percentile(lengths,95)),'max':max(lengths)},'sparsity_unique_pairs':1-unique_pairs/(len(users)*len(items)),'definition':'Before truncation/splitting; sparsity = 1 - unique(user,item)/(users*observed_items)'}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('csv',type=Path); args=p.parse_args()
    print(json.dumps(summarize(args.csv),indent=2))
