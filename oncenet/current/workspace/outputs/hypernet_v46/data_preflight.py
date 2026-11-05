"""Run the independent source/input audit before any real neural inference."""
from common import *

def main():
 assert not (R/'evaluation/protocol.json').exists()
 assert not (R/'evaluation/results.json').exists()
 assert not (R/'evaluation_data_preflight.json').exists()
 rows=read(R/'data/records.json')
 audit=module('v46_independent_preflight',R/'evaluation_audit.py')
 review=R/'data/collision_review.json'
 stub=dict(row_count=len(rows),split_counts={k:sum(r['split']==k for r in rows) for k in sorted({r['split'] for r in rows})},files={str(review):sha(review)})
 checked,evidence=audit.data_audit(stub)
 assert len(checked)==len(rows)
 files=[Path(__file__),R/'evaluation_audit.py',R/'data/audit.json',R/'data/collision_review.json',R/'data/records.json',R/'data/inputs_binding.json',R/'data/input_pixel_hashes.json',R/'data/acquisition_audit.json',R/'data/plan.json']
 save(R/'evaluation_data_preflight.json',dict(passed=True,no_model_scores=True,checked_epoch=time.time(),data=evidence,files={str(p):sha(p) for p in files}))
 print('V46 INDEPENDENT DATA PREFLIGHT PASSED',flush=True)

if __name__=='__main__':main()
