from pathlib import Path
from collections import defaultdict
import json,sys,tempfile,unittest
from unittest.mock import patch
import numpy as np
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import data_protocol as data
from evaluate import calibration,predict_references

class EvaluationTests(unittest.TestCase):
    def test_reference_batch_order_and_baseline_fallback(self):
        banks=[{'id':4},{'id':2},{'id':9}];query=object()
        class Batched:
            def predict_many(self,received,encoded):
                self.assertions=(received is banks,encoded is query)
                return [[{'score':bank['id']}]for bank in received]
            def predict(self,*args):raise AssertionError('Sequential branch should not execute')
        class Sequential:
            def predict(self,bank,encoded):
                if encoded is not query:raise AssertionError('Query must be reused unchanged')
                return [{'score':bank['id']}]
        batched=Batched();expected=[[{'score':4}],[{'score':2}],[{'score':9}]]
        self.assertEqual(predict_references(batched,banks,query),expected)
        self.assertEqual(batched.assertions,(True,True))
        self.assertEqual(predict_references(Sequential(),banks,query),expected)
        class Broken:
            def predict_many(self,banks,encoded):return []
        with self.assertRaisesRegex(ValueError,'target count'):predict_references(Broken(),banks,query)
    def test_fresh_roster_has_no_prior_ids_or_crowds(self):
        plan=data.load_plan();test=plan['test'];ids={r['image_id']for r in test}
        self.assertEqual(len(ids),576);self.assertFalse(ids&set(plan['prior_excluded_image_ids']))
        self.assertEqual(len(plan['prior_excluded_image_ids']),3648)
        self.assertEqual(sum(bool(b)for r in test for b in r['boxes_by_target'].values()),483)
        self.assertTrue(all(all(r['valid_by_target'].values())for r in test))
    def test_selection_order_independent_and_global_unique(self):
        ims={i:{'width':100,'height':100}for i in range(30)};anns=defaultdict(list);targets=[{'category_id':1},{'category_id':2}]
        for i in range(20):anns[i]=[{'category_id':i%2+1,'bbox':[0,0,20,20],'iscrowd':0}]
        anns[3].append({'category_id':1,'bbox':[0,0,10,10],'iscrowd':1})
        a,_=data.choose(ims,anns,targets,{2},2,2);b,_=data.choose(dict(reversed(list(ims.items()))),anns,targets,{2},2,2)
        self.assertEqual(a,b);self.assertEqual(len({r[0]for r in a}),8);self.assertFalse({2,3}&{r[0]for r in a})
    def test_stale_freeze_rejected_without_network(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);plan=root/'plan.json';plan.write_text('{}');source=root/'code.py';source.write_text('original')
            protocol=root/'freeze.json';value={'model_selection_complete':True,'test_scores_computed_before_freeze':False,'data_plan_sha256':data.sha(plan),
              'methods':{m:{}for m in data.METHODS},'behavior_files':{str(source):data.sha(source)},'checkpoint_files':{}}
            protocol.write_text(json.dumps(value));expected=data.sha(protocol)
            with patch.object(data,'PLAN',plan),patch.object(data.urllib.request,'urlopen')as network:
                with self.assertRaisesRegex(ValueError,'hash'):data.verify_protocol(protocol,'0'*64)
                source.write_text('mutated')
                with self.assertRaisesRegex(ValueError,'Bound file'):data.verify_protocol(protocol,expected)
                network.assert_not_called()
    def test_two_operating_points_are_dev_fitted_and_conservative(self):
        rows=[{'boxes_by_target':{'1':[]},'valid_by_target':{'1':True}}for _ in range(100)]
        rows.append({'boxes_by_target':{'1':[[0,0,1,1]]},'valid_by_target':{'1':True}})
        scores=np.arange(101,dtype=float)[:,None];overlaps=np.ones_like(scores);result=calibration(scores,overlaps,rows,[{'category_id':1}])
        self.assertEqual(set(result),{'primary_5pct','secondary_1pct'})
        self.assertEqual(result['primary_5pct']['summary']['false_presence'],5)
        self.assertEqual(result['secondary_1pct']['summary']['false_presence'],1)

if __name__=='__main__':unittest.main()
