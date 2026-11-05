"""CPU-only protocol tests. Does not import PyTorch or run any training."""
import copy
import json
import unittest
from pathlib import Path
from train_revision import select_support
from run_experiments import plan


class ProtocolChecks(unittest.TestCase):
    def setUp(self):
        self.manifest={'classes':[str(i) for i in range(10)],'splits':{'train':[],'val':[],'test':[]}}
        for label in range(10):
            for i in range(25):
                self.manifest['splits']['train'].append({'path':f'{label}/{i}.jpg','class':str(label),'label':label,'pixel_sha256':f'{label}-{i}'})
            for split in ['val','test']:
                self.manifest['splits'][split].append({'path':f'{split}/{label}.jpg','class':str(label),'label':label,'pixel_sha256':f'{split}-{label}'})

    def test_support_is_nested_and_never_contains_held_out(self):
        previous=set()
        for k in [1,2,4,8,16]:
            selected=select_support(self.manifest,k,42)
            rows=selected['splits']['train'];paths={r['path'] for r in rows}
            self.assertEqual(len(rows),10*k)
            self.assertTrue(previous.issubset(paths))
            self.assertEqual(selected['splits']['val'],self.manifest['splits']['val'])
            self.assertEqual(selected['splits']['test'],self.manifest['splits']['test'])
            self.assertTrue(all(sum(r['label']==c for r in rows)==k for c in range(10)))
            previous=paths

    def test_seed_and_baseline_controls(self):
        before=copy.deepcopy(self.manifest)
        self.assertEqual(select_support(self.manifest,8,17),select_support(self.manifest,8,17))
        self.assertNotEqual(select_support(self.manifest,8,17)['splits']['train'],select_support(self.manifest,8,42)['splits']['train'])
        self.assertEqual(self.manifest,before)

    def test_insufficient_class_is_rejected(self):
        self.manifest['splits']['train']=[r for r in self.manifest['splits']['train'] if r['label']!=0]
        with self.assertRaises(ValueError):select_support(self.manifest,1,42)

    def test_no_duplicate_runs_in_full_plan(self):
        all_runs=plan('all')
        self.assertEqual(len(all_runs),87)
        self.assertEqual(len(set(all_runs)),87)
        self.assertEqual(len(plan('core')),6)
        self.assertEqual({r for m,r,k,s in plan('rank')},{2,4,16})
        self.assertEqual({s for m,r,k,s in all_runs},{17,42,2026})


if __name__=='__main__':unittest.main()
