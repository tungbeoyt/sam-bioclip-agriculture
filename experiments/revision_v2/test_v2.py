import unittest
from audit_groups import components
from experiment_v2 import variants
from test_protocol import ProtocolChecks

class RevisionChecks(unittest.TestCase):
    def test_transitive_object_groups(self):
        groups=components([{'path':str(i)} for i in range(5)],[(0,1),(1,3)])
        self.assertEqual(sorted(sorted(r['path'] for r in g) for g in groups),[['0','1','3'],['2'],['4']])

    def test_required_experiments(self):
        self.assertEqual({v['method'] for v in variants('core')},{'linear','lora','full','resnet50','mobilenetv2'})
        self.assertEqual({v['shots'] for v in variants('fewshot')},{1,2,4,8,16})
        ab=variants('ablation')
        self.assertEqual({v['placement'] for v in ab},{'mlp','attention','both'})
        self.assertEqual({v['rank'] for v in ab},{4,8,16})
        self.assertEqual(len({tuple(sorted(v.items())) for v in ab}),len(ab))

if __name__=='__main__':unittest.main()
