import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.defaults import DEFAULT_SEEDS
from tools.run_multiseed import make_plan, parse_args


class LaunchDefaultsTests(unittest.TestCase):
    def test_batch_entry_points_share_defaults_and_allow_custom_repetitions(self):
        from tools import run_event_transfer, run_transfer_study, run_all
        self.assertEqual(5,len(DEFAULT_SEEDS))
        self.assertEqual(len(DEFAULT_SEEDS),len(set(DEFAULT_SEEDS)))
        with patch('sys.argv',['run_multiseed.py','--output','/tmp/batch-plan']):
            args=parse_args()
        self.assertEqual(list(DEFAULT_SEEDS),args.seeds)
        for runner in [run_event_transfer,run_transfer_study]:
            with patch('sys.argv',['runner']):self.assertEqual(list(DEFAULT_SEEDS),runner.parse_args().seeds)
        seen=[]
        with patch('sys.argv',['run_all.py','--dry-run']),patch('tools.run_multiseed.main',side_effect=lambda:seen.extend(__import__('sys').argv)):
            run_all.main()
        start=seen.index('--seeds')+1
        self.assertEqual(list(map(str,DEFAULT_SEEDS)),seen[start:start+len(DEFAULT_SEEDS)])
        args.experiments=['bright_dehcd_s']
        plan=make_plan(args)
        self.assertEqual(list(DEFAULT_SEEDS),[j['seed'] for j in plan['jobs']])
        args.seeds=[11,29]
        custom=make_plan(args)
        self.assertEqual(args.seeds,[j['seed'] for j in custom['jobs']])
        for seeds in [[],[11,11],[-1],[2**32]]:
            args.seeds=seeds
            with self.assertRaises(ValueError):make_plan(args)

    def test_reports_derive_completeness_and_statistics_from_requested_seeds(self):
        from tools.compare_results import make_report
        seeds=[11,13,17,19,23]
        jobs=[{'id':f'{name}_{seed}','experiment':name,'seed':seed,'dataset':'example',
               'config':{'experiment':{'id':name,'comparison':{'reference':'reference' if name=='candidate' else ''}}}}
              for name in ['reference','candidate'] for seed in seeds]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'summary').mkdir()
            (root/'protocol.json').write_text(json.dumps({'layout_version':2,'seeds':seeds,'jobs':jobs}))
            def write_rows(missing=False):
                with (root/'summary/per_seed.csv').open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=['experiment','seed','foreground_miou','checkpoint','config_sha256'])
                    writer.writeheader()
                    for name,value in [('reference',.8),('candidate',.7)]:
                        for seed in seeds:
                            if missing and name=='candidate' and seed==seeds[-1]:continue
                            writer.writerow(dict(experiment=name,seed=seed,foreground_miou=value,checkpoint='weights',config_sha256='config'))
            with patch('tools.compare_results.aggregate',return_value={}),patch('tools.compare_results.pair_policy',return_value={}),patch('tools.compare_results.comparability',return_value={}):
                write_rows()
                report=make_report(root,metrics=['foreground_miou'],repeats=1000)
                self.assertTrue(all(r['complete'] and r['expected_n']==len(seeds) for r in report['tables']))
                self.assertAlmostEqual(.0625,report['comparisons'][0]['p_value'])
                write_rows(missing=True)
                report=make_report(root,metrics=['foreground_miou'],repeats=1000)
                self.assertEqual('incomplete',report['status'])
                self.assertIsNone(report['comparisons'][0]['p_value'])
                from tools.compare_results import main
                export=root/'report'
                with patch('sys.argv',['compare_results.py','--campaign',str(root),'--output',str(export)]),patch('tools.compare_results.make_report',return_value=report),patch('builtins.print'):
                    main()
                markdown=(export/'raw_metrics.md').read_text()
                self.assertIn('n/expected',markdown)
                self.assertIn('| 4/5 |',markdown)
                self.assertIn('| 5/5 |',markdown)


if __name__=='__main__':unittest.main()
