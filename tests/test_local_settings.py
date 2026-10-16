from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.local_settings import launch_settings, local_source_reviews
from utils.protocol import digest
from utils.campaign_reuse import source_changes, source_review_for


class LocalSettingsTests(unittest.TestCase):
    def test_public_defaults_are_portable_and_local_env_precedence_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            root=Path(tmp)
            defaults=launch_settings(root)
            self.assertEqual(str(root/'data'), defaults['data_base'])
            self.assertEqual('none', defaults['runtime_profile'])
            self.assertEqual('auto', defaults['device'])
            (root/'local').mkdir()
            (root/'local/settings.json').write_text(json.dumps({'output':'results','runtime_profile':'rtx5090'}))
            self.assertEqual(str(root/'results'),launch_settings(root)['output'])
            os.environ['DEHCD_OUTPUT']=str(root/'other')
            self.assertEqual(str(root/'other'),launch_settings(root)['output'])

    def test_invalid_settings_do_not_silently_change_output_or_profile(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            root=Path(tmp);(root/'local').mkdir()
            for value in [{'outpt':'results'},{'output':''},{'runtime_profile':'typo'},[]]:
                (root/'local/settings.json').write_text(json.dumps(value))
                with self.assertRaises(ValueError):launch_settings(root)

    def test_local_source_records_allow_only_the_exact_verified_transition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            self.assertEqual([],local_source_reviews(root))
            a={'files':{'tools/run_all.py':'a'}};a['sha256']=digest(a['files'])
            b={'files':{'tools/run_all.py':'b'}};b['sha256']=digest(b['files'])
            record={'from_sha256':a['sha256'],'to_sha256':b['sha256'],
                    'status':'training_compatible','changes':source_changes(a,b)}
            (root/'local').mkdir();(root/'local/source_compatibility.json').write_text(json.dumps({'source_reviews':[record]}))
            reviews=local_source_reviews(root)
            self.assertEqual(record,source_review_for(a,b,reviews))
            c={'files':{'tools/run_all.py':'b','models/dehcd_net.py':'changed'}};c['sha256']=digest(c['files'])
            with self.assertRaises(ValueError):source_review_for(a,c,reviews)

    def test_study_accepts_checked_source_change_without_rewriting_or_relaxing_scientific_contract(self):
        from tools.run_transfer_study import seal_study_plan
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'local').mkdir()
            a={'files':{'tools/run_all.py':'a'}};a['sha256']=digest(a['files'])
            b={'files':{'tools/run_all.py':'b'}};b['sha256']=digest(b['files'])
            review={'from_sha256':a['sha256'],'to_sha256':b['sha256'],
                    'status':'training_compatible','changes':source_changes(a,b)}
            (root/'local/source_compatibility.json').write_text(json.dumps({'source_reviews':[review]}))
            path=root/'study_plan.json';old={'source':a,'seeds':[42],'dataset_sha256':'data'}
            path.write_text(json.dumps(old));original=path.read_bytes()
            with patch('tools.run_transfer_study.ROOT',root):
                seal_study_plan(path,dict(old,source=b))
                self.assertEqual(original,path.read_bytes())
                with self.assertRaises(ValueError):seal_study_plan(path,dict(old,source=b,seeds=[1234]))
                with self.assertRaises(ValueError):seal_study_plan(path,dict(old,source=b,dataset_sha256='changed'))
            with patch('utils.local_settings.local_source_reviews',return_value=[review]):
                from utils.campaign_reuse import verify_source_import
                verify_source_import(root,a,b)
                c={'files':{'tools/run_all.py':'b','models/dehcd_net.py':'changed'}};c['sha256']=digest(c['files'])
                with self.assertRaises(ValueError):verify_source_import(root,a,c)


if __name__=='__main__':unittest.main()
