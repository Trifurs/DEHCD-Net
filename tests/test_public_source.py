import tempfile
import unittest
import subprocess
from pathlib import Path

from scripts.check_source import source_files, validate_source, validate_git_candidates


class PublicSourceTests(unittest.TestCase):
    def test_local_artifacts_are_excluded_and_model_resources_and_licenses_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            names=['README.md','local/settings.json','.git/config','runs/test/result.json',
                   'models/weights.pth','models/__pycache__/x.pyc','models/network.py',
                   'compare/official/model/LICENSE','compare/official/wavehfg/pytorch_wavelets/dtcwt/data/filter.npz']
            for name in names:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(b'x=1\n' if path.suffix=='.py' else b'example')
            files=source_files(root)
            expected={'README.md','models/network.py','compare/official/model/LICENSE',
                      'compare/official/wavehfg/pytorch_wavelets/dtcwt/data/filter.npz'}
            self.assertEqual(expected,{p.as_posix() for p in files})
            validate_source(root,files)

    def test_private_paths_are_reported_without_printing_their_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'README.md'
            path.write_text('/'+'home'+'/somebody/private/data')
            with self.assertRaisesRegex(ValueError,'Workstation path in public source: README.md'):
                validate_source(root,source_files(root))

    def test_git_ignores_machine_files_and_generated_tables_at_any_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            subprocess.run(['git','init','-q',str(root)],check=True)
            (root/'.gitignore').write_bytes((Path(__file__).resolve().parents[1]/'.gitignore').read_bytes())
            names=['README.md','notes.md','local/settings.json','configs/nested/results/result.json',
                   'configs/metrics.xlsx','docs/table.csv','configs/result.json','data/labels.npz',
                   'compare/official/wavehfg/pytorch_wavelets/dtcwt/data/filter.npz']
            for name in names:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('example')
            candidates=subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],cwd=root).decode().split('\0')
            expected={'.gitignore','README.md','compare/official/wavehfg/pytorch_wavelets/dtcwt/data/filter.npz'}
            self.assertEqual(expected,set(filter(None,candidates)))
            self.assertEqual(len(expected),validate_git_candidates(root,source_files(root)))

    def test_previously_tracked_private_file_is_rejected_even_when_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            subprocess.run(['git','init','-q',str(root)],check=True)
            (root/'.gitignore').write_text('/local/\n')
            path=root/'local/settings.json';path.parent.mkdir();path.write_text('{}')
            subprocess.run(['git','add','-f','local/settings.json'],cwd=root,check=True)
            with self.assertRaisesRegex(ValueError,'Non-source files in Git candidates: local/settings.json'):
                validate_git_candidates(root,source_files(root))


if __name__=='__main__':unittest.main()
