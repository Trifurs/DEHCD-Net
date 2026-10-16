import tempfile
import unittest
from pathlib import Path

from scripts.check_source import source_files, validate_source


class PublicSourceTests(unittest.TestCase):
    def test_local_artifacts_are_excluded_and_model_resources_and_licenses_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            names=['README.md','local/settings.json','.git/config','runs/test/result.json',
                   'models/weights.pth','models/__pycache__/x.pyc','models/network.py',
                   'compare/official/model/LICENSE','compare/official/model/data/filter.npz']
            for name in names:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(b'x=1\n' if path.suffix=='.py' else b'example')
            files=source_files(root)
            expected={'README.md','models/network.py','compare/official/model/LICENSE',
                      'compare/official/model/data/filter.npz'}
            self.assertEqual(expected,{p.as_posix() for p in files})
            validate_source(root,files)

    def test_private_paths_are_reported_without_printing_their_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'README.md'
            path.write_text('/'+'home'+'/somebody/private/data')
            with self.assertRaisesRegex(ValueError,'Workstation path in public source: README.md'):
                validate_source(root,source_files(root))


if __name__=='__main__':unittest.main()
