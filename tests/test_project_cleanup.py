"""Release staging must never accidentally include model/data/outputs binaries."""
import tempfile
import unittest
from pathlib import Path

from scripts.project_cleanup import classify, stage, audit


class CleanupTests(unittest.TestCase):
    def test_classification_blocks_model_and_output(self):
        self.assertEqual(classify(Path('models/a.pt'), 1)[1],False)
        self.assertEqual(classify(Path('outputs/demo/movie.mp4'), 1)[1],False)
        self.assertEqual(classify(Path('data/a.jpg'), 1)[1],False)
        self.assertEqual(classify(Path('src/x.py'), 1)[1],True)
        self.assertEqual(classify(Path('.env'), 1)[1],False)
        self.assertEqual(classify(Path('docs/evidence/curated/e.json'), 1)[1],True)

    def test_stage_copies_only_source_and_never_deletes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for fn, body in {'src/a.py':'print(1)', 'tests/test_a.py':'pass',
                             'models/a.pt':'binary', 'data/cam1.mp4':'video',
                             'outputs/evaluation/raw.json':'raw', '.env':'top secret',
                             'docs/SHOWCASE.md':'hi', '.gitignore':'/data/**\n',
                             'outputs/rtsp/events.jsonl':'person id'} .items():
                p=root/fn;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body)
            destination=root/'outputs/release_candidate'
            stage(root,destination)
            self.assertTrue((destination/'src/a.py').exists())
            self.assertTrue((destination/'docs/SHOWCASE.md').exists())
            self.assertFalse((destination/'models/a.pt').exists())
            self.assertFalse((destination/'data/cam1.mp4').exists())
            self.assertFalse((destination/'outputs/evaluation/raw.json').exists())
            self.assertFalse((destination/'.env').exists())
            self.assertTrue((root/'data/cam1.mp4').exists())
            self.assertTrue((root/'outputs/rtsp/events.jsonl').exists())
            with self.assertRaises(FileExistsError):stage(root,destination)

    def test_audit_never_changes_existing_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            p=root/'models/weights.pt';p.parent.mkdir();p.write_bytes(b'checkpoint')
            audit(root,'outputs/audit.json')
            self.assertEqual(p.read_bytes(),b'checkpoint')
            self.assertTrue((root/'outputs/audit.json').exists())
            with self.assertRaises(FileExistsError):audit(root,'outputs/audit.json')


if __name__=='__main__':unittest.main()
