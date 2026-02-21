import unittest
import tempfile
from pathlib import Path
import os

from engine.loader import load_from_pyscript
from config.py_loader import PYLoader
from config.manager import ConfigManager
from config.config import Config

class TestLoader(unittest.TestCase):
    def test_load_from_pyscript_resolves_cwd(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            script = td / 'script.py'
            # script will create a file relative to its directory during import
            script.write_text("""
from pathlib import Path
p = Path('created.txt')
p.write_text('hello')
CREATED = str(p)
""")
            res = load_from_pyscript(str(script), 'CREATED')
            # file should exist in the script's directory
            created = td / 'created.txt'
            self.assertTrue(created.exists())
            self.assertEqual(created.read_text(), 'hello')

    def test_py_loader_returns_uppercase(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            script = td / 'cfg.py'
            script.write_text("""
FOO = 123
_bar = 5
""")
            loader = PYLoader()
            out = loader.load(str(script))
            self.assertIn('FOO', out)
            self.assertNotIn('_bar', out)

class TestManager(unittest.TestCase):
    def test_load_config_tree_merge_behavior(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            a = td / 'a.json'
            b = td / 'b.json'
            a.write_text('{"CONFIG": "b.json", "BATCH_SIZE": 100}')
            b.write_text('{"BATCH_SIZE": 50, "FINAL_OUTPUT_PATH": "out.pt"}')

            mgr = ConfigManager()
            cfg = mgr.load_config_tree([str(a)])
            # parent (a.json) should override child b.json according to manager logic
            self.assertEqual(cfg.BATCH_SIZE, 100)
            # path was resolved relative to file b.json's directory
            self.assertTrue(str(cfg.FINAL_OUTPUT_PATH).endswith('out.pt'))

if __name__ == '__main__':
    unittest.main()
