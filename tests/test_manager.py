import unittest
import os
import argparse
from config.manager import ConfigManager
from config.config import Config

class TestManagerOverrides(unittest.TestCase):
    def setUp(self):
        # ensure a clean env
        self.env_keys = ['PROFILE', 'LOG_LEVEL', 'TESTING_BATCH_SIZE']
        for k in self.env_keys:
            os.environ.pop(k, None)
        self.mgr = ConfigManager()

    def tearDown(self):
        for k in self.env_keys:
            os.environ.pop(k, None)

    def test_apply_env_overrides_boolean(self):
        cfg = Config()
        self.assertFalse(cfg.PROFILE)
        os.environ['PROFILE'] = 'true'
        self.mgr.apply_env_overrides(cfg)
        self.assertTrue(cfg.PROFILE)

    def test_apply_env_overrides_nonboolean(self):
        cfg = Config()
        os.environ['TESTING_BATCH_SIZE'] = '128'
        self.mgr.apply_env_overrides(cfg)
        # numeric strings are not cast except for booleans; expect string
        self.assertEqual(cfg.TESTING_BATCH_SIZE, '128')

    def test_apply_cli_overrides_train_and_generic(self):
        cfg = Config()
        args = argparse.Namespace(
            train=True,
            dont_train=False,
            test=None,
            dont_test=None,
            test_while_training=False,
            test_on_training_data=False,
            test_checkpoints=False,
            log_level=None,
            profile=None,
            show=None,
            final_output_path='cli_out.pt',
            config=None,
        )
        cfg = self.mgr.apply_cli_overrides(cfg, args)
        self.assertTrue(cfg.TRAIN)
        self.assertEqual(cfg.FINAL_OUTPUT_PATH, 'cli_out.pt')

if __name__ == '__main__':
    unittest.main()
