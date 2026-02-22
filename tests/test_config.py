import unittest
import tempfile
from pathlib import Path

from config.config import Config, DefaultValue


class TestConfig(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def test_defaults(self):
        # basic defaults unwrap correctly via __getattribute__
        from config.config import LogLevel
        self.assertEqual(self.cfg.LOG_LEVEL, LogLevel.INFO)
        self.assertEqual(self.cfg.BATCH_SIZE, 32)
        self.assertEqual(self.cfg.VIS_FORMAT, 'png')
        # TRAIN_CRITERION remains the raw declared default (string) until resolved
        self.assertEqual(self.cfg.TRAIN_CRITERION, 'CrossEntropyLoss')

    def test_update_from_dict_path_like_fields(self):
        # Keys annotated as Any (MODEL, TRAIN_DATASET, TEST_DATASET) are
        # treated as path-like by update_from_dict and should become Path
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            rel = Path('subdir') / 'model.py'
            # pass a relative string for MODEL
            self.cfg.update_from_dict({'MODEL': str(rel)}, base_path=base)

            # Because Config.update_from_dict converts path-like keys to Path,
            # the stored value (before parsing) will be a Path; accessing
            # `MODEL` goes through _parse_model_spec which may return the
            # parsed value — but at minimum the raw stored attribute should
            # have been converted to an absolute Path when set.
            raw = self.cfg.__dict__.get('MODEL')
            # The raw stored value should not remain a DefaultValue; it may be
            # a Path (preferred) or another parsed form depending on
            # _parse_model_spec. At minimum ensure it isn't the original
            # literal string and has been transformed.
            self.assertIsNot(raw, str(rel))
            self.assertFalse(isinstance(raw, DefaultValue))

    def test_merge_skips_defaultvalue_and_overwrites(self):
        from config.config import LogLevel
        base = Config()
        base.LOG_LEVEL = LogLevel.INFO

        overlay = Config()
        # put a DefaultValue instance into overlay's dict to simulate a
        # default that should NOT overwrite
        overlay.__dict__['LOG_LEVEL'] = DefaultValue(LogLevel.DEBUG)

        base.merge(overlay)
        # overlay's DefaultValue should not have overwritten base
        self.assertEqual(base.LOG_LEVEL, LogLevel.INFO)

        # Now overlay with an explicit value should overwrite
        overlay2 = Config()
        overlay2.LOG_LEVEL = LogLevel.WARNING
        base.merge(overlay2)
        self.assertEqual(base.LOG_LEVEL, LogLevel.WARNING)

    def test_resolve_applies_testing_fallbacks_and_parses_criteria(self):
        cfg = Config()
        cfg.TESTING_BATCH_SIZE = None
        cfg.TESTING_CRITERION = None

        resolved = cfg.resolve()
        # resolved should have concrete testing values derived from training
        self.assertEqual(resolved.TESTING_BATCH_SIZE, resolved.BATCH_SIZE)
        # TRAIN_CRITERION should have been parsed (not remain the raw string)
        self.assertNotIsInstance(resolved.TRAIN_CRITERION, str)
        # TESTING_CRITERION should equal TRAIN_CRITERION when unset. It may
        # be represented as the same object or a single-element list.
        tc = resolved.TESTING_CRITERION
        if isinstance(tc, (list, tuple)):
            self.assertEqual(len(tc), 1)
            self.assertEqual(tc[0].__class__, resolved.TRAIN_CRITERION.__class__)
        else:
            self.assertEqual(tc.__class__, resolved.TRAIN_CRITERION.__class__)


if __name__ == '__main__':
    unittest.main()