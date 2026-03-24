import unittest
from unittest.mock import MagicMock
from ml_runner.core.engine.event_manager import EventManager
from ml_runner.core.registries.callbacks import Callback, attach
from ml_runner.core.engine.engine import Engine, EngineEvent
from ml_runner.core.extensions.base_extension import BaseExtension
from ml_runner.core.config.schema import RunConfig

class TestNewEventSystem(unittest.TestCase):

    def test_manual_subscribe(self):
        em = EventManager()
        mock_handler = MagicMock()
        em.subscribe(mock_handler, "test_event")

        em.emit("test_event", data="hello")
        mock_handler.assert_called_once_with(data="hello")

    def test_decorator_registration(self):
        class MyListener:
            def __init__(self):
                self.called = False

            @Callback("test_event")
            def on_test(self, data):
                self.called = True
                self.data = data

        em = EventManager()
        listener = MyListener()
        attach(listener, em)

        em.emit("test_event", data="world")
        self.assertTrue(listener.called)
        self.assertEqual(listener.data, "world")

    def test_argument_filtering(self):
        em = EventManager()

        def handler_all(**kwargs):
            handler_all.called = True
            handler_all.kwargs = kwargs

        def handler_specific(data):
            handler_specific.called = True
            handler_specific.data = data

        em.subscribe(handler_all, "test_event")
        em.subscribe(handler_specific, "test_event")

        em.emit("test_event", data="filtered", extra="ignored")

        self.assertTrue(handler_all.called)
        self.assertEqual(handler_all.kwargs, {"data": "filtered", "extra": "ignored"})

        self.assertTrue(handler_specific.called)
        self.assertEqual(handler_specific.data, "filtered")

    def test_extension_setup(self):
        class MyExtension(BaseExtension):
            def setup(self, event_manager, global_config, config=None):
                if config and config.get("enabled"):
                    event_manager.subscribe(self.handle, "test_event")

            def handle(self, msg):
                self.msg = msg

        em = EventManager()
        ext = MyExtension()

        # Test disabled
        ext.setup(em, None, {"enabled": False})
        em.emit("test_event", msg="fail")
        self.assertFalse(hasattr(ext, "msg"))

        # Test enabled
        ext.setup(em, None, {"enabled": True})
        em.emit("test_event", msg="pass")
        self.assertEqual(ext.msg, "pass")

if __name__ == "__main__":
    unittest.main()
