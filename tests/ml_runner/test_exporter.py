from ml_runner.core.registries.exporter import ExporterRegistry


def test_data_exporter_registered():
    assert 'data' in ExporterRegistry.all()
    assert 'plot' in ExporterRegistry.all()
