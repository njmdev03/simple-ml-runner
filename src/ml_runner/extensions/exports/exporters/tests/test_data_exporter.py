import tempfile
import json
from pathlib import Path

from ml_runner.extensions.exports.exporters.data import DataExporter, DataExportConfig


def write_sample(path: Path, content: dict):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(content, f)


def test_data_exporter_csv_roundtrip(tmp_path):
    # create sample metadata files
    md_dir = tmp_path / 'md'
    md_dir.mkdir()
    write_sample(md_dir / 'model_epoch_1.json', {
        'epoch': 1,
        'train_loss': 0.27,
        'eval_metrics': {'loss': 0.13, 'accuracy': 0.95},
        'timing': {'epoch_duration': 7.9}
    })
    write_sample(md_dir / 'model_epoch_2.json', {
        'epoch': 2,
        'train_loss': 0.2,
        'eval_metrics': {'loss': 0.1, 'accuracy': 0.96},
        'timing': {'epoch_duration': 8.1}
    })

    out_dir = tmp_path / 'out'
    out_dir.mkdir()

    cfg = DataExportConfig(formats=['csv'], output=str(out_dir), metadata_dir=str(md_dir))
    exporter = DataExporter()
    exporter.run(cfg)

    out_file = out_dir / 'metrics.csv'
    assert out_file.exists()
    text = out_file.read_text()
    assert 'epoch' in text
    assert 'accuracy' in text