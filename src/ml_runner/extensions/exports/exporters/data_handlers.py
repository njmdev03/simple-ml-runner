"""Write handlers for the Data exporter.

These handlers are registered with the application's `MetadataWriter` registry
so they can be discovered by the `DataExporter` without embedding handler code
directly in the exporter.
"""
from pathlib import Path
import json
import csv

from ml_runner.core.registries.metadata_writer import MetadataWriter


@MetadataWriter('json')
def write_json(rows, cols, out_path: Path):
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(rows, fh, indent=4)


@MetadataWriter('csv')
def write_csv(rows, cols, out_path: Path):
    with open(out_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow(cols)
        for r in rows:
            writer.writerow([r.get(c, '') for c in cols])


@MetadataWriter('xlsx')
def write_xlsx(rows, cols, out_path: Path):
    try:
        from openpyxl import Workbook
    except Exception:
        return
    wb = Workbook()
    ws = wb.active
    ws.append(cols)
    for r in rows:
        ws.append([r.get(c, '') for c in cols])
    wb.save(str(out_path))


@MetadataWriter('yaml')
def write_yaml(rows, cols, out_path: Path):
    try:
        import yaml
    except Exception:
        return
    with open(out_path, 'w', encoding='utf-8') as fh:
        yaml.dump(rows, fh, default_flow_style=False)
