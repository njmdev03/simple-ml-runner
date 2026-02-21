import pandas as pd
import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class Exporter:
    @staticmethod
    def export(results: List[Dict[str, Any]], path: str):
        if not results:
            logger.warning("No results to export.")
            return

        df = pd.DataFrame(results)

        # Re-order columns: epoch, source, then others
        cols = list(df.columns)
        priority_cols = ['epoch', 'dataset', 'source']

        for col in reversed(priority_cols):
            if col in cols:
                # Move to the very front
                cols.insert(0, cols.pop(cols.index(col)))

        df = df[cols]

        ext = os.path.splitext(path)[1].lower()

        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None

        if ext == '.csv':
            df.to_csv(path, index=False)
            logger.info(f"Results saved to {path}")
        elif ext in ['.xlsx', '.xls']:
            df.to_excel(path, index=False)
            logger.info(f"Results saved to {path}")
        else:
            # Fallback to CSV
            csv_path = path + ".csv"
            df.to_csv(csv_path, index=False)
            logger.warning(f"Unknown extension {ext}, saved to {csv_path} as CSV instead.")
