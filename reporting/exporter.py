import pandas as pd
import os
from typing import List, Dict, Any

class Exporter:
    @staticmethod
    def export(results: List[Dict[str, Any]], path: str):
        if not results:
            print("No results to export.")
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
            print(f"Results saved to {path}")
        elif ext in ['.xlsx', '.xls']:
            df.to_excel(path, index=False)
            print(f"Results saved to {path}")
        else:
            # Fallback to CSV
            csv_path = path + ".csv"
            df.to_csv(csv_path, index=False)
            print(f"Unknown extension {ext}, saved to {csv_path} as CSV instead.")
