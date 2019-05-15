from pathlib import Path

import pandas as pd

here = Path(__file__).absolute().parent
all_path = here / "count_frac_area_all.csv"

all_df = pd.read_csv(all_path, index_col=False)
