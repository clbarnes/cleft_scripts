from collections import defaultdict

import os
import logging
from datetime import datetime
import re
import shutil

from clefts.constants import PACKAGE_ROOT

logger = logging.getLogger("__name__")

timestamp_re = re.compile(r"^(?P<prefix>.+)(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)(?P<suffix>\.\w+)$")

timestamp = datetime.now().isoformat()
fig_dir = PACKAGE_ROOT / "manual_label"

tgt_dir = fig_dir / f"latest_plots_{timestamp}"
shutil.copytree(fig_dir / "figs", tgt_dir)
for root, _, files in os.walk(tgt_dir):
    files_by_key = defaultdict(list)
    for fname in files:
        fpath = os.path.join(root, fname)
        match = timestamp_re.search(fname)
        if not match:
            continue
        key = match.group("prefix") + match.group("suffix")
        files_by_key[key].append(fpath)

    for key, fpaths in files_by_key.items():
        for fpath in sorted(fpaths)[:-1]:
            logger.info("Removing %s", fpath)
            os.remove(fpath)
