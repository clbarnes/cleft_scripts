from collections import defaultdict
from fnmatch import fnmatch
from itertools import zip_longest
import os
import logging
from datetime import datetime
import re
import shutil
from pathlib import Path
from typing import Iterator, Tuple, List, Optional
import subprocess as sp

from clefts.constants import PACKAGE_ROOT

logger = logging.getLogger("__name__")

timestamp_re = re.compile(r"^(?P<prefix>.+)(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)(?P<suffix>\.\w+)$")

timestamp = datetime.now().isoformat()

DEFAULT_ARCHIVE = '.zip'
IGNORE_GLOBS = ('.gitignore', )


def subtract_base(base: os.PathLike, path: os.PathLike):
    base_parts = Path(base).absolute().parts
    path_parts = Path(path).absolute().parts
    different_parts = list()
    for base_part, path_part in zip_longest(base_parts, path_parts):
        if base_part is None:
            different_parts.append(path_part)
        elif base_part != path_part:
            raise ValueError("base path is not a parent of other path")
    return Path(os.path.join(*different_parts))


def get_latest_plots(src_dir: os.PathLike, ignore_globs=IGNORE_GLOBS) -> Iterator[Tuple[Path, List[str]]]:
    """
    Get files and directores from the given tree, filtering out old versions of timestamped files.

    Yields 2-tuples, where the first item is a Path to the containing directory,
    relative to the given source directory, and the second is a list of file names inside that directory.

    Skips directories with no files in them
    """
    src_dir = Path(src_dir)

    for root, dirs, files in os.walk(src_dir, topdown=False):
        files_by_key = defaultdict(list)
        out_files = list()

        for fname in files:
            if ignore_globs and any(fnmatch(fname, g) for g in ignore_globs):
                logger.debug("File named '%s' matches an ignore pattern; ignoring", fname)
                continue
            match = timestamp_re.search(fname)
            if not match:
                out_files.append(fname)
            key = match.group("prefix") + match.group("suffix")
            files_by_key[key].append(fname)

        for key, fpaths in files_by_key.items():
            out_files.append(max(fpaths))

        if out_files:
            yield subtract_base(src_dir, root), out_files


def copy_latest_plots(src: os.PathLike, tgt: os.PathLike, archive_ext: Optional[str]=DEFAULT_ARCHIVE):
    """"""
    assert src != tgt, "src and tgt are the same"
    src = Path(src)
    tgt = Path(tgt)

    for root, fnames in get_latest_plots(src):
        src_dir = src / root
        tgt_dir = tgt / root
        tgt_dir.mkdir(exist_ok=True, parents=True)
        for fname in fnames:
            logging.debug("Copying %s to %s", src_dir / fname, tgt_dir / fname)
            shutil.copy2(src_dir / fname, tgt_dir / fname)

    if not archive_ext:
        return
    os.chdir(tgt)
    archive_name = tgt.name + archive_ext
    logging.info("Attempting to archive to %s", os.path.join(tgt, archive_name))

    if archive_ext == '.zip':
        sp.run(["zip", "-r", archive_name, '.'])
    elif archive_ext == '.tar.gz':
        sp.run(["tar", "-cvzf", archive_name, "."])
    else:
        raise ValueError("Archive extension given but not recognised, aborting")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    local_root = PACKAGE_ROOT / "manual_label"
    src = local_root / "figs"
    tgt = local_root / f"latest_plots_{timestamp}"

    copy_latest_plots(src, tgt)
