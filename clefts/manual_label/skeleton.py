from itertools import chain, compress

import pandas as pd

import os

import json
import re
from abc import ABCMeta
from enum import Enum
from io import TextIOBase
from json import JSONDecodeError
from typing import Sequence, Iterable, Dict
from pathlib import Path


class Side(Enum):
    RIGHT = "r"
    LEFT = "l"
    R = "r"
    L = "l"
    NONE = ""

    @classmethod
    def from_string(cls, s):
        if isinstance(s, cls):
            return s
        elif not s:
            return cls.NONE
        elif s.lower() in {'r', 'right'}:
            return cls.RIGHT
        elif s.lower() in {'l', 'left'}:
            return cls.LEFT
        else:
            raise ValueError(f"String '{s}' could not be interpreted as a side")

    def opposite(self):
        tp = type(self)
        if self == tp.RIGHT:
            return tp.LEFT
        elif self == tp.LEFT:
            return tp.RIGHT
        else:
            return tp.NONE


class Segment(Enum):
    NONE = ""
    T1 = "t1"
    T2 = "t2"
    T3 = "t3"
    A1 = "a1"
    A2 = "a2"
    A3 = "a3"
    A4 = "a4"
    A5 = "a5"
    A6 = "a6"
    A7 = "a7"
    A8 = "a8"
    A9 = "a9"

    @classmethod
    def from_string(cls, s):
        if isinstance(s, cls):
            return s
        elif s is None:
            return cls.NONE

        segment_dict = {seg.value: seg for seg in cls}

        try:
            return segment_dict[s.lower()]
        except KeyError:
            raise ValueError(f"String '{s}' could not be interpreted as a segment")


class Skeleton:
    def __init__(
            self, skid, name: str, side: Side, segment: Segment,
            classes: Iterable=None, superclasses: Iterable=None,
            annotations=None
    ):
        self.id = int(skid)
        self.name = name
        self.side = side
        self.segment = segment
        self.classes = {item.lower() for item in classes} if classes else set()
        self.superclasses = {item.lower() for item in superclasses} if superclasses else set()
        self.annotations = set(annotations) if annotations else set()

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return isinstance(other, Skeleton) and self.id == other.id

    def find_mirror(self, others):
        for other in others:
            if all([
                other.side == self.side.opposite(),
                other.segment == self.segment,
                other.classes == self.classes,
                other.superclasses == self.superclasses
            ]):
                return other

    def find_segment_copies(self, others):
        return [
            other for other in others
            if all([
                other.side == self.side,
                other.segment != self.segment,
                other.classes == self.classes,
                other.superclasses == self.superclasses
            ])
        ]

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "side": self.side.value,
            "segment": self.segment.value,
            "classes": sorted(self.classes),
            "superclasses": sorted(self.superclasses),
            "annotations": sorted(self.annotations)
        }

    def to_json(self, fpath=None, mode="r", **kwargs):
        d = self.to_dict()
        s = json.dumps(d, **kwargs)
        if fpath:
            with open(fpath, mode) as f:
                f.write(s)
        return s

    @classmethod
    def from_dict(cls, d):
        return cls(
            d["id"], d["name"],
            Side.from_string(d.get("side")), Segment.from_string(d["segment"]),
            d.get("classes", []), d.get("superclasses", []),
            d.get("annotations", [])
        )

    @classmethod
    def from_json(cls, s):
        if isinstance(s, TextIOBase):
            return json.load(s)
        try:
            return json.loads(s)
        except JSONDecodeError as e:
            if os.path.isfile(s):
                with open(s) as f:
                    return cls.from_json(f)
            raise e
        except TypeError as e:
            if isinstance(s, Path):
                return cls.from_json(str(s))
            raise e

    @classmethod
    def from_name(cls, skid, name, annotations=None, **kwargs):
        if "Basin" in name:
            return cls._from_basin_name(skid, name, annotations, **kwargs)
        if any(name.startswith(c) for c in ["Ladder", "Drunken", "Griddle"]):
            return cls._from_LN_name(skid, name, annotations, **kwargs)
        if " ORN " in name or " PN " in name:
            return cls._from_ORN_PN_name(skid, name, annotations, **kwargs)
        if re.match(".+ch.+a1[lr]", name):
            return cls._from_cho_name(skid, name, annotations, **kwargs)

        raise ValueError(f"Could not infer skeleton information from name '{name}' and given annotations")

    @classmethod
    def _from_basin_name(cls, skid, name, annotations=None, **kwargs):
        alphanum_class, seg_side, basin_class = name.split(' ')
        return cls(
            skid, name,
            Side.from_string(seg_side[-1]), Segment.from_string(seg_side[:-1]),
            [alphanum_class, basin_class],
            [alphanum_class[:-1], basin_class.split('-')[0]],
            annotations
        )

    @classmethod
    def _from_LN_name(cls, skid, name, annotations=None, **kwargs):
        class_, seg_side = name.split(' ')
        if seg_side.endswith('l') or seg_side.endswith('r'):
            side = Side.from_string(seg_side[-1])
            seg = Segment.from_string(seg_side[:-1])
        else:
            side = Side.NONE
            seg = Segment.from_string(seg_side)

        return cls(
            skid, name,
            side, seg,
            [class_],
            [class_.split('-')[0]],
            annotations
        )

    @classmethod
    def _from_ORN_PN_name(cls, skid, name, annotations=None, **kwargs):
        number, class_, side = name.split()
        return cls(
            skid, name,
            Side.from_string(side), Segment.NONE,
            ['{} {}'.format(number, class_)],
            [number, class_],
            annotations
        )

    @classmethod
    def _from_cho_name(cls, skid, name, annotations=None, **kwargs):
        name, seg_side = name.split()
        seg = Segment.from_string(seg_side[:-1])
        side = Side.from_string(seg_side[-1])

        classes = [name]
        superclasses = ["chordotonal"]
        if "'" in name:
            superclasses.append(name[:4])
        else:
            superclasses.append(name[:3])

        if "-" in name:
            superclasses.append(name.split("-")[0])

        return cls(
            skid, name,
            side, seg,
            classes,
            superclasses,
            annotations
        )

    @classmethod
    def from_hdf5(cls, path, group_key="skeletons/"):
        """
        Expects an HDF5 file like that produced by skeletons_to_tables

        Yields
        ------
        Skeleton
        """
        d = {
            key: one_hot_decode(pd.DataFrame.read_hdf(path, group_key + key))
            for key in ["classes", "superclasses", "annotations"]
        }

        for idx, row in pd.DataFrame.read_hdf(path, group_key + "skeletons"):
            yield cls(
                row["id"],
                row["name"],
                Side.from_string(row["side"]),
                Segment.from_string(row["segment"]),
                d["classes"].get(idx),
                d["superclasses"].get(idx),
                d["annotations"].get(idx)
            )


def one_hot_encode(categorical_data: Dict[int, Iterable]):
    """"""
    categories = sorted(set(chain.from_iterable(categorical_data.values())))
    out = dict()
    for key, value in categorical_data.items():
        out[key] = {cat: cat in value for cat in categories}
    return out


def one_hot_decode(table):
    d = dict()
    for idx, row in table.iterrows():
        d[idx] = compress(list(table), row)
    return d


def skeletons_to_tables(skeletons: Iterable[Skeleton]) -> Dict[str, pd.DataFrame]:
    """
    Represent skeletons as pandas dataframes which can be serialised to hdf5 by
    one-hot encoding categorical fields

    Parameters
    ----------
    skeletons

    Returns
    -------
    dict
        "skeletons": a table containing ID, name, side, segment
        "classes": a table containing classes for each skeleton, one-hot encoded
        "superclasses": as above
        "annotations": as above
    """
    skel_data = dict()
    for skel in skeletons:
        d = skel.to_dict()
        skel_data[skel.id] = {
            attr: d[attr] for attr in ["id", "name", "side", "segment"]
        }

    data_dicts = {"skeletons": skel_data}
    for key in ["classes", "superclasses", "annotations"]:
        data_dicts[key] = one_hot_encode(
            {skel.id: getattr(skel, key) for skel in skeletons}
        )

    return {
        key: pd.DataFrame.from_dict(data, orient="index")
        for key, data in data_dicts.items()
    }
