import functools
from itertools import chain, compress

import pandas as pd

import os

import json
import random
import re
from abc import ABCMeta, abstractmethod
from io import TextIOBase
from json import JSONDecodeError
from typing import Iterable, Dict, Optional
from pathlib import Path

from clefts.common import StrEnum
from clefts.manual_label.plot.constants import USE_TEX


def hdf_join(path, *args):
    """Like os.path.join, but for HDF5 hierarchies. N.B. strips trailing, but not leading, slash from entire path"""
    path = path.rstrip("/")
    for arg in args:
        arg = arg.strip("/")
        path += "/" + arg
    return path


@functools.total_ordering
class Side(StrEnum):
    RIGHT = "r"
    LEFT = "l"
    R = "r"
    L = "l"
    # BILATERAL = 'bi'  # groups only
    UNDEFINED = ""

    @classmethod
    def from_str(cls, s):
        if not s:
            s = ""

        if isinstance(s, str):
            s = {"r": "r", "l": "l", "right": "r", "left": "l", "": ""}[
                s.strip().lower()
            ]

        return cls(s)

    def opposite(self):
        tp = type(self)
        if self == tp.RIGHT:
            return tp.LEFT
        elif self == tp.LEFT:
            return tp.RIGHT
        else:
            return tp.UNDEFINED

    @classmethod
    def from_group(cls, sides, ignore_none=False):
        """
        Get the side of a group of sides. If ignore_none is False, any NONE will set the group side to NONE.
        """
        sides = set(sides)
        if ignore_none:
            sides.discard(cls.UNDEFINED)
        elif cls.UNDEFINED in sides:
            return cls.UNDEFINED

        if len(sides) == 1:
            return sides.pop()
        else:
            return cls.UNDEFINED

    def __lt__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        return str(self) < str(other)


@functools.total_ordering
class Segment(StrEnum):
    UNDEFINED = ""
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
    # MULTI = "multi"

    @classmethod
    def from_str(cls, s):
        if not s:
            s = ""

        if isinstance(s, str):
            s = s.lower().strip()

        return cls(s)

    @classmethod
    def from_group(cls, segments, ignore_none=False):
        segments = set(segments)
        if ignore_none:
            segments.discard(cls.UNDEFINED)
        elif cls.UNDEFINED in segments:
            return cls.UNDEFINED

        if len(segments) == 1:
            return segments.pop()
        else:
            return cls.UNDEFINED

    def part_num(self):
        if self == type(self).UNDEFINED:
            return None
        part = self.value[0]
        num = int(self.value[1:])
        return part, num

    def __lt__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented

        values = list(type(self))
        return values.index(self) < values.index(other)


@functools.total_ordering
class CircuitNode(metaclass=ABCMeta):
    def __init__(
        self,
        skid,
        name: Optional[str],
        side: Side,
        segment: Segment,
        classes: Iterable = None,
        superclasses: Iterable = None,
        annotations=None,
    ):
        self.id = int(skid)
        self.side = Side.from_str(side)
        self.segment = Segment.from_str(segment)
        self.classes = frozenset(item.lower() for item in classes) if classes else frozenset()
        self.superclasses = (
            frozenset(item.lower() for item in superclasses)
            if superclasses
            else frozenset()
        )
        self.annotations = frozenset(annotations) if annotations else frozenset()
        self._name = name or self.create_name()

    @abstractmethod
    def create_name(self) -> str:
        pass

    def find_mirrors(self, others):
        return [
            other
            for other in others
            if all(
                [
                    other.side == self.side.opposite(),
                    other.segment == self.segment,
                    other.classes == self.classes,
                    other.superclasses == self.superclasses,
                ]
            )
        ]

    def find_copies(self, others):
        return [other for other in others if self == other]

    def find_segment_copies(self, others):
        return [
            other
            for other in others
            if all(
                [
                    other.side == self.side,
                    other.segment != self.segment,
                    other.classes == self.classes,
                    other.superclasses == self.superclasses,
                ]
            )
        ]

    def _to_sort_key(self):
        return (
            sorted(self.superclasses),
            sorted(self.classes),
            self.segment,
            self.side,
            self.id,
        )

    def __lt__(self, other):
        if not isinstance(other, CircuitNode):
            raise TypeError(
                f"'<' not supported between instances of '{type(self).__name__}' and '{type(other).__name__}'"
            )
        return self._to_sort_key() < other._to_sort_key()

    def __eq__(self, other):
        return isinstance(other, CircuitNode) and all(
            [
                getattr(self, attr) == getattr(other, attr)
                for attr in ["side", "segment", "classes", "superclasses"]
            ]
        )

    @property
    def name(self):
        return self._name

    def __str__(self):
        return self.name


@functools.total_ordering
class Crossing(StrEnum):
    UNDEFINED = ""
    IPSI = "ipsi"
    IPSILATERAL = "ipsi"
    CONTRA = "contra"
    CONTRALATERAL = "contra"

    @classmethod
    def from_str(cls, s):
        if isinstance(s, cls):
            return s

        if not s:
            return cls.UNDEFINED

        if s.lower().startswith("ipsi"):
            return cls.IPSILATERAL
        elif s.lower().startswith("contra"):
            return cls.CONTRALATERAL
        else:
            raise ValueError(f"{repr(s)} could not be interpreted as a {cls.__name__}")

    @classmethod
    def from_sides(cls, src, tgt):
        if src == Side.UNDEFINED or tgt == Side.UNDEFINED:
            return cls.UNDEFINED
        elif src == tgt:
            return cls.IPSILATERAL
        else:
            return cls.CONTRALATERAL

    @classmethod
    def from_skeletons(cls, src: CircuitNode, tgt: CircuitNode):
        return cls.from_sides(src.side, tgt.side)

    @classmethod
    def from_group(cls, crossings, ignore_none=False):
        crossings = set(crossings)
        if ignore_none:
            crossings.discard(cls.UNDEFINED)
        elif cls.UNDEFINED in crossings:
            return cls.UNDEFINED

        if cls.CONTRALATERAL in crossings:
            return cls.CONTRALATERAL
        elif cls.IPSILATERAL in crossings:
            return cls.IPSILATERAL
        else:
            return cls.UNDEFINED

    def __lt__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        values = list(type(self))
        return values.index(self) < values.index(other)


class Skeleton(CircuitNode):
    # def __str__(self):
    #     return self.name

    def __repr__(self):
        return f"<Skeleton {self.id} ({self.name})>"

    def __hash__(self):
        return hash(self.id)

    def create_name(self):
        segside = self.segment.value + self.side.value
        classes = " ".join(sorted(self.classes))
        return f"{classes} {segside}" if segside else classes

    def to_dict(self):
        return {
            key: getattr(self, key)
            for key in [
                "id",
                "name",
                "side",
                "segment",
                "classes",
                "superclasses",
                "annotations",
            ]
        }

    def to_json(self, fpath=None, mode="r", **kwargs):
        d = self.to_dict()

        # StrEnum members
        for key in ["side", "segment"]:
            d[key] = str(d[key])

        # set members
        for key in ["classes", "superclasses", "annotations"]:
            d[key] = sorted(d[key])

        s = json.dumps(d, **kwargs)
        if fpath:
            with open(fpath, mode) as f:
                f.write(s)
        return s

    @classmethod
    def from_dict(cls, d):
        return cls(
            d["id"],
            d["name"],
            d.get("side"),
            d.get("segment"),
            d.get("classes", []),
            d.get("superclasses", []),
            d.get("annotations", []),
        )

    @classmethod
    def from_json(cls, s):
        if isinstance(s, TextIOBase):
            return cls.from_dict(json.load(s))
        try:
            return cls.from_dict(json.loads(s))
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
        if "broad" in name:
            return cls._from_broad_name(skid, name, annotations, **kwargs)
        if re.match(r"^[lv]\'?ch.*\sa1[lr]$", name):
            return cls._from_cho_name(skid, name, annotations, **kwargs)

        raise ValueError(
            f"Could not infer skeleton information from name '{name}' and given annotations"
        )

    @classmethod
    def _from_basin_name(cls, skid, name, annotations=None, **kwargs):
        alphanum_class, seg_side, basin_class = name.split(" ")
        return cls(
            skid,
            name,
            Side.from_str(seg_side[-1]),
            Segment.from_str(seg_side[:-1]),
            [alphanum_class, basin_class],
            [alphanum_class[:-1], basin_class.split("-")[0]],
            annotations,
        )

    @classmethod
    def _from_LN_name(cls, skid, name, annotations=None, **kwargs):
        class_, seg_side = name.split(" ")
        if seg_side.endswith("l") or seg_side.endswith("r"):
            side = Side.from_str(seg_side[-1])
            seg = Segment.from_str(seg_side[:-1])
        else:
            side = Side.UNDEFINED
            seg = Segment.from_str(seg_side)

        return cls(skid, name, side, seg, [class_], [class_.split("-")[0]], annotations)

    @classmethod
    def _from_ORN_PN_name(cls, skid, name, annotations=None, **kwargs):
        number, class_, side = name.split()
        return cls(
            skid,
            name,
            Side.from_str(side),
            Segment.UNDEFINED,
            ["{} {}".format(number, class_)],
            [number, class_],
            annotations,
        )

    @classmethod
    def _from_broad_name(cls, skid, name, annotations=None, **kwargs):
        class_, subclass, side = name.split()
        subclass_letters = subclass.rstrip('0123456789')
        return cls(
            skid,
            name,
            Side.from_str(side),
            Segment.UNDEFINED,
            [f"{class_} {subclass}"],  # classes
            [class_, subclass, subclass_letters],  # superclasses
            annotations
        )

    @classmethod
    def _from_cho_name(cls, skid, name, annotations=None, **kwargs):
        class_, seg_side = name.split()
        seg = Segment.from_str(seg_side[:-1])
        side = Side.from_str(seg_side[-1])

        classes = [class_]
        superclasses = ["chordotonal"]
        if "'" in class_:
            superclasses.append(class_[:4])
        else:
            superclasses.append(class_[:3])

        if "-" in class_:
            superclasses.append(class_.split("-")[0])

        if '/' in name:
            name += f" ({skid})"

        return cls(skid, name, side, seg, classes, superclasses, annotations)

    @classmethod
    def from_hdf5(cls, path, group_key="skeletons/"):
        """
        Expects an HDF5 file like that produced by skeletons_to_tables

        Yields
        ------
        Skeleton
        """
        d = {
            key: one_hot_decode(pd.read_hdf(path, hdf_join(group_key, key)))
            for key in ["classes", "superclasses", "annotations"]
        }

        for row in pd.read_hdf(path, hdf_join(group_key, "skeletons")).itertuples(
            index=False
        ):
            yield cls(
                row.id,
                row.name,
                Side.from_str(row.side),
                Segment.from_str(row.segment),
                d["classes"].get(row.id),
                d["superclasses"].get(row.id),
                d["annotations"].get(row.id),
            )


class SkeletonGroup(CircuitNode):
    def __init__(self, skeletons: Iterable[Skeleton] = None, ignore_none=False):
        self.skeletons = frozenset(skeletons or [])
        sides = set()
        segments = set()
        class_groups = set()

        classes = set()
        superclasses = set()
        annotations = set()

        for skel in self.skeletons:
            sides.add(skel.side)
            segments.add(skel.segment)
            class_groups.add(tuple(sorted(skel.classes)))
            classes.update(skel.classes)
            superclasses.update(skel.superclasses)
            annotations.update(skel.annotations)

        self._sides = frozenset(sides)
        self._segments = frozenset(segments)
        self._class_groups = frozenset(class_groups)

        super().__init__(
            self._get_id(),
            None,
            Side.from_group(sides, ignore_none),
            Segment.from_group(segments, ignore_none),
            classes,
            superclasses,
            annotations,
        )

    def create_name(self):
        skel_names = {str(skel) for skel in self.skeletons}
        if len(skel_names) == 1:
            return skel_names.pop()
        segside = self.segment.value + self.side.value
        classes = ", ".join(" ".join(sorted(grp)) for grp in sorted(self._class_groups))
        return f"{classes} {segside}" if segside else classes

    def __hash__(self):
        return hash(self.skeletons)

    def _get_id(self):
        r = random.Random(hash(self))
        return r.randint(0, 2 ** 64 - 1)

    def __contains__(self, item):
        if isinstance(item, Skeleton):
            return item in self.skeletons

        try:
            return int(item) in {skel.id for skel in self.skeletons}
        except (TypeError, ValueError) as e:
            raise ValueError from e

    def union(self, *others, ignore_none=False):
        skels = set(self.skeletons)
        for other in others:
            if isinstance(other, Skeleton):
                skels.add(other)
            else:
                skels.update(other)
        return type(self)(skels, ignore_none)

    def __iter__(self):
        return iter(self.skeletons)


def one_hot_encode(categorical_data: Dict[int, Iterable]):
    """"""
    all_categories = sorted(set(chain.from_iterable(categorical_data.values())))
    out = dict()
    for key, value in categorical_data.items():
        out[key] = {cat: cat in value for cat in all_categories}
    return out


def one_hot_decode(table):
    d = dict()
    for idx, row in table.iterrows():
        d[idx] = list(compress(list(table), row))
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
        skel_data[skel.id] = {
            "id": skel.id,
            "name": skel.name,
            "side": str(skel.side),
            "segment": str(skel.segment),
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


def edge_name(src, tgt, tex=USE_TEX):
    arrow = r"$\rightarrow$" if tex else "->"
    crossing = Crossing.from_skeletons(src, tgt)
    return f"{src} {arrow} {tgt} {str(crossing).upper()}"
