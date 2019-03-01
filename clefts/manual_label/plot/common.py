from abc import ABCMeta
from enum_custom import StrEnum


class SpecialCharacter(metaclass=ABCMeta):
    RIGHTWARDS_ARROW: str


class Unicode(SpecialCharacter):
    RIGHTWARDS_ARROW = "\u2192"


class TeX(SpecialCharacter):
    RIGHTWARDS_ARROW = r"\rightarrow"
