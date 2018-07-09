import math
import logging

from clefts.constants import DIMS, RESOLUTION, TRANSLATION, CoordZYX


logger = logging.getLogger(__name__)


def offset_shape_to_roi(offset, shape, dims='zyx'):
    """

    Parameters
    ----------
    offset : Mapping
    shape : Mapping
    dims : sequence

    Returns
    -------
    [[start_coords], [stop_coords]]
    """
    start = [offset.get(dim, 0) for dim in dims]
    stop = [offset.get(dim, 0) + shape.get(dim, 1) for dim in dims]
    return [start, stop]


def offset_shape_to_dicts(offset, shape):
    """

    Parameters
    ----------
    offset : Mapping
    shape : Mapping

    Returns
    -------
    {
        'min': dict,
        'max': dict
    }
    """
    return {
        'min': offset.copy(),
        'max': {dim: offset.get(dim, 0) + shape.get(dim, 1) for dim in offset}
    }


def center_radius_to_offset_shape(center, radius):
    """

    Parameters
    ----------
    center : dict
    radius : Number

    Returns
    -------
    dict
        offset
    dict
        shape
    """
    return {dim: center[dim] - radius for dim in DIMS}, {dim: radius*2 for dim in DIMS}


def center_side_to_offset_shape(center, side):
    """

    Parameters
    ----------
    center : Coordinate
    side : Number

    Returns
    -------
    CoordZYX
        Offset
    CoordZYX
        Shape
    """
    return center - side / 2, CoordZYX(1, 1, 1) * side


def px_to_nm(offset_px, shape_px=None):
    """

    Parameters
    ----------
    offset_px
    shape_px

    Returns
    -------

    """
    offset_nm = offset_px * RESOLUTION + TRANSLATION
    if shape_px:
        return offset_nm, shape_px * RESOLUTION
    else:
        return offset_nm


def nm_to_px(offset_nm, shape_nm=None):
    """

    Parameters
    ----------
    offset_nm
    shape_nm

    Returns
    -------

    """
    offset_px = math.floor((offset_nm - TRANSLATION) / RESOLUTION)
    if shape_nm:
        return offset_px, math.ceil(shape_nm / RESOLUTION)
    else:
        return offset_px


def offset_shape_to_slicing(offset, shape):
    max_bound = offset + shape
    return tuple(slice(offset[dim], max_bound[dim]) for dim in 'zyx')


def resolve_padding(padding_low=0, padding_high=None, fn=None, *args, **kwargs):
    """

    Parameters
    ----------
    padding_low : Coordinate or Number
    padding_high : Coordinate or Number, optional
        Default same as padding_low
    fn : callable
        Callable which takes a Coordinate as its first argument
    *args
        Additional arguments to pass to fn after the coordinate
    **kwargs
        Additional keyword arguments to pass to fn after the coordinate

    Returns
    -------

    """
    padding_low = (padding_low or 0) * CoordZYX(1, 1, 1)

    if padding_high is None:
        padding_high = padding_low
    padding_high = padding_high * CoordZYX(1, 1, 1)

    if fn is None:
        return padding_low, padding_high
    else:
        return fn(padding_low, *args, **kwargs), fn(padding_high, *args, **kwargs)
