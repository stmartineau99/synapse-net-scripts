"""
Utility functions for drawing segmentation masks from filament coordinates.

Adapted from TARDIS (tardis_em/cnn/data_processing/draw_mask.py)
Original authors: Robert Kiewisz, Tristan Bepler (MIT License 2021-2025)
"""

from math import pow, sqrt
from typing import Tuple, Iterable, Union
import numpy as np

def draw_instances(
    mask_size: Union[list, tuple],
    coordinate: np.ndarray,
    pixel_size: float,
    circle_size=70,
    label=True,
    dtype=None,
) -> np.ndarray:
    """
    Draws labeled or binary 3D masks based on the input coordinates, mask size, and additional parameters.

    If label generation is enabled, unique segment labels are created to distinguish between different segments.
    Spheres centered around specific points are drawn in the masks, with sizes determined by the input circle size
    and pixel size. Input parameters related to the shape, size, and type of the mask, as well as the labeling behavior,
    are fully customizable. This function is suitable for constructing semantic or instance masks.

    :param mask_size: The dimensions of the mask to be created.
    :type mask_size: list | tuple
    :param coordinate: An array of coordinates specifying the locations to draw the mask,
        shape [Label x Z x Y x X].
    :type coordinate: np.ndarray
    :param pixel_size: Pixel size of the mask, used to scale the mask appropriately.
    :type pixel_size: float
    :param circle_size: Diameter of the sphere to be drawn at each coordinate point. Defaults to 250.
    :type circle_size: int, optional
    :param label: Flag indicating whether a labeled mask or a binary mask should be created. Defaults to True.
    :type label: bool, optional
    :param dtype: Data type for the output mask. If not provided, defaults to np.uint16 or np.uint8.
    :type dtype: str | None, optional

    :return: A 3D mask generated based on the input parameters.
    :rtype: np.ndarray
    """
    if label:
        if coordinate.ndim != 2 or coordinate.shape[1] not in [3, 4]:
            raise ValueError(
                "Coordinates are of not correct shape, expected: "
                f"shape [Label x Z x Y x X] but {coordinate.shape} given!"
            )

        label_mask = np.zeros(mask_size, dtype=np.uint16 if dtype is None else dtype)

        segments = np.unique(coordinate[:, 0])

        for i in segments:
            pts = coordinate[np.where(coordinate[:, 0] == i)[0]][:, 1:]

            all_cz, all_cy, all_cx = draw_filament(pts, label_mask, pixel_size, circle_size=circle_size)

            label_mask[all_cz, all_cy, all_cx] = i + 1

        return label_mask
    else:
        label_mask = np.zeros(mask_size, dtype=np.uint8 if dtype is None else dtype)

        all_cz, all_cy, all_cx = draw_filament(coordinate, label_mask, pixel_size, circle_size=circle_size)
        label_mask[all_cz, all_cy, all_cx] = 1

        return label_mask


def draw_filament(
    coordinates: np.ndarray, mask: np.ndarray, pixel_size: float, circle_size=70, interpolate=True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Draws spheres along a filament and returns the covered voxel indices.

    :param coordinates: Filament points of shape (N, 3) as [Z x Y x X].
    :param mask: The 3D mask the filament will be drawn into.
    :param pixel_size: Pixel size of the mask, used to scale the sphere radius.
    :param circle_size: Diameter of the sphere drawn at each point.
    :return: A tuple of (cz, cy, cx) index arrays for the covered voxels.
    """
    if pixel_size == 0:
        pixel_size = 1

    r = circle_size / 2

    all_cz, all_cy, all_cx = [], [], []

    if interpolate:
        coordinates = interpolation(coordinates)
    for c in coordinates:
        cz, cy, cx = draw_sphere(radius=r, coordinates=c, label_mask=mask, pixel_size=pixel_size)
        all_cz.append(cz)
        all_cy.append(cy)
        all_cx.append(cx)

    all_cz, all_cy, all_cx = (
        np.concatenate(all_cz),
        np.concatenate(all_cy),
        np.concatenate(all_cx),
    )

    return all_cz, all_cy, all_cx


def draw_sphere(
    radius: int, coordinates: np.ndarray, label_mask: np.ndarray, pixel_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return coordinates of a sphere centered at c.

    :param r: The radius of the sphere in Angstrom.
    :param c: A numpy array containing the center coordinates [z, y, x].
    :param label_mask: A numpy array representing the label mask (3D).
    :return: A tuple of (cz, cy, cx) index arrays for the sphere voxels.
    """
    if label_mask.ndim != 3:
        raise ValueError(f"Unsupported dimensions {label_mask.ndim}, expected 3.")

    # sphere bounding box centered at the origin
    r = int(np.ceil(radius / pixel_size))
    offsets = np.arange(-r, r + 1)
    Z, Y, X = np.meshgrid(offsets, offsets, offsets, indexing="ij")

    # sphere test based on the physical radius
    distance = np.sqrt(
        (Z * pixel_size) ** 2 +
        (Y * pixel_size) ** 2 +
        (X * pixel_size) ** 2
    )
    sphere = distance <= radius

    sphere_offsets = np.argwhere(sphere) - r
    c = coordinates + sphere_offsets

    # remove negative and out of frame coordinates
    valid = np.all((c >= 0) & (c < np.asarray(label_mask.shape)), axis=1)
    c = c[valid]

    return c[:, 0], c[:, 1], c[:, 2]

def interpolation(points: np.ndarray) -> np.ndarray:
    """
    Performs 3D interpolation for XYZ coordinates using a given interpolation
    generator over sequential point pairs, and appends the last point to the
    output in integer format. The function creates a continuous series of
    interpolated points connecting all input coordinates.

    :param points: An array of shape (N, 3) representing a sequence of 3D points.

    :return: An array of shape (M, 3) containing the interpolated 3D coordinates,
             where M >= N due to newly generated intermediate points.
    """
    new_coord = []
    for i in range(0, len(points) - 1):
        """3D interpolation for XYZ dimension"""
        new_coord.append(list(interpolate_generator(points[i : i + 2, :])))

    # Append last point
    new_coord.append(list(np.round(points[-1, :]).astype(np.int32)))

    return np.vstack(new_coord)


def interpolate_generator(points: np.ndarray) -> Iterable:
    """
    Generates interpolated points between two given 2D or 3D points. The function
    determines the interpolation path in a pixel-grid-like manner between the start
    and end points, based on Bresenham-like algorithms. Interpolation supports
    only 2D and 3D coordinate systems, and exactly two points must be provided for
    interpolation.

    :param points: An array of shape (2, 2) for 2D points or (2, 3) for 3D points.
                   Defines the two points between which interpolation is performed.
    :return: An iterable generator that yields tuples representing (z, y) in 2D
             space and (z, y, x) in 3D space.
    """
    if points.shape not in [(2, 3), (2, 2)]:
        raise ValueError(
            "Interpolation supports only 2D/3D for 2 points at a time; "
            f"but {points.shape} was given!"
        )

    points = np.round(points).astype(np.int32)
    if points.shape == (2, 2):
        dim_ = 2
    else:
        dim_ = 3

    # Collect first and last point in array for ZYX
    z0, z1 = points[0, 0], points[1, 0]
    y0, y1 = points[0, 1], points[1, 1]
    if dim_ == 2:
        x0, x1 = 0, 0
    else:
        x0, x1 = points[0, 2], points[1, 2]

    # Delta between first and last point to interpolate
    delta_z, delta_y, delta_x = z1 - z0, y1 - y0, x1 - x0

    # Calculate axis to iterate throw
    max_delta = np.where(
        (abs(delta_z), abs(delta_y), abs(delta_x))
        == np.max((abs(delta_z), abs(delta_y), abs(delta_x)))
    )[0][0]
    if delta_z == 0 and delta_y == 0 and delta_x == 0:
        max_delta = 3

    # Calculate scaling direction + or - or None
    dz_sign, dy_sign, dx_sign = np.sign(delta_z), np.sign(delta_y), np.sign(delta_x)

    # Calculating scaling threshold
    delta_err_z = (
        0.0
        if delta_z == 0
        else (
            abs(delta_z / delta_y)
            if delta_y != 0
            else abs(delta_z / delta_x) if delta_x != 0 else 0.0
        )
    )
    delta_err_y = (
        0.0
        if delta_y == 0
        else (
            abs(delta_y / delta_z)
            if delta_z != 0
            else abs(delta_y / delta_x) if delta_x != 0 else 0.0
        )
    )

    if dim_ != 2:
        delta_err_x = (
            0.0
            if delta_x == 0
            else (
                np.minimum(abs(delta_x / delta_z), abs(delta_x / delta_y))
                if delta_z != 0 and delta_y != 0
                else (
                    abs(delta_x / delta_y)
                    if delta_z == 0 and delta_y != 0
                    else abs(delta_x / delta_z) if delta_z != 0 else 0.0
                )
            )
        )

    # Zero out threshold
    error_z, error_y, error_x = 0, 0, 0
    z, y, x = z0, y0, x0

    if max_delta == 0:  # Scale ZYX by iterating throw Z axis
        for z in range(z0, z1, dz_sign):
            if dim_ != 2:
                yield z, y, x
            else:
                yield z, y

            # Iteratively add and scale Y axis
            error_y = error_y + delta_err_y
            while error_y >= 0.5:
                y += dy_sign
                error_y -= 1

            if dim_ != 2:
                # Iteratively add and scale X axis
                error_x = error_x + delta_err_x
                while error_x >= 0.5:
                    x += dx_sign
                    error_x -= 1
    if max_delta == 1:  # Scale ZYX by iterating throw Y axis
        for y in range(y0, y1, dy_sign):
            if dim_ != 2:
                yield z, y, x
            else:
                yield z, y

            # Iteratively add and scale Z axis
            error_z = error_z + delta_err_z
            while error_z >= 0.5:
                z += dz_sign
                error_z -= 1

            if dim_ != 2:
                # Iteratively add and scale X axis
                error_x = error_x + delta_err_x
                while error_x >= 0.5:
                    x += dx_sign
                    error_x -= 1
    if max_delta == 2:  # Scale ZYX by iterating throw X axis
        for x in range(x0, x1, dx_sign):
            if dim_ != 2:
                yield z, y, x
            else:
                yield z, y
    if max_delta == 3:  # Nothing to do
        if dim_ != 2:
            yield z, y, x
        else:
            yield z, y