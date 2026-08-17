"""
Conversion helpers: SPECS Prodigy .xy export (see import_xy.py) -> the
Node/Spectrum data model used throughout import_h5.py.

Kept separate from import_h5.py because the .xy format's own internal
structure (Group/Region/Scan, kind classification, energy-axis quirks,
scan-completeness handling) is unrelated to HDF5 concerns and
self-contained enough to develop/test in isolation. This module has no
dependency on import_h5.py (only on import_xy.py + numpy), so
import_h5.py's SpecsXYLoader can import from here without any
circular-import trouble.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass

import numpy as np

from scripts.import_xy import Region, Scan

logger = logging.getLogger(__name__)

EPOCH = np.datetime64("1970-01-01T00:00:00")

# A scan missing more than this fraction of the reference region's energy
# points is dropped outright rather than folded into the stacked array as
# a mostly-empty (NaN) row — real acquisitions can drop a handful of
# points per scan (fine to keep, padded with NaN), but a scan missing
# *most* of its points means something went wrong with that scan
# specifically and it shouldn't drag every other scan down to its size.
MAX_MISSING_FRACTION = 0.5


def _acquisition_datetime(scan: Scan) -> np.datetime64:
    raw = scan.metadata.get("Acquisition Date")
    if raw is None:
        raise ValueError("Scan is missing 'Acquisition Date'")
    # e.g. "2023-10-27 08:34:22 UTC" -> "2023-10-27T08:34:22"
    return np.datetime64(raw.replace(" UTC", "").replace(" ", "T"))


def _reference_energy_axis(scans: list[Scan]) -> np.ndarray:
    """
    Pick the energy axis that the majority of `scans` agree on exactly,
    to use as the grid every scan (including incomplete ones) gets
    aligned onto. Deliberately doesn't trust region metadata point
    counts (some are stale, e.g. "Number of Scans" was observed to
    always read 1 regardless of the real scan count).
    """
    counts: Counter = Counter()
    axis_by_key: dict[tuple, np.ndarray] = {}
    for s in scans:
        if len(s.energy) == 0:
            continue
        key = (len(s.energy), round(float(s.energy[0]), 6), round(float(s.energy[-1]), 6))
        counts[key] += 1
        axis_by_key.setdefault(key, s.energy)

    if not counts:
        raise ValueError("No scan in region has any data points")

    best_key, _ = counts.most_common(1)[0]
    return axis_by_key[best_key]


def _align_scan_to_reference(reference: np.ndarray, scan: Scan, step: float) -> np.ndarray | None:
    """
    Map one scan's (possibly incomplete) energy/intensity pair onto the
    region's reference energy grid, returning an intensity array the
    same length as `reference` with NaN at any point this scan is
    missing. Returns None if the scan is missing more than
    MAX_MISSING_FRACTION of the reference points (see module docstring).
    """
    if len(scan.energy) < len(reference) * (1 - MAX_MISSING_FRACTION):
        return None

    if len(scan.energy) == len(reference) and np.allclose(scan.energy, reference, atol=abs(step) / 2):
        return scan.intensity

    # Missing points aren't necessarily a contiguous run at one edge, so
    # place each point by its own position on the (evenly-spaced)
    # reference grid rather than assuming a trimmed prefix/suffix.
    idx = np.round((scan.energy - reference[0]) / step).astype(int)
    aligned = np.full(len(reference), np.nan)
    valid = (idx >= 0) & (idx < len(reference))
    aligned[idx[valid]] = scan.intensity[valid]
    return aligned


@dataclass
class StackedRegion:
    """Plain result handed back to import_h5.py's SpecsXYLoader."""

    xps: np.ndarray  # shape (n_kept_scans, n_energy_points)
    energy_axis: np.ndarray  # Binding Energy, shape (n_energy_points,)
    binding_energy: np.ndarray  # shape (n_energy_points,)
    kinetic_energy: np.ndarray  # shape (n_energy_points,)
    iter_axis: np.ndarray
    acq_time: np.ndarray  # epoch seconds, shape (n_kept_scans,)
    start_time: np.ndarray  # datetime64[s], shape (n_kept_scans,)
    stop_time: np.ndarray  # datetime64[s], shape (n_kept_scans,)
    native_energy_mode: str  # "Binding" or "Kinetic", as found in the file
    discarded_scans: list[tuple[int, int]]  # (cycle_index, scan_index) of dropped scans
    work_function: float | None = None  # "Eff. Workfunction" from region metadata, if present
    position_y: np.ndarray | None = None
    position_z: np.ndarray | None = None
    position_step: np.ndarray | None = None


def stack_region(region: Region, file_metadata: dict) -> StackedRegion:
    """
    Stack every (kept) scan in `region` into a single 2D (n_scans,
    n_energy_points) array, regardless of region.kind — a "single"
    region trivially stacks to one row, "series"/"map" to many.

    Both binding_energy and kinetic_energy are always populated, whichever
    the file's own export setting (or, as a fallback, the region's "Scan
    Variable") says is native: the other is derived via `Excitation Energy
    - Eff. Workfunction - energy` (Eff. Workfunction defaults to 0 if the
    region's metadata doesn't have it). `energy_axis` remains Binding
    Energy, for backward compatibility with existing callers.
    """
    reference = _reference_energy_axis(region.scans)
    step = float(np.median(np.diff(reference)))

    axis_setting = file_metadata.get("Energy Axis") or region.metadata.get("Scan Variable", "Binding Energy")
    native_mode = "Kinetic" if "Kinetic" in axis_setting else "Binding"

    rows: list[np.ndarray] = []
    start_times: list[np.datetime64] = []
    positions_y: list[float] = []
    positions_z: list[float] = []
    positions_step: list[float] = []
    discarded: list[tuple[int, int]] = []

    for scan in region.scans:
        aligned = _align_scan_to_reference(reference, scan, step)
        if aligned is None:
            discarded.append((scan.cycle_index, scan.scan_index))
            logger.warning(
                "Discarding scan (cycle=%d, scan=%d) in region '%s': only %d/%d energy points present",
                scan.cycle_index, scan.scan_index, region.name, len(scan.energy), len(reference),
            )
            continue

        rows.append(aligned)
        start_times.append(_acquisition_datetime(scan))
        positions_y.append(scan.parameters.get("Y [mm]", np.nan))
        positions_z.append(scan.parameters.get("Z [mm]", np.nan))
        positions_step.append(scan.parameters.get("Step", np.nan))

    if not rows:
        raise ValueError(f"Region '{region.name}': every scan was discarded as incomplete")

    dwell_time = float(region.metadata.get("Dwell Time") or 0.0)

    # "Dwell Time" is the time spent *per energy point* (e.g. "# Dwell
    # Time: 0.1" alongside "# Values/Curve: 141" in the file), not the
    # whole scan's duration -- a full scan sweeps every point in
    # `reference`, so its real duration is dwell_time * len(reference).
    # Using dwell_time alone here previously left stop_time only a
    # fraction of a second after start_time regardless of how long the
    # scan actually took, which made a single scan's time window far
    # too narrow to ever overlap an auxiliary log's sample points (see
    # _spectrum_time_window/_filter_node_by_time in
    # utils/spectrum_utils.py) even though the *region's* (multi-scan)
    # window, spanning start_time.min() to stop_time.max() across
    # scans, was wide enough to work fine.
    scan_duration = dwell_time * len(reference)

    # Acquisition Date strings only carry whole-second resolution, but
    # scan_duration is commonly sub-second -- rounding the start->stop
    # offset to whole seconds would collapse stop_time onto start_time,
    # so the offset itself is applied in milliseconds.
    start_time = np.array(start_times, dtype="datetime64[s]").astype("datetime64[ms]")
    stop_time = start_time + np.timedelta64(int(round(scan_duration * 1000)), "ms")
    acq_time = (start_time - EPOCH) / np.timedelta64(1, "s") + scan_duration / 2

    work_function_raw = region.metadata.get("Eff. Workfunction")
    work_function = float(work_function_raw) if work_function_raw is not None else None

    if native_mode == "Binding":
        binding_energy = reference
        try:
            kinetic_energy = _to_kinetic_energy(reference, region, work_function)
        except ValueError:
            # kinetic_energy is a derived convenience here (binding_energy,
            # the native axis, is already fully known) -- missing
            # Excitation Energy shouldn't block loading the region over it.
            logger.warning(
                "Region '%s': no 'Excitation Energy' to derive kinetic_energy "
                "from binding_energy; leaving it as NaN", region.name,
            )
            kinetic_energy = np.full_like(reference, np.nan)
    else:
        kinetic_energy = reference
        binding_energy = _to_binding_energy(reference, region, work_function)
    energy_axis = binding_energy

    return StackedRegion(
        xps=np.stack(rows, axis=0),
        energy_axis=energy_axis,
        binding_energy=binding_energy,
        kinetic_energy=kinetic_energy,
        iter_axis=np.arange(len(rows)),
        acq_time=acq_time,
        start_time=start_time,
        stop_time=stop_time,
        native_energy_mode=native_mode,
        discarded_scans=discarded,
        work_function=work_function,
        position_y=np.array(positions_y) if region.kind == "map" else None,
        position_z=np.array(positions_z) if region.kind == "map" else None,
        position_step=np.array(positions_step) if region.kind == "map" else None,
    )


def _to_binding_energy(
    kinetic_axis: np.ndarray, region: Region, work_function: float | None = None
) -> np.ndarray:
    Eph = region.metadata.get("Excitation Energy")
    if Eph is None:
        raise ValueError(
            f"Region '{region.name}' has Kinetic-energy data but no 'Excitation Energy' to convert with"
        )
    Wf = work_function if work_function is not None else 0.0
    return float(Eph) - Wf - kinetic_axis


def _to_kinetic_energy(
    binding_axis: np.ndarray, region: Region, work_function: float | None = None
) -> np.ndarray:
    Eph = region.metadata.get("Excitation Energy")
    if Eph is None:
        raise ValueError(
            f"Region '{region.name}' has Binding-energy data but no 'Excitation Energy' to convert with"
        )
    Wf = work_function if work_function is not None else 0.0
    return float(Eph) - Wf - binding_axis
