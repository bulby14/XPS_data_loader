"""
Pure data-transformation helpers for Node/Spectrum objects (see
scripts/import_h5.py) — no Qt/GUI dependency, so these can be used or
tested independently of the gui package.
"""

from __future__ import annotations

import numpy as np

from scripts.import_h5 import Node, Spectrum


def _spectrum_time_window(node: Node) -> tuple[np.datetime64, np.datetime64] | None:
    """
    Get the overall (earliest start, latest stop) time window for a
    spectrum's raw node, or None if it has no start_time/stop_time
    arrays.
    """
    if "start_time" not in node.arrays or "stop_time" not in node.arrays:
        return None
    start_time = node["start_time"]
    stop_time = node["stop_time"]
    if len(start_time) == 0 or len(stop_time) == 0:
        return None
    return start_time.min(), stop_time.max()


def _single_scan_spectrum(spectrum: Spectrum, scan_index: int) -> Spectrum:
    """
    Slice one row out of a multi-scan spectrum's 'raw' node into its
    own 1-row Spectrum, so a Scan tree entry can go through the exact
    same display pipeline as any other spectrum.
    """
    node = spectrum.nodes["raw"]
    single = Spectrum(
        label=f"{spectrum.label} (scan {scan_index})", source_file=spectrum.source_file
    )
    add_kwargs = dict(
        xps=node["xps"][scan_index:scan_index + 1],
        energy_axis=node["energy_axis"],
        iter_axis=node["iter_axis"][scan_index:scan_index + 1],
        acq_time=node["acq_time"][scan_index:scan_index + 1],
        start_time=node["start_time"][scan_index:scan_index + 1],
        stop_time=node["stop_time"][scan_index:scan_index + 1],
    )
    # binding_energy/kinetic_energy are per-energy-point (not
    # per-scan), so they carry over unsliced, same as energy_axis —
    # optional here only for older/minimal test fixtures that predate
    # them (see scripts/import_h5.py, every real loader sets both).
    for key in ("binding_energy", "kinetic_energy"):
        if key in node:
            add_kwargs[key] = node[key]
    single.add_node("raw", meta=dict(node.meta), **add_kwargs)
    return single


def _xy_spectrum_label(spectrum: Spectrum, spectra: list[Spectrum]) -> str:
    """
    e.g. 'Si2p_c_4' from a .xy Region's own name and its Spectrum ID (as
    read from the file's "Spectrum ID" field) — that ID is expected to
    be unique within one file, but on the rare file where it isn't,
    disambiguate by appending this spectrum's occurrence index among
    the others sharing it (e.g. 'Si2p_c_4_2').

    `spectra` is every Spectrum loaded from the same file (for the
    uniqueness check); falls back to the file's stem if `spectrum`
    wasn't loaded by SpecsXYLoader (no region_name/spectrum_id meta).
    """
    meta = spectrum.nodes["raw"].meta
    region_name = meta.get("region_name")
    spectrum_id = meta.get("spectrum_id")
    if region_name is None or spectrum_id is None:
        return spectrum.source_file.stem

    base = f"{region_name}_{spectrum_id}"
    # Identity (`is`), not `==` — Spectrum is a dataclass whose default
    # __eq__ would compare the "raw" node's numpy arrays and raise
    # ("truth value of an array is ambiguous") on any real data.
    matches = [sp for sp in spectra if sp.nodes["raw"].meta.get("spectrum_id") == spectrum_id]
    if len(matches) <= 1:
        return base
    index = next(i for i, sp in enumerate(matches, start=1) if sp is spectrum)
    return f"{base}_{index}"


def _filter_node_by_time(node: Node, start: np.datetime64, stop: np.datetime64) -> Node:
    """Return a new Node with every array filtered to node['time'] within [start, stop]."""
    time = node["time"]
    mask = (time >= start) & (time <= stop)
    filtered_arrays = {name: values[mask] for name, values in node.arrays.items()}
    return Node(name=node.name, arrays=filtered_arrays, meta=node.meta)
