"""
Node-based loading architecture for XPS acquisition data.

Design
------
- Node          :   one pipeline stage's arrays + metadata for a single spectrum
                    (e.g. 'raw', 'bg_subtracted'). Nodes are never overwritten in
                    place, so any stage remains inspectable.
- Spectrum      :   one core-level run (e.g. O1s, C1s), tracked as an ordered
                    sequence of nodes.
- BaseXPSLoader :   interface every concrete file-format loader implements. 
                    Returns list[Spectrum], one entry per run found in the file.
- XPSDataLoader :   main entry point; dispatches to the right concrete loader by 
                    name, so multiple file formats can share one call site.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import h5py
from pathlib import Path
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Literal
import posixpath

from scripts.import_xy import parse_xy_file
from scripts.xy_loader import stack_region


# basic import structure of the h5 XPS files I have now (2026-07-20)

logger = logging.getLogger(__name__)

VALID_ENERGY_MODES = {"Binding", "Kinetic"}
EPOCH = np.datetime64("1970-01-01T00:00:00")

'''@dataclass
class XPSSequence:
    "Container for a loaded XPS acquisition sequence"
    xps: np.ndarray             # shape (n_loops, n_energy_points)
    energy_axis: np.ndarray     # binding or kinetic energy, shape (n_energy_points,)
    iter_axis: np.ndarray       # loop index, shape (n_loops,)
    acq_time: np.ndarray        # midpoint acquisition time, epoch seconds, shape (n_loops,)
    start_time: np.ndarray      # np.datetime64[s], shape (n_loops,)
    stop_time: np.ndarray       # np.datetime64[s], shape (n_loops,)'''

@dataclass
class Node:
    """
    A single pipeline stage's worth of data for one spectrum.
 
    Each stage (raw load, background subtraction, energy calibration, ...) writes its outputs into a
    new Node keyed by array name, rather than overwriting the previous stage's data — so every 
    intermediate result stays inspectable and plottable.
    """

    name: str
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> np.ndarray:
        return self.arrays[key]
    
    def __setitem__(self, key: str, value: np.ndarray) -> None:
        self.arrays[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self.arrays
    
@dataclass
class Spectrum:
    """
    One spectrum (a single core-level run, e.g. O1s, C1s, N1s) tracked
    through a pipeline of named Nodes.
 
    Parameters
    ----------
    label : str
        Identifier for this spectrum — typically the run name/core level
        (e.g. "O1s"). Used to distinguish spectra loaded from the same file.
    source_file : Path
        File this spectrum was loaded from, kept for provenance/debugging.
    """

    label: str
    source_file: Path
    nodes: dict[str, Node] = field(default_factory=dict)

    def add_node(self, name: str, meta: dict[str, Any] | None = None, **arrays: np.ndarray) -> Node:
        """Create and register a new Node, e.g. spectrum.add_node('raw', xps=...)"""
        node = Node(name=name, arrays=dict(arrays), meta=meta or {})
        self.nodes[name] = node
        return node
    
    def get(self, node_name: str, array_name: str) -> np.ndarray:
        return self.nodes[node_name][array_name]
    
    @property
    def latest(self) -> Node:
        """Most recent Node (assumes insertion order == 'pipeline' order)."""
        if not self.nodes:
            raise ValueError(f"Spectrum '{self.label}' has no nodes yet.")
        return next(reversed(self.nodes.values()))
    
    def __repr__(self) -> str:
        return f"Spectrum(label={self.label!r}, nodes={list(self.nodes)})"
        

class XPSPaths:
    SPECTRUM_DATA = "acquisition/spectrum/data/data"
    X_AXIS = "acquisition/spectrum/data/x_axis"
    ENERGY_MODE = "acquisition/spectrum_logs/spectrum_log_loop_0000/acquisition/spectrum_definition/energy_mode"
    EXCITATION_ENERGY = "acquisition/spectrum_logs/spectrum_log_loop_0000/instrument/analyser/excitation_source/energy"
    # Work function is a property of the run's calibration, not of each
    # individual loop, so (like EXCITATION_ENERGY above) this is only ever
    # read from loop 0000 regardless of how many loops the run has.
    WORK_FUNCTION = "acquisition/spectrum_logs/spectrum_log_loop_0000/instrument/analyser/work_function"
    SAMPLE_WORK_FUNCTION = "acquisition/spectrum_logs/spectrum_log_loop_0000/sample/work_function"
    USE_ANALYSER_WORKFUNCTION = "acquisition/spectrum_logs/spectrum_log_loop_0000/acquisition/energy_conversion/use_analyser_workfunction"
    USE_SAMPLE_WORKFUNCTION = "acquisition/spectrum_logs/spectrum_log_loop_0000/acquisition/energy_conversion/use_sample_workfunction"

    @staticmethod
    def loop_start_time(k: int) -> str:
        return f"acquisition/spectrum_logs/spectrum_log_loop_{k:04d}/acquisition/start_time"

    @staticmethod
    def loop_stop_time(k: int) -> str:
        return f"acquisition/spectrum_logs/spectrum_log_loop_{k:04d}/acquisition/stop_time"


class XPSSnapshotPaths:
    """
    Expected HDF5 paths for a single-spectrum "snapshot" file — one
    acquisition, no "shortcuts"/run grouping, so paths sit directly at
    the file root (like XPSEventPaths). energy_mode and the excitation
    energy live at different spots than in a sequence file's per-run
    layout, which is why this isn't just XPSPaths reused.
    """
    SPECTRUM_DATA = "acquisition/spectrum/data/data"
    X_AXIS = "acquisition/spectrum/data/x_axis"
    ENERGY_MODE = "acquisition/spectrum_definition/energy_mode"
    EXCITATION_ENERGY = "instrument/analyser/excitation_source/energy"
    WORK_FUNCTION = "instrument/analyser/work_function"
    SAMPLE_WORK_FUNCTION = "sample/work_function"
    USE_ANALYSER_WORKFUNCTION = "acquisition/spectrum_log/energy_conversion/use_analyser_workfunction"
    USE_SAMPLE_WORKFUNCTION = "acquisition/spectrum_log/energy_conversion/use_sample_workfunction"
    DWELL_TIME = "acquisition/spectrum_definition/dwell_time"
    START_TIME = "acquisition/spectrum_log/start_time"
    STOP_TIME = "acquisition/spectrum_log/stop_time"


def resolve_path(
        f: h5py.File, 
        run: str, 
        relative_path: str,
        on_ambiguous: Literal["warn", "raise"] = "warn") -> str:
    """
    Resolve a dataset/group path within a run, tolerating schema drift.

    Tries the expected path first (run/relative_path). If that doesn't
    exist, falls back to a recursive search within the run for any
    node whose path ends with the same trailing components as
    `relative_path` (i.e. the same "tail", regardless of what sits
    above it).

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle.
    run : str
        Name of the run group to search within (e.g. runs[0]).
    relative_path : str
        Expected path relative to `run`, e.g. "acquisition/spectrum/data/data".

    Returns
    -------
    str
        Full resolved path into `f`, ready to use as `f[path]`.

    Raises
    ------
    KeyError
        If no matching node is found, or matches are
        ambiguous even after the fallback search.
    """

    direct_path = posixpath.join(run, relative_path)
    if direct_path in f:
        return direct_path
    
    logger.warning(
        "Expected path '%s' not found - searching for a relocated match"
        "(tail: '%s')", direct_path, relative_path
    )

    matches: list[str] = []

    def visitor(name: str, obj) -> None:
        if name.endswith(relative_path):
            matches.append(name)

    search_root = f[run] if run else f
    search_root.visititems(visitor)

    if not matches:
        raise KeyError(
            f"Could not find '{relative_path}' anywhere under "
            f"{'run ' + repr(run) if run else 'file root'} "
            f"— schema may have changed beyond a simple relocation."
        )
    
    if len(matches) > 1:
        msg = f"Multiple candidates found for tail '{relative_path}': {matches}"
        if on_ambiguous == "raise":
            raise KeyError(msg)
        logger.warning("%s - using first: %s", msg, matches[0])

    resolved = posixpath.join(run, matches[0])
    logger.info("Resolved '%s' -> '%s'", relative_path, resolved)
    return resolved

def _decode_if_bytes(value: bytes | str) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _select_work_function(
        f: h5py.File,
        run: str,
        paths: type,
        datapath: Path) -> tuple[float, dict[str, Any]]:
    """
    Read the analyser and sample work functions plus the
    use_analyser_workfunction/use_sample_workfunction flags the
    acquisition software records for a run, and pick whichever one it
    actually applied — rather than assuming analyser is always used.

    Analyser work function is the standard choice (BE = hv - KE -
    phi_analyser relies only on sample and analyser sharing a common
    ground, not on the sample's own work function). Sample work
    function is used instead when the operator has determined the
    grounded-Fermi-level assumption doesn't hold well enough for this
    measurement (e.g. poorly-conductive/charging samples) and supplied
    a measured/estimated sample work function to use in its place.

    Returns (selected_work_function, metadata) where metadata records
    both raw values and which one was selected, for provenance.
    """
    Wf_analyser = float(f[resolve_path(f, run, paths.WORK_FUNCTION)][()])
    Wf_sample = float(f[resolve_path(f, run, paths.SAMPLE_WORK_FUNCTION)][()])
    use_analyser = bool(f[resolve_path(f, run, paths.USE_ANALYSER_WORKFUNCTION)][()])
    use_sample = bool(f[resolve_path(f, run, paths.USE_SAMPLE_WORKFUNCTION)][()])

    if use_analyser:
        source, Wf = "analyser", Wf_analyser
    elif use_sample:
        source, Wf = "sample", Wf_sample
    else:
        logger.warning(
            "Neither use_analyser_workfunction nor use_sample_workfunction is "
            "set for run '%s' in %s; defaulting to analyser work function.",
            run, datapath,
        )
        source, Wf = "analyser", Wf_analyser

    meta = {
        "analyser_work_function": Wf_analyser,
        "sample_work_function": Wf_sample,
        "work_function_source": source,
        "work_function": Wf,
    }
    return Wf, meta


def _load_xps_and_energy_axis(
        f: h5py.File,
        run: str,
        paths: type,
        datapath: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, dict[str, Any]]:
    """
    Read summed XPS counts and derive the binding/kinetic energy axes
    for one run/spectrum.

    Shared between loaders whose files store a single spectrum's counts,
    x_axis, energy_mode and excitation energy the same way — they only
    disagree on where the run sits and where energy_mode/Eph/work
    function live within it (see XPSPaths vs XPSSnapshotPaths), which
    `paths` supplies.

    Returns (xps, energy_axis, binding_energy, kinetic_energy, en_mode,
    meta). `energy_axis` is kept as an alias of `binding_energy` (its
    historical meaning) for callers that only need one axis; meta
    carries the excitation energy and work-function provenance (see
    `_select_work_function`).
    """
    xps0 = f[resolve_path(f, run, paths.SPECTRUM_DATA)][()]
    xs = f[resolve_path(f, run, paths.X_AXIS)][()]

    xps = np.sum(xps0, axis=1)

    en_mode_raw = f[resolve_path(f, run, paths.ENERGY_MODE)][()]
    en_mode = _decode_if_bytes(en_mode_raw)
    if en_mode not in VALID_ENERGY_MODES:
        raise ValueError(
            f"Unexpected energy_mode {en_mode!r} for run '{run}' in {datapath};"
            f"expected on eof {VALID_ENERGY_MODES}"
        )

    Eph = float(f[resolve_path(f, run, paths.EXCITATION_ENERGY)][()])
    Wf, wf_meta = _select_work_function(f, run, paths, datapath)

    if en_mode == "Binding":
        binding_energy = xs
        kinetic_energy = Eph - Wf - xs
    else:
        kinetic_energy = xs
        binding_energy = Eph - Wf - xs
    energy_axis = binding_energy

    meta = {"excitation_energy": Eph, **wf_meta}

    return xps, energy_axis, binding_energy, kinetic_energy, en_mode, meta

class BaseXPSLoader(ABC):
    """Interface every concrete file-format loader must implement."""

    name: str = "base"

    @abstractmethod
    def load(self, datapath: str | Path) -> list[Spectrum]:
        """
        Load all spectra found in 'datapath'.

        Returns
        -------
        list[Spectrum]
            One spectrum per run/core-level found in the file,
            each with (at minimum) a 'raw' Node populated.
        """
        raise NotImplementedError
    
class XPSDataLoader:
    """
    Main entry point for loading XPS data, dispatching to the
    appropriate concrete loader by name.
 
    Examples
    --------
    >>> loader = XPSDataLoader()
    >>> spectra = loader.load("sequence.h5")  # uses default_loader
    >>> spectra = loader.load("sequence.h5", loader_name="scienta_peak_h5")
    """

    def __init__(self, default_loader: str = "scienta_peak_h5") -> None:
        self._loaders: dict[str, BaseXPSLoader] = {}
        self.default_loader = default_loader
        self.register(ScientaPeakLoader())
        self.register(ScientaPeakEventLoader())
        self.register(ScientaPeakSnapshotLoader())
        self.register(SpecsXYLoader())

    def register(self, loader: BaseXPSLoader) -> None:
        if loader.name in self._loaders:
            logger.warning("Overwriting already-registered loader '%s'", loader.name)
        self._loaders[loader.name] = loader

    def available_loaders(self) -> list[str]:
        return list(self._loaders)
    
    def load(self, datapath: str | Path, loader_name: str | None = None, **kwargs: Any) -> list[Spectrum]:
        name = loader_name or self.default_loader
        if name not in self._loaders:
            raise ValueError(
                f"Unknown loader '{name}'; available: {self.available_loaders()}"
            )
        return self._loaders[name].load(datapath, **kwargs)


class ScientaPeakLoader(BaseXPSLoader):
    """Loader for .h5 sequence files generated by the Scienta Peak Software - maybe."""

    name = "scienta_peak_h5"

    def load(self, datapath: str | Path) -> list[Spectrum]:
        """
        Load every run (core-level spectrum) found in a Scienta PEAK
        HDF5 sequence file.
 
        Parameters
        ----------
        datapath : str or Path
            Path to the .h5 sequence file.
 
        Returns
        -------
        list[Spectrum]
            One Spectrum per run (e.g. O1s, C1s, N1s), each with a
            "raw" Node containing: xps, energy_axis, iter_axis,
            acq_time, start_time, stop_time.
 
        Raises
        ------
        FileNotFoundError
            If `datapath` does not exist.
        KeyError
            If expected HDF5 groups/datasets are missing, even after
            attempting to resolve relocated paths.
        ValueError
            If `energy_mode` has an unexpected value, or no runs are found.
        """

        datapath = Path(datapath)
        if not datapath.exists():
            raise FileNotFoundError(f".h5 file not found {datapath}")
        if datapath.suffix.lower() != ".h5":
            raise ValueError(f"Expected a .h5 file, got {datapath.suffix} instead")
        
        logger.info("Loading XPS sequence from '%s'", datapath)

        spectra: list[Spectrum] = []

        with h5py.File(datapath, 'r') as f:
            if "shortcuts" not in f:
                raise KeyError(
                    f"'shortcuts' group missing - is {datapath} a valid file?"
                )
            
            runs = list(f["shortcuts"].keys())

            if not runs:
                raise ValueError(f"No runs found under 'shortcuts' in {datapath}")
            
            logger.info(f"Found {len(runs)} run(s) in {datapath}: {runs}")

            for run in runs:
                spectra.append(self._load_run(f, run, datapath))
        
        return spectra
    
    def _load_run(self, f: h5py.File, run: str, datapath: Path) -> Spectrum:
        """Load a single run/core-level into a Spectrum with a 'raw' Node."""
        logger.debug("Loading run '%s'", run)

        xps, energy_axis, binding_energy, kinetic_energy, en_mode, energy_meta = (
            _load_xps_and_energy_axis(f, run, XPSPaths, datapath)
        )
        no = xps.shape[0]

        start_time = np.zeros(no)
        stop_time = np.zeros(no)
        t1_list: list[np.datetime64] = []
        t2_list: list[np.datetime64] = []

        for k in range(no):
            t1_raw = f[resolve_path(f, run, XPSPaths.loop_start_time(k))][()]
            t2_raw = f[resolve_path(f, run, XPSPaths.loop_stop_time(k))][()]

            t1_dt64 = np.datetime64(_decode_if_bytes(t1_raw))
            t2_dt64 = np.datetime64(_decode_if_bytes(t2_raw))

            start_time[k] = (t1_dt64 - EPOCH) / np.timedelta64(1, "s")
            stop_time[k] = (t2_dt64 - EPOCH) / np.timedelta64(1, "s")

            t1_list.append(t1_dt64)
            t2_list.append(t2_dt64)

        t1_list = np.array(t1_list, dtype=np.datetime64)
        t2_list = np.array(t2_list, dtype=np.datetime64)

        assert start_time.shape == stop_time.shape == (no, ), (
            "start_time/stop_time length missmatch - internal loop bug..."
        )

        iter_axis = np.arange(no)
        acq_time = (start_time + stop_time) / 2

        spectrum = Spectrum(label=run, source_file=datapath)
        spectrum.add_node(
            "raw",
            meta=energy_meta,
            xps=xps,
            energy_axis=energy_axis,
            binding_energy=binding_energy,
            kinetic_energy=kinetic_energy,
            iter_axis=iter_axis,
            acq_time=acq_time,
            start_time=t1_list,
            stop_time=t2_list,
        )

        logger.info(
            "Loaded run '%s': %d loop(s), energy_mode=%s", run, no, en_mode
        )

        return spectrum


class ScientaPeakSnapshotLoader(BaseXPSLoader):
    """
    Loader for single-spectrum "snapshot" .h5 files generated by the
    Scienta Peak software.

    Unlike a sequence file, these hold exactly one acquisition (no
    "shortcuts"/run grouping, no separately-timestamped repeated
    loops) — but that one acquisition still has a rows-of-energy-scans
    "iterations" axis, evenly spaced by dwell_time across the single
    acquisition window rather than each iteration having its own
    logged start/stop time. Per-iteration timestamps are synthesized
    from the single acquisition's start_time + iteration_index * dwell_time.
    """

    name = "scienta_peak_h5_snapshot"

    def load(self, datapath: str | Path) -> list[Spectrum]:
        """
        Load the single spectrum found in a Scienta PEAK snapshot file.

        Parameters
        ----------
        datapath : str or Path
            Path to the .h5 snapshot file.

        Returns
        -------
        list[Spectrum]
            A single-element list containing one Spectrum, with a "raw"
            Node containing: xps, energy_axis, iter_axis, acq_time,
            start_time, stop_time (same fields as ScientaPeakLoader's,
            for downstream compatibility).

        Raises
        ------
        FileNotFoundError
            If `datapath` does not exist.
        KeyError
            If expected HDF5 datasets are missing, even after
            attempting to resolve relocated paths.
        ValueError
            If `energy_mode` has an unexpected value.
        """

        datapath = Path(datapath)
        if not datapath.exists():
            raise FileNotFoundError(f".h5 file not found {datapath}")
        if datapath.suffix.lower() != ".h5":
            raise ValueError(f"Expected a .h5 file, got {datapath.suffix} instead")

        logger.info("Loading XPS snapshot from '%s'", datapath)

        with h5py.File(datapath, "r") as f:
            xps, energy_axis, binding_energy, kinetic_energy, en_mode, energy_meta = (
                _load_xps_and_energy_axis(f, "", XPSSnapshotPaths, datapath)
            )
            no = xps.shape[0]

            dwell_time = float(f[resolve_path(f, "", XPSSnapshotPaths.DWELL_TIME)][()])

            start_raw = f[resolve_path(f, "", XPSSnapshotPaths.START_TIME)][()]
            start_dt64 = np.datetime64(_decode_if_bytes(start_raw), "ns")

        iter_axis = np.arange(no)
        offsets = (iter_axis * dwell_time * 1e9).astype("int64").astype("timedelta64[ns]")
        dwell_ns = np.timedelta64(int(round(dwell_time * 1e9)), "ns")

        start_time = start_dt64 + offsets
        stop_time = start_time + dwell_ns
        acq_time = (start_time.astype("datetime64[ns]") - EPOCH) / np.timedelta64(1, "s") + dwell_time / 2

        spectrum = Spectrum(label=datapath.stem, source_file=datapath)
        spectrum.add_node(
            "raw",
            meta={"dwell_time": dwell_time, **energy_meta},
            xps=xps,
            energy_axis=energy_axis,
            binding_energy=binding_energy,
            kinetic_energy=kinetic_energy,
            iter_axis=iter_axis,
            acq_time=acq_time,
            start_time=start_time,
            stop_time=stop_time,
        )

        logger.info(
            "Loaded snapshot '%s': %d iteration(s), energy_mode=%s", datapath.stem, no, en_mode
        )

        return [spectrum]


class SpecsXYLoader(BaseXPSLoader):
    """
    Loader for SPECS Prodigy .xy export files (see xy_loader.py and
    import_xy.py for the format's own parsing/stacking logic).

    Unlike the h5 loaders, one .xy file can hold many Group/Region
    blocks of different kinds (single spectrum, repeated series,
    positional map) side by side — each Region becomes its own
    Spectrum, mirroring how ScientaPeakLoader turns each h5 run into
    its own Spectrum.
    """

    name = "specs_xy"

    def load(self, datapath: str | Path) -> list[Spectrum]:
        """
        Load every Region found in a SPECS Prodigy .xy export file.

        Returns
        -------
        list[Spectrum]
            One Spectrum per Region, each with a "raw" Node containing:
            xps, energy_axis, iter_axis, acq_time, start_time,
            stop_time (same fields as the h5 loaders', for downstream
            compatibility) — plus position_y/position_z/position_step
            for "map" regions (position-resolved acquisitions).

        Raises
        ------
        FileNotFoundError
            If `datapath` does not exist.
        ValueError
            If `datapath` isn't a .xy file, or every region in the
            file failed to parse/stack.
        """
        datapath = Path(datapath)
        if not datapath.exists():
            raise FileNotFoundError(f".xy file not found: {datapath}")
        if datapath.suffix.lower() != ".xy":
            raise ValueError(f"Expected a .xy file, got {datapath.suffix} instead")

        logger.info("Loading XPS spectra from '%s'", datapath)

        xyfile = parse_xy_file(datapath)
        if not xyfile.regions:
            raise ValueError(f"No regions found in {datapath}")

        spectra: list[Spectrum] = []
        for region in xyfile.regions:
            if not region.scans:
                logger.warning("Skipping region '%s': no scans", region.name)
                continue
            try:
                stacked = stack_region(region, xyfile.file_metadata)
            except ValueError as e:
                logger.warning("Skipping region '%s': %s", region.name, e)
                continue

            label = f"{region.group} / {region.name}_{region.spectrum_id}"
            spectrum = Spectrum(label=label, source_file=datapath)

            arrays: dict[str, np.ndarray] = dict(
                xps=stacked.xps,
                energy_axis=stacked.energy_axis,
                binding_energy=stacked.binding_energy,
                kinetic_energy=stacked.kinetic_energy,
                iter_axis=stacked.iter_axis,
                acq_time=stacked.acq_time,
                start_time=stacked.start_time,
                stop_time=stacked.stop_time,
            )
            if region.kind == "map":
                arrays["position_y"] = stacked.position_y
                arrays["position_z"] = stacked.position_z
                arrays["position_step"] = stacked.position_step

            spectrum.add_node(
                "raw",
                meta={
                    "kind": region.kind,
                    "group": region.group,
                    "region_name": region.name,
                    "scan_mode": region.scan_mode,
                    "spectrum_id": region.spectrum_id,
                    "native_energy_mode": stacked.native_energy_mode,
                    "discarded_scans": stacked.discarded_scans,
                    "region_metadata": region.metadata,
                    "work_function": stacked.work_function,
                    # Top-level, like the h5 loaders' "excitation_energy" —
                    # None if this region's header omitted it (see
                    # xy_loader.py's binding<->kinetic conversion, which
                    # already tolerates that same absence).
                    "excitation_energy": region.metadata.get("Excitation Energy"),
                },
                **arrays,
            )
            spectra.append(spectrum)

            logger.info(
                "Loaded region '%s' (kind=%s): %d scan(s) kept, %d discarded",
                region.name, region.kind, len(stacked.iter_axis), len(stacked.discarded_scans),
            )

        if not spectra:
            raise ValueError(f"No usable regions found in {datapath}")

        return spectra


def load_auxiliary_log(
        logpath: str | Path,
        column_names: list[str] | None = None
) -> Node:
    """
    Load a tab-delimited auxiliary time-series log (e.g. temperature,
    pressure channels logged alongside an XPS acquisition) into a Node.
 
    Expects a header row followed by rows of: ISO timestamp, then N
    tab-separated numeric columns.
 
    Parameters
    ----------
    logpath : str or Path
        Path to the .txt log file.
    column_names : list[str] or dict[str, str], optional
        Short names for the numeric columns. Two forms:
 
        - list[str]: positional — the i-th name is assigned to the
          i-th numeric column, in file order. Must match the file's
          column count exactly.
        - dict[str, str]: mapping of {full_column_name_in_file: short_name}.
          Safer when full names are long/collision-prone (e.g. Tango
          device paths), since you match by the actual name rather
          than by position — a column reordering in the file won't
          silently mislabel your data. Columns not present in the
          dict keep their original full name.
 
        If omitted entirely, the file's own header names are used as-is.
        Regardless of which form is used, the original full column
        names are always preserved in the returned Node's
        `meta["original_columns"]` as {short_name: full_name}.
 
    Returns
    -------
    Node
        A Node named "auxiliary_log" with arrays: "time" (datetime64[s])
        plus one array per numeric column, named per `column_names`
        (or the file's own header if not given). `meta["original_columns"]`
        maps each array name back to the full name found in the file.
 
    Raises
    ------
    FileNotFoundError
        If `logpath` does not exist.
    ValueError
        If `column_names` is a list whose length doesn't match the
        number of numeric columns found in the file, or a dict that
        references a column name not present in the file's header.
    """

    logpath = Path(logpath)
    if not logpath.exists():
        raise FileNotFoundError(f"Auxiliary log file not found: {logpath}")
    
    logger.info("Loading auxiliary log from '%s'", logpath)

    with open(logpath, "r", encoding="utf-8") as fh:
        header_line = fh.readline().strip()
    file_column_names = header_line.split("\t")[1:] # skipping the first column = timestamp

    # Archiver dumps occasionally log a dropped/unavailable sample as
    # the literal string "None" (on top of the "nan" genfromtxt already
    # understands) — left to its own numeric-dtype inference, a single
    # "None" anywhere in a column silently turns that *entire* column
    # into strings instead of floats, breaking any later plotting/math
    # on it. Converting both to NaN up front keeps every numeric column
    # as float64 regardless.
    def _parse_numeric(value: str) -> float:
        value = value.strip()
        return np.nan if value in ("None", "nan", "NaN", "") else float(value)

    converters = {0: lambda s: np.datetime64(s.strip())}
    converters.update({i: _parse_numeric for i in range(1, len(file_column_names) + 1)})

    metadata = np.genfromtxt(
        logpath,
        skip_header=1,
        delimiter="\t",
        dtype=None,
        encoding="utf-8",
        converters=converters,
    )

    time_data = metadata["f0"].astype("datetime64[s]")
    n_numeric_cols = len(metadata.dtype.names) - 1
    meta_values = np.column_stack(
        [metadata[f"f{i}"] for i in range(1, n_numeric_cols + 1)]
    )

    if column_names is None:
        short_names = list(file_column_names)

    elif isinstance(column_names, dict):
        unknown = set(column_names) - set(file_column_names)
        if unknown:
            raise ValueError(
                f"column_names dict references column(s) not found in {logpath}:"
                f"{unknown}. File header was: {file_column_names}"
            )
        short_names = [column_names.get(full, full) for full in file_column_names]

    else:
        if len(column_names) != n_numeric_cols:
            raise ValueError(
                f"column_names has {len(column_names)} entries but {logpath} "
                f"has {n_numeric_cols} numeric columns — "
                f"header was: {file_column_names}"
            )
        short_names = list(column_names)

    original_columns = dict(zip(short_names, file_column_names))

    arrays = {"time": time_data}
    for i, name in enumerate(short_names):
        arrays[name] = meta_values[:, i]

    logger.info(
        "Loaded auxiliary log: %d samples, columns=%s", len(time_data), short_names
    )

    return Node(name="auxiliary_log", arrays=arrays, meta={"source_file": logpath, "original_columns": original_columns})


# ---------------------------------------------------------------------------
# Concrete loader: Scienta PEAK HDF5 event-mode files
# ---------------------------------------------------------------------------
 
#: ts/exposure_time in the event files are stored in picoseconds;
#: bin sizes/ranges below are specified in seconds and converted internally.
TIME_UNIT_SCALE = 1e12
 
 
class XPSEventPaths:
    """Expected HDF5 paths for a Scienta PEAK event-mode file (no run grouping)."""
 
    TIMESTAMPS = "acquisition/spectrum_events/events/ts"
    KINETIC_OFFSET = "acquisition/spectrum_events/events/xs"
    EXPOSURE_TIME = "acquisition/spectrum_events/events/exposure_time"
    EXCITATION_ENERGY = "instrument/analyser/excitation_source/energy"
    WORK_FUNCTION = "instrument/analyser/work_function"
    SAMPLE_WORK_FUNCTION = "sample/work_function"
    USE_ANALYSER_WORKFUNCTION = "acquisition/spectrum_log/energy_conversion/use_analyser_workfunction"
    USE_SAMPLE_WORKFUNCTION = "acquisition/spectrum_log/energy_conversion/use_sample_workfunction"
    DWELL_TIME = "acquisition/spectrum_definition/dwell_time"
    START_TIME = "acquisition/spectrum_log/start_time"
    STOP_TIME = "acquisition/spectrum_log/stop_time"
 
 
def _compute_bin_edges(
    data: np.ndarray,
    bin_size: float,
    data_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Compute histogram bin edges for `data`, clipped to the data's own extent.
 
    If `data_range` is given but extends beyond what's actually present
    in `data`, it's clipped inward to the data's min/max (mirrors the
    original loader's behavior: never bin outside the range of real
    events). If `bin_size` is zero, non-positive, or larger than the
    full range, a single bin covering the whole range is returned.
    """
    data_min, data_max = float(np.min(data)), float(np.max(data))
    if data_range is None:
        lo, hi = data_min, data_max
    else:
        lo = max(data_range[0], data_min)
        hi = min(data_range[1], data_max)
 
    if bin_size <= 0 or bin_size >= (hi - lo):
        return np.array([lo, hi])
    return np.arange(lo, hi, bin_size)
 
 
class ScientaPeakEventLoader(BaseXPSLoader):
    """
    Loader for .h5 event-mode files generated by Scienta PEAK software.
 
    Unlike the sequence loader, event-mode files record individual
    detection events (timestamp + kinetic-energy-offset pairs) rather
    than pre-binned spectra, and have no "shortcuts"/run grouping —
    paths sit directly under the file root. This loader bins events
    into a 2D (time, energy) histogram.
    """
 
    name = "scienta_peak_h5_events"
 
    def load(
        self,
        datapath: str | Path,
        energy_bin_size: float,
        energy_range: tuple[float, float] | None = None,
        time_bin_size: float | None = None,
        time_range: tuple[float, float] | None = None,
        exposure_margin: float = 1.1,
        progress_callback: Callable[[int], None] | None = None,
    ) -> list[Spectrum]:
        """
        Load and bin event-mode XPS data.

        Parameters
        ----------
        datapath : str or Path
            Path to the .h5 event-mode file.
        energy_bin_size : float
            Energy bin width in eV.
        energy_range : (float, float), optional
            (min, max) energy in eV to bin over. Clipped to the actual
            data extent if wider than the real event range. Defaults
            to the full extent of the recorded events.
        time_bin_size : float, optional
            Time bin width in seconds. Defaults to dwell_time / 100
            (~100 time bins). Pass 0 explicitly for a single bin
            covering the entire exposure (i.e. a plain energy spectrum
            with no time resolution).
        time_range : (float, float), optional
            (min, max) time in seconds to bin over. Same clipping
            behavior as `energy_range`. Defaults to (0, dwell_time),
            with dwell_time read from the file's spectrum_definition
            (the raw per-event timestamps are not reliably usable for
            this on their own — e.g. they can wrap around well before
            the nominal dwell time on longer acquisitions).
        exposure_margin : float, default 1.1
            Events with timestamps beyond `exposure_time * exposure_margin`
            are discarded as acquisition artifacts, matching the
            original implementation's 10% margin.
        progress_callback : Callable[[int], None], optional
            Called periodically with a 0-100 completion percentage
            while binning — intended for a background-thread caller to
            report progress back to a UI. The histogramming step (by
            far the slowest part for large files) is done in chunks
            specifically to make this possible; the earlier read/filter
            steps are comparatively fast and aren't broken down further.

        Returns
        -------
        list[Spectrum]
            A single-element list containing one Spectrum (event-mode
            files hold one dataset, not multiple runs), with a "raw"
            Node containing: xps (2D array, shape (n_time_bins,
            n_energy_bins)), energy_axis, time_axis, plus single-element
            start_time/stop_time arrays (the whole acquisition's overall
            start/stop, read from spectrum_log — not per-time-bin) so
            downstream code that keys off a spectrum's overall
            acquisition window (e.g. the metadata/archiver-log plot,
            see utils.spectrum_utils._spectrum_time_window) works the
            same as it does for the other loaders.
 
        Raises
        ------
        FileNotFoundError
            If `datapath` does not exist.
        KeyError
            If expected HDF5 datasets are missing, even after
            attempting to resolve relocated paths.
        """
        datapath = Path(datapath)
        if not datapath.exists():
            raise FileNotFoundError(f"H5 file not found: {datapath}")
        if datapath.suffix.lower() != ".h5":
            raise ValueError(f"Expected a .h5 file, got: {datapath.suffix}")
 
        logger.info("Loading XPS event data from %s", datapath)
 
        with h5py.File(datapath, "r") as f:
            ts = f[resolve_path(f, "", XPSEventPaths.TIMESTAMPS)][()]
            xs = f[resolve_path(f, "", XPSEventPaths.KINETIC_OFFSET)][()]
            Eph = float(f[resolve_path(f, "", XPSEventPaths.EXCITATION_ENERGY)][()])
            Wf, wf_meta = _select_work_function(f, "", XPSEventPaths, datapath)
            exptime = f[resolve_path(f, "", XPSEventPaths.EXPOSURE_TIME)][()]
            dwell_time = f[resolve_path(f, "", XPSEventPaths.DWELL_TIME)][()]
            start_raw = f[resolve_path(f, "", XPSEventPaths.START_TIME)][()]
            stop_raw = f[resolve_path(f, "", XPSEventPaths.STOP_TIME)][()]

        logger.info("Loaded %d raw events; filtering to exposure window", len(ts))

        valid = ts < exptime * exposure_margin * TIME_UNIT_SCALE
        ts = ts[valid]
        xs = xs[valid]

        # Kinetic energy is always recorded for events (no energy_mode switch
        # needed here, unlike the sequence loader) — xs (each event's raw
        # per-event energy, from KINETIC_OFFSET) is already kinetic, so it's
        # used directly rather than transformed. Confirmed against real
        # data: xs clusters right next to Eph (as kinetic energy should),
        # while Eph - Wf - xs lands in a physically sensible Binding Energy
        # range for the scan. An earlier version of this loader applied the
        # binding<->kinetic conversion here regardless, which silently
        # swapped binding_energy and kinetic_energy for every event-mode
        # file — invisible while the app only ever read one combined
        # energy_axis (which happened to land on the right numbers via a
        # second, compensating swap below), but wrong once binding_energy/
        # kinetic_energy became separately exposed.
        kinetic_energy = xs

        if time_range is None:
            # The raw per-event timestamps aren't reliable as a source
            # for the overall time span on their own (observed to wrap
            # around well before the nominal dwell time on some longer
            # acquisitions) — dwell_time is the authoritative duration.
            time_range = (0.0, float(dwell_time))
        if time_bin_size is None:
            time_bin_size = float(dwell_time) / 100
        time_range_ps = (time_range[0] * TIME_UNIT_SCALE, time_range[1] * TIME_UNIT_SCALE)
        energy_edges = _compute_bin_edges(kinetic_energy, energy_bin_size, energy_range)
        time_edges = _compute_bin_edges(ts, time_bin_size * TIME_UNIT_SCALE, time_range_ps)
 
        logger.info(
            "Binning energy: %.4f to %.4f eV, step %.4f eV",
            energy_edges[0], energy_edges[-1], energy_bin_size,
        )
        logger.info(
            "Binning time: %.4f to %.4f s, step %.4f s",
            time_edges[0] / TIME_UNIT_SCALE, time_edges[-1] / TIME_UNIT_SCALE, time_bin_size,
        )
 
        if progress_callback is not None:
            progress_callback(5)

        # Binned in chunks (rather than one np.histogram2d call over the
        # full arrays) purely so progress_callback has something to
        # report between — histogramming per-chunk and summing the
        # partial 2D histograms gives the exact same result, since bin
        # edges are fixed and identical across chunks.
        n_chunks = min(50, max(1, len(ts) // 500_000))
        data_binned = np.zeros((len(time_edges) - 1, len(energy_edges) - 1))
        for i, (ts_chunk, energy_chunk) in enumerate(
            zip(np.array_split(ts, n_chunks), np.array_split(kinetic_energy, n_chunks))
        ):
            partial, _, _ = np.histogram2d(ts_chunk, energy_chunk, bins=[time_edges, energy_edges])
            data_binned += partial
            if progress_callback is not None:
                progress_callback(5 + int(95 * (i + 1) / n_chunks))

        # binding_energy is derived from the same (now correctly kinetic)
        # midpoints via the same affine map used per-event above.
        # energy_axis = binding_energy_axis, same as the other three
        # loaders (see module-level convention notes on Node/Spectrum).
        kinetic_energy_axis = (energy_edges[:-1] + energy_edges[1:]) / 2
        binding_energy_axis = Eph - Wf - kinetic_energy_axis
        energy_axis = binding_energy_axis
        time_axis = (time_edges[:-1] + time_edges[1:]) / 2 / TIME_UNIT_SCALE

        start_dt64 = np.datetime64(_decode_if_bytes(start_raw))
        stop_dt64 = np.datetime64(_decode_if_bytes(stop_raw))

        spectrum = Spectrum(label=datapath.stem, source_file=datapath)
        spectrum.add_node(
            "raw",
            meta={"dwell_time": float(dwell_time), "excitation_energy": Eph, **wf_meta},
            xps=data_binned,
            energy_axis=energy_axis,
            binding_energy=binding_energy_axis,
            kinetic_energy=kinetic_energy_axis,
            time_axis=time_axis,
            start_time=np.array([start_dt64]),
            stop_time=np.array([stop_dt64]),
        )
 
        logger.info("Event binning complete: xps shape=%s", data_binned.shape)
        return [spectrum]