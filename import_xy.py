"""
Parser for SPECS Prodigy .xy XPS export files.
 
Full hierarchy (this file revealed a level the single-region example didn't have):
 
    File
      -> Group session   ("# Group: <name>" + Scan Mode, calibration/lens settings,
                            Source Parameters. Recurs throughout the file -- every
                            time the operator switches experiment/measurement context
                            in SpecsLab Prodigy, a new Group block is written, and every
                            Region that follows belongs to that Group until the next one.)
          -> Region       ("# Region: <name>" + Spectrum ID + acquisition settings)
              -> Scan     ("# Cycle: n[, Curve: m], Scan: k" + optional "# Parameter: ..."
                            lines carrying e.g. manipulator Step/Y/Z position)
                  -> energy, intensity arrays
 
A Region's *kind* (single spectrum / repeated series / positional map) follows directly
from what's inside it:
  - one Scan, no Parameters                -> "single"  (one-off spectrum)
  - >1 Scan, no Parameters                 -> "series"  (repeat scans of the same
                                               acquisition -- averaging or time-tracking)
  - any Scan carries Parameters (Y/Z/Step) -> "map"      (position-resolved, i.e. the 2D data)
"""
 
import re
from dataclasses import dataclass, field
import numpy as np
from utils.utils import _smart_cast
 
_GROUP_RE = re.compile(r'^#\s*Group:\s*(.*)$')
_REGION_RE = re.compile(r'^#\s*Region:\s*(.*)$')
_CYCLE_RE = re.compile(r'^#\s*Cycle:\s*(\d+)\s*$')
# "Scan" and "Channel" are the same concept under two different export-setting names
# (SpecsLab uses "Channel" when "Separate Channel Data" is enabled instead of "Separate Scan Data").
_SCAN_HDR_RE = re.compile(r'^#\s*Cycle:\s*(\d+)\s*(?:,\s*Curve:\s*(\d+))?\s*,\s*(?:Scan|Channel):\s*(\d+)\s*$')
_PARAM_RE = re.compile(r'^#\s*Parameter:\s*"([^"]+)"\s*=\s*(.*)$')
_KV_RE = re.compile(r'^#\s*([^:]+?):\s*(.*)$')
 
 
@dataclass
class Scan:
    scan_index: int
    cycle_index: int
    metadata: dict            # e.g. Acquisition Date
    parameters: dict          # e.g. {'Step': 1, 'Y [mm]': 2.0, 'Z [mm]': -2.1}
    energy: np.ndarray
    intensity: np.ndarray
 
    def __len__(self):
        return len(self.energy)
 
    def __repr__(self):
        p = f", params={self.parameters}" if self.parameters else ""
        return f"Scan(scan={self.scan_index}, n_points={len(self.energy)}{p})"
 
 
@dataclass
class Region:
    spectrum_id: int
    name: str
    group: str
    scan_mode: str
    metadata: dict
    scans: list = field(default_factory=list)
 
    @property
    def kind(self):
        if any(s.parameters for s in self.scans):
            return "map"
        if len(self.scans) > 1:
            return "series"
        return "single"
 
    def __repr__(self):
        return (f"Region(id={self.spectrum_id}, name={self.name!r}, group={self.group!r}, "
                f"kind={self.kind!r}, n_scans={len(self.scans)})")
 
 
@dataclass
class GroupSession:
    name: str
    metadata: dict
 
 
@dataclass
class XYFile:
    file_metadata: dict
    groups: list = field(default_factory=list)
    regions: list = field(default_factory=list)
 
    def by_kind(self):
        out = {"single": [], "series": [], "map": []}
        for r in self.regions:
            out[r.kind].append(r)
        return out
 
    def __repr__(self):
        return f"XYFile(n_groups={len(self.groups)}, n_regions={len(self.regions)})"
 
 
def _try_parse_data_line(line: str):
    parts = line.split()
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None
 
 
def parse_xy_file(path, encoding="latin-1") -> XYFile:
    file_metadata = {}
    groups, regions = [], []
 
    current_group = None
    current_region = None
    current_scan_header = None
    current_scan_meta = {}
    current_cycle_params = {}   # belongs to the *Cycle*, shared by every Scan/Channel inside it
    energies, intensities = [], []
 
    def flush_scan():
        nonlocal energies, intensities, current_scan_header, current_scan_meta
        if current_scan_header is not None and energies:
            cyc, curv, scn = current_scan_header
            current_region.scans.append(Scan(
                scan_index=scn, cycle_index=cyc,
                metadata=dict(current_scan_meta), parameters=dict(current_cycle_params),
                energy=np.array(energies), intensity=np.array(intensities),
            ))
        energies, intensities = [], []
        current_scan_header = None
        current_scan_meta = {}
 
    with open(path, encoding=encoding) as f:
        for raw_line in f:
            line = raw_line.rstrip("\r\n")
            stripped = line.strip()
            if not stripped:
                continue
 
            if stripped.startswith("#"):
                m = _GROUP_RE.match(stripped)
                if m:
                    flush_scan()
                    current_group = GroupSession(name=m.group(1).strip(), metadata={})
                    groups.append(current_group)
                    current_region = None
                    continue
 
                m = _REGION_RE.match(stripped)
                if m:
                    flush_scan()
                    current_region = Region(
                        spectrum_id=-1, name=m.group(1).strip(),
                        group=current_group.name if current_group else None,
                        scan_mode=(current_group.metadata.get("Scan Mode") if current_group else None),
                        metadata={},
                    )
                    regions.append(current_region)
                    continue
 
                m = _CYCLE_RE.match(stripped)
                if m:
                    flush_scan()               # a new Cycle always closes out whatever came before
                    current_cycle_params = {}  # fresh position bucket for this Cycle
                    continue
 
                m = _SCAN_HDR_RE.match(stripped)
                if m:
                    flush_scan()
                    cyc = int(m.group(1))
                    curv = int(m.group(2)) if m.group(2) is not None else 0
                    scn = int(m.group(3))
                    current_scan_header = (cyc, curv, scn)
                    continue
 
                m = _PARAM_RE.match(stripped)
                if m and current_region is not None:
                    current_cycle_params[m.group(1)] = _smart_cast(m.group(2).strip())
                    continue
 
                m = _KV_RE.match(stripped)
                if m:
                    key, value = m.group(1).strip(), _smart_cast(m.group(2).strip())
                    if current_scan_header is not None:
                        current_scan_meta[key] = value
                    elif current_region is not None:
                        current_region.metadata[key] = value
                        if key == "Spectrum ID":
                            current_region.spectrum_id = value
                    elif current_group is not None:
                        current_group.metadata[key] = value
                    else:
                        file_metadata[key] = value
                continue
 
            data = _try_parse_data_line(stripped)
            if data:
                energies.append(data[0])
                intensities.append(data[1])
 
    flush_scan()
    return XYFile(file_metadata=file_metadata, groups=groups, regions=regions)