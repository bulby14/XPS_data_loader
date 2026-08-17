# XPS Importer

Tools for loading X-ray Photoelectron Spectroscopy (XPS) data exported from
Scienta PEAK (`.h5`) and SPECS Prodigy (`.xy`) software into simple Python
data structures for inspection and plotting.

## Install

```bash
conda env create -f environment.yml
conda activate XPS_data_loader
jupyter lab
```

## Contents

- `scripts/import_h5.py` — `XPSDataLoader`, which reads `.h5` files. It
  supports three formats depending on how the data was acquired:
  - `scienta_peak_h5` (default) — sequence files with pre-binned runs.
  - `scienta_peak_h5_snapshot` — single-acquisition snapshot files.
  - `scienta_peak_h5_events` — event-mode files, binned into (time, energy)
    histograms on load.
- `scripts/import_xy.py` / `scripts/xy_loader.py` — `parse_xy_file`, which
  reads `.xy` files into a `File → Group → Region → Scan` hierarchy (see
  `xy_import.ipynb` for details on what each level means).
- `h5_import.ipynb` / `xy_import.ipynb` — walkthrough notebooks demonstrating
  how to load and plot each file type, with explanations before each step.
- `utils/` — small shared helpers (e.g. type coercion for `.xy` metadata).
- `data/` — not included in this repo. Place your own `.h5`/`.xy` files here
  (the notebooks expect paths like `data/<filename>`) before running them.

## How it works

Both loaders parse the raw instrument export files and return in-memory
objects exposing the underlying arrays (intensity, energy axes, timestamps,
metadata) without modifying the source files. Start from the notebooks —
they load a sample file from `data/` and plot the result — then swap in
your own file path to explore new data.
