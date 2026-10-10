"""High-quality 3D: COLMAP structure-from-motion on many anonymised video frames, then Gaussian splats (Brush).

Thin CLI over src/avtopassport/recon3d/hq.py (the same code the API's optional AVTO_HQ3D=1 stage runs).

Stages (run from the repo root):
    .venv-3d/bin/python scripts/recon3d_hq.py sfm   --dataset outputs/recon3d/hq   # needs images/ (anonymised)
    .venv/bin/python   scripts/recon3d_hq.py brush --dataset outputs/recon3d/hq --steps 30000
    .venv/bin/python   scripts/recon3d_hq.py all   --dataset outputs/recon3d/hq2 --frames-dir <blurred frames>
Inputs must already be anonymised (written by the inspection pipeline / blur step); this script never reads
data/raw. Outputs stay under the dataset directory (gitignored outputs/). Every number in the JSON reports is
measured by the code.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from avtopassport.recon3d.hq import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
