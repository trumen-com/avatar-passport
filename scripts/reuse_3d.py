"""Admin: index the finished 3D builds of a data dir, and attach a stored build to finished jobs that lack one.

    python scripts/reuse_3d.py                      # (re)index DATA_DIR/cache/recon3d_index.json from the job states
    python scripts/reuse_3d.py --job <id> [--job ...] [--kind recon3d_hq]

For each --job whose recon3d / recon3d_hq is not "done" (e.g. a high-quality build that was interrupted), a finished
build of exactly the same 3D input (frames_sha256 of the blurred frames + frame config + model identity, see
avtopassport.api.reuse3d) is linked into the job with the honest "reused" fields; nothing is built or invented. A job
without a matching finished build is left unchanged. The data dir is AVTO_DATA_DIR (default REPO_ROOT/var); the HQ
steps of the key are AVTO_HQ3D_STEPS (default 30000), as on the server.

The server keeps job states in memory: restart it after attaching so it serves the updated state.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from avtopassport.api import reuse3d
from avtopassport.api.jobs import JOB_ID_RE, JobManager, Settings


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=None, help="default AVTO_DATA_DIR or REPO_ROOT/var")
    ap.add_argument("--job", action="append", default=[], help="job id to attach a stored build to (repeatable)")
    ap.add_argument("--kind", choices=reuse3d.KINDS, action="append", default=None,
                    help="which builds to attach (default both)")
    a = ap.parse_args(argv)
    settings = Settings.from_env(a.data_dir.resolve() if a.data_dir else None)
    jm = JobManager(settings)  # not started: no worker threads, no models; reads / writes job states on disk
    for state_file in sorted(jm.jobs_dir.glob("*/state.json")):
        if JOB_ID_RE.match(state_file.parent.name):
            jm.internal(state_file.parent.name)  # load into the manager for the backfill
    changed = jm.backfill_index()
    entries = jm.reuse_index.read()
    print(f"index {jm.reuse_index.path}: {len(entries)} entries ({changed} changed)")
    for key, e in sorted(entries.items()):
        print(f"  {key}\n    -> job {e.get('job_id')} ({e.get('kind')}, {e.get('seconds')} s on {e.get('device')}, "
              f"inspection {e.get('created_utc')})")
    rc = 0
    for job_id in a.job:
        st = jm.internal(job_id)
        if st is None:
            print(f"{job_id}: job not found", file=sys.stderr)
            rc = 1
            continue
        if st.get("status") != "done":
            print(f"{job_id}: inspection status is {st.get('status')!r}, not done; skipped", file=sys.stderr)
            rc = 1
            continue
        for kind in a.kind or reuse3d.KINDS:
            status = (st.get(kind) or {}).get("status")
            if status == "done":
                print(f"{job_id} {kind}: already done; unchanged")
                continue
            ok, msg = jm.reuse_build(job_id, kind)
            print(f"{job_id} {kind}: {'ATTACHED' if ok else 'not attached'} (was {status!r}): {msg}")
            if ok:
                rec = (jm.internal(job_id) or {}).get(kind) or {}
                print("  " + json.dumps({k: rec.get(k) for k in ("status", "reused_from_job", "reused_from_created_utc",
                                                                 "original_seconds", "original_device",
                                                                 "reuse_files", "reuse_seconds")}))
    return rc


if __name__ == "__main__":
    sys.exit(main())
