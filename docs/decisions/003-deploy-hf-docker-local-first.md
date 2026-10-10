# ADR 003: Deploy to a Hugging Face Docker Space, but test locally first

- **Status:** Accepted as the plan; the purchase is pending. Nothing has been bought. HF PRO is bought only at the Run 2 decision point, if the local end-to-end test passes, and only with the user's explicit approval.
- **Date:** 2026-10-10. Sources were checked the same day.

## Context

- **The brief.**
  - The app must be "deployed publicly (web, works on mobile browsers), deployed from this repo", and after the deadline only the submitted commit may be redeployed (CLAUDE.md section 2).
  - Phase 4 plans "Dockerfile → Hugging Face Spaces (Docker) or Render".
- **HF docs change of 2026-07-21** ([hub-docs PR #2624][hf-2624], "Update Spaces docs: paid plan required for compute Spaces, ZeroGPU free tier"). The [Spaces overview][hf-overview] now says:
  > "Static Spaces are free for everyone. Gradio and Docker Spaces run on compute and require a paid plan to create: PRO for personal accounts, Team or Enterprise for organizations. Free personal accounts in good standing can still host up to 2 Gradio Spaces running on ZeroGPU."
- **PRO price.** PRO costs **$9/month** ([pricing][hf-pricing], retrieved 2026-10-10).
- **CPU Basic hardware** has no hourly cost: 2 vCPU, 16 GB RAM and 50 GB of disk that is not persistent ([overview][hf-overview]).
  - It sleeps after 48 hours without visitors and restarts on the next visit ([sleep time][hf-sleep]).
  - Outbound requests may use only ports 80, 443 and 8080. Our external calls (NHTSA vPIC, Overpass, freetsa) all use HTTPS.
- **Licence.** The repo is AGPL-3.0 (ADR 001). Section 13 applies to network use.

### Alternatives considered

| option | cost | limits | verdict |
|---|---|---|---|
| HF Docker Space on CPU Basic | PRO $9/month to create; CPU Basic $0/hour | 2 vCPU, 16 GB RAM, 50 GB non-persistent disk; sleeps after 48 h idle | **chosen**, purchase pending the local test |
| HF CPU Upgrade | $0.03/hour on top of PRO | 8 vCPU, 32 GB | only if 2 vCPU proves too slow; ask first |
| HF Static Space | free | static files only | cannot run our Python/ONNX backend |
| HF ZeroGPU Space | free for personal accounts in good standing (verified email, account older than 30 days), up to 2 Spaces | Gradio SDK only; shared GPU with a daily usage quota (PRO gets 8x more) | a different stack from our FastAPI + static web design, plus quotas; fallback only |
| Render free | $0 | 0.1 CPU and 512 MB RAM; spins down after 15 min idle and takes about 1 min to restart; files (including SQLite) wiped on redeploy, restart or spin-down; no disks | too weak for YOLO segmentation |
| Render paid | $25/month for 1 CPU / 2 GB ($7/month for 0.5 CPU / 512 MB); disks $0.25/GB/month | 1 CPU | costs more than HF PRO for fewer CPUs |
| Google Cloud Run free tier | $0 within 180,000 vCPU-seconds, 360,000 GB-seconds and 2 million requests per month | "A Google Cloud billing account is required to access the Google Cloud Free Tier"; the Free Trial sign-up asks for a credit card or other payment method | needs billing set up |
| Laptop plus phones (same Wi-Fi, or a Cloudflare Quick Tunnel) | $0 | not a public deploy. Quick Tunnels: no account needed, "for testing and development", no uptime guarantee, up to 200 in-flight requests, no Server-Sent Events, a new hostname each time | used for the local end-to-end test only |

Sources:
- **Hugging Face:** [Spaces overview][hf-overview], [pricing][hf-pricing], [sleep time][hf-sleep], [ZeroGPU][hf-zerogpu].
- **Render:** [free tier][render-free], [compute plans][render-plans], as recorded in `phase0_findings_deploy_market.md` section 1.
- **Google Cloud:** [free features][gcp-free].
- **Cloudflare:** [Quick Tunnels][cf-quick].

### Persistence

- **The Space's disk is not persistent.** The old Spaces persistent storage was shut down (HF staff on the forum, 2026-03-27). Its replacement is Storage Buckets mounted as volumes, at $18/TB/month private or $12/TB/month public.
- **SQLite must not live on a bucket mount.** The hf-mount README warns of no cross-client locking, data up to 10 s stale, and slow random I/O (`phase0_findings_deploy_market.md` section 1).

## Decision

1. **Build and test locally first (Run 2).**
   - Run the same app (FastAPI plus the static `web/` page, from the Docker image where possible) on the laptop.
   - Phones connect in one of two ways:
     - over the same Wi-Fi at `http://<laptop-LAN-IP>:<port>`;
     - through a free Cloudflare Quick Tunnel (`cloudflared tunnel --url http://localhost:<port>`), which gives a temporary HTTPS URL without an account. The user installs `cloudflared` themselves; the repo does not depend on it.
   - Because Quick Tunnels do not support Server-Sent Events, the web page reports progress by polling.
2. **Decision point in Run 2.**
   - If the local end-to-end test passes, the user decides whether to buy HF PRO ($9/month). The test means a full inspection on the laptop and on at least 3 different phones, as Phase 4 asks.
   - With the user's approval, we then create the Docker Space on CPU Basic.
   - If the test fails, we fix the app first and do not buy anything.
   - The decision point must leave enough time before the 13:00 deadline on 2026-10-11 to deploy and test the public URL.
3. **Deploy from the repo.**
   - The Space README front matter sets `sdk: docker`. The app listens on `app_port` (default 7860), and the container runs as user ID 1000.
   - The GitHub Actions workflow (`.github/workflows/deploy.yml`) syncs the repo with `huggingface/hub-sync@v0.3.0`, the `HF_TOKEN` secret and `space_sdk: docker`, because the default is gradio. A `git push` to the Space repo is the alternative.
4. **Persistence.**
   - SQLite lives on the container's local disk.
   - At boot, the app restores the latest snapshot from a **private HF dataset repo**.
   - A [`huggingface_hub.CommitScheduler`][hf-scheduler] pushes a snapshot folder every few minutes. The interval is set in code and is an assumption until tested.
   - Each snapshot is a consistent copy made with SQLite's backup API (or `VACUUM INTO`), never the live database file.
   - Fallback if this fails: Turso's free tier (5 GB, 100 databases), as reported in Phase 0.
5. **AGPL-3.0 section 13.** The deployed app shows a visible "Source code (AGPL-3.0)" link to the repository at the exact deployed commit (`.../tree/<commit-sha>`).
   - The CI passes the commit SHA into the image at build time, and an `/api/version` endpoint returns it.
   - Section 13 of `LICENSE`: "if you modify the Program, your modified version must prominently offer all users interacting with it remotely through a computer network (if your version supports such interaction) an opportunity to receive the Corresponding Source of your version by providing access to the Corresponding Source from a network server at no charge, ...".
6. **Measure on the real host.** After the deploy, measure the full inspection time on the Space, as CLAUDE.md section 8 requires. A script writes it into the docs; it is never typed by hand.

## Consequences

- **Fixed hosting cost.** Once bought, PRO costs $9/month. It is recorded in `docs/pitch/cost_model.md` as "not purchased - pending local test".
- **Risk: no public URL.** Without PRO there is no public URL, which the rules require. Each fallback needs the user's decision:
  1. Render paid ($25/month for 1 CPU / 2 GB).
  2. A ZeroGPU Gradio wrapper (free, but a different stack, with quotas and account-age rules).
  3. Cloud Run (needs a billing account).
- **Throughput.** Two vCPUs process about one inspection at a time, so capacity per hour depends on the measured inspection time. No number exists yet.
- **Cold start.** The first visit after 48 h idle waits for the Space to restart; that wait has not been measured. Open the Space before the demo.
- **Snapshot loss window.** A container restart loses whatever was written since the last snapshot.
- **Deployed commit = submitted commit.** The source link makes it visible that the deployed commit matches the submitted one (CLAUDE.md sections 2 and 9).

[hf-2624]: https://github.com/huggingface/hub-docs/pull/2624
[hf-overview]: https://huggingface.co/docs/hub/spaces-overview
[hf-pricing]: https://huggingface.co/pricing
[hf-sleep]: https://huggingface.co/docs/hub/spaces-gpus#sleep-time
[hf-zerogpu]: https://huggingface.co/docs/hub/spaces-zerogpu
[hf-scheduler]: https://huggingface.co/docs/huggingface_hub/guides/upload#scheduled-uploads
[render-free]: https://render.com/docs/free
[render-plans]: https://render.com/docs/compute-plans
[gcp-free]: https://cloud.google.com/free/docs/free-cloud-features
[cf-quick]: https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/

**Update 2026-10-11:** implemented as opt-in (`src/avtopassport/api/persist.py`), enabled by `AVTO_SNAPSHOT_REPO` + `HF_TOKEN`. It uses `HfApi.upload_file` of one archive (DB backup + small job files + manifest) every `AVTO_SNAPSHOT_EVERY_MIN` minutes (default 5) and after each inspection, instead of `CommitScheduler`.
