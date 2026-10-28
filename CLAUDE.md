@AGENTS.md

# Local fork: offline datasets and policy evaluation

This checkout is a fork of huggingface/lerobot-dataset-visualizer (`origin` is
the user's fork; don't push unless asked). It shows the user's
own LeRobot datasets offline and runs trained policies over recorded episodes.

## Hard rules

- **Never upload or modify the datasets or checkpoints.** Everything under
  `LOCAL_LEROBOT_DATA_ROOT` is read-only. The policy server sets
  `HF_HUB_OFFLINE=1`. After testing, `find <root> -newer <marker>` is a cheap
  check that nothing was written.
- Small changes to the visualizer are fine; ask before bigger refactors of
  upstream code. Keep fork changes contained (new files where possible) so
  upstream merges stay easy.

## Data layout

`LOCAL_LEROBOT_DATA_ROOT` (set in the git-ignored `.env`, see `.env.example`):

- `datasets/<org>/<dataset>/`: LeRobot v3.0 datasets
- `train/<run>/checkpoints/last/pretrained_model`: trained policies. Always
  use the `last` checkpoint symlink.
- `bags/`: ignored.

## Running

- `bun run dev:local` (`scripts/dev-local.sh`) loads `.env` (environment
  variables win), sets `DATASET_URL=/api/local/datasets`, starts
  `backend/policy_server.py` with `POLICY_PYTHON` (a Python with LeRobot 0.6.1
  - CUDA torch, default `python3`) on `POLICY_PORT` (7862), then `next dev`.
- Never commit machine-specific paths; they belong in `.env`.
- For browser testing use the launch config `visualizer-local` in the parent
  folder's `.claude/launch.json` (port 3005). The user
  often runs their own server: if the port or GPU is busy, ask rather than kill it.
  Stop the preview server when done.

## How the pieces fit

- `src/app/api/local/datasets/[...path]/route.ts`: serves files from
  `<root>/datasets` with Range support and path-traversal checks. `DATASET_URL`
  is inlined into the client bundle via `next.config.ts` `env`.
- `src/app/page.tsx`: lists local datasets on the start page when local mode is on.
- `next.config.ts`: rewrites `/api/policy/*` to `POLICY_BACKEND_URL` (same
  origin, no CORS); `POLICY_BACKEND_ENABLED` tells the client the panel is usable.
- `backend/policy_server.py`: stdlib HTTP server, one job at a time.
  - `GET /api/policy/policies` → `[{name, checkpoint}]` (only runs with a valid `last` checkpoint)
  - `GET /api/policy/info?policy=&repo_id=` → `type`, `chunk_size`, `input_features`, `default_remap`, `dataset_observation_keys`
  - `POST /api/policy/jobs {policy, repo_id, episode, actions_per_chunk, remap}`; `GET /api/policy/jobs/<id>` polls progress/result; `DELETE` cancels
  - `policy_dir(run)` only accepts names from `list_policies()`; never accept raw paths from the client.
  - Remap: `{dataset_key: policy_key}`; `""` drops a stream. Non-visual inputs are required, at least one image input must be fed. Default comes from the checkpoint's saved `rename_map`, else from matching names.
- `src/components/policy-panel.tsx`: policy select (remembered in
  localStorage `lerobot-viz-policy`), actions per chunk k (default 25, capped
  at `chunk_size`, not persisted), input remapping, progress + cancel, per-joint
  RMSE stats. The policy is queried every k frames and the first k actions of
  each chunk are used.
- `withPredictedActions` (in `policy-panel.tsx`) adds flat `predicted | <joint>`
  series into the chart group holding `action | <joint>`, used from
  `episode-viewer.tsx`.

## Gotchas

- Use **flat** `predicted | <joint>` keys: nested objects get overwritten when
  `mergeGroups` builds the combined chart view.
- The episode chart stretches timestamps to the video segment, so predicted
  rows are matched to chart rows **by position** (`round(i*last/(n-1))`), not by timestamp.
- SmolVLA pads missing cameras (`empty_cameras`) and its predictions vary a
  little between runs (sampling noise); don't treat small differences as bugs.
  Dataset cameras often need remapping (e.g. `observation.images.kinect` →
  `observation.images.camera1`).
- Fork UI defaults: combined chart view (`expanded = true` in
  `data-recharts.tsx`), 4-decimal legend values, episodes open paused
  (`simple-videos-player.tsx`), "← Datasets" link in the episode tab bar.
- `README.md` section "Local datasets and policy evaluation" documents this
  for the user; keep it in sync.
