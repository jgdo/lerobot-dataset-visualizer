@AGENTS.md

# Local fork: offline datasets and policy evaluation

This checkout is a fork of huggingface/lerobot-dataset-visualizer (`origin` is
the user's fork; don't push unless asked). It shows the user's
own LeRobot datasets offline and runs trained policies over recorded episodes.

## Hard rules

- **Never upload or modify the datasets or existing checkpoints.** The only
  writes are the training page's new runs in `train/<name>` (and deleting a
  run it was told to stop and delete) plus its queue in `train/.visualizer/`.
  The only upload is a trained policy to the Hub when the user asks for it in
  the UI (`backend/hf_upload.py`). **When testing, never upload for real**:
  `VISUALIZER_HF_DRY_RUN=1` (set in `visualizer-test-copy`).
- **Test training only against a copy of the data**, e.g.
  `../arips_lerobot_data_copy` via the launch
  config `visualizer-test-copy` (env var overrides `.env`; policy port 7863),
  with tiny runs (`steps=100`, `log_freq=10`). Never against the `.env` root.
  Evaluation is read-only. The policy server sets
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

## Training (`/training`, `backend/training.py`)

- `TrainingManager` keeps the queue in `train/.visualizer/queue.json`; one
  job runs at a time as `bash -c 'python -m lerobot.scripts.lerobot_train …;
echo $? > <id>.exit'` in its own process group (survives server restarts,
  found again via `VISUALIZER_TRAINING_JOB` in `/proc/<pid>/environ`). Output
  goes to `<id>.log`; `LogFollower` parses it incrementally (only the
  `Training:` tqdm bar counts; loss steps = line index × log_freq because
  lerobot prints `step:1K`).
- Stop = SIGINT to the process group, SIGTERM after 20 s, SIGKILL after 60 s;
  mode `delete` removes `train/<name>` afterwards.
- Endpoints: `GET train/options|runs|jobs|name?name=`, `POST train/jobs`,
  `POST train/jobs/<id>/stop {mode}`, `DELETE train/jobs/<id>` (cancel queued /
  dismiss finished). Allowed parameters are whitelisted in `clean_params`.
- The dataset of a run is `train_config.json` → `dataset.repo_id`
  (`config.json`'s `repo_id` is the policy's own Hub id).
- SmolVLA/pi0-fast fine-tune `lerobot/smolvla_base` / `lerobot/pi0fast-base`
  via `--policy.path`; dataset cameras are renamed onto the base's camera
  slots (`--rename_map`) and the remaining slots become `empty_cameras`. The
  training subprocess drops `HF_HUB_OFFLINE` so base models can download.
  pi0-fast also needs the gated `google/paligemma-3b-pt-224` tokenizer.
- Augmentation: `color_augmentation` / `position_augmentation` select groups
  of LeRobot's default transforms (`COLOR_TRANSFORMS`, `POSITION_TRANSFORMS`),
  passed as one JSON `--dataset.image_transforms.tfs=` (draccus rejects
  per-key overrides like `tfs.affine.weight`). Old queued jobs with
  `image_transforms` mean both groups. For `target_reach` the same flags become
  `--policy.color_augmentation` / `--policy.position_augmentation` and the dataset
  transforms stay off: the plugin's `augment.py` augments on the GPU after resizing.
- HF upload (`backend/hf_upload.py`, UI `src/components/hf-upload.tsx`):
  `UploadManager` runs `hf_upload.py <model_dir> <run> [repo_id]` as a
  subprocess without `HF_HUB_OFFLINE`; state in `train/.visualizer/uploads.json`,
  logs in `upload-<id>.log`. It uploads `*.safetensors|json|yaml|md` of the
  `last` `pretrained_model` plus a LeRobot model card in one commit (not
  lerobot's `push_to_hub`). A job queued with `upload: {repo_id, private}` starts
  it after exit code 0 (`after_done`). Endpoints: `GET train/hf` (whoami),
  `GET|POST train/uploads {run, repo_id, private}`, `DELETE train/uploads/<id>`.
  An empty repo id or one without `/` is prefixed with the logged-in user.
- Starting a training frees the evaluation policy's GPU memory
  (`release_policy`); `/api/policy/info` reports `training_active`.
- `target_reach` (plugin package `../lerobot_policy_target_reach`, installed
  editable into the venv with
  `uv pip install --python ../venv/bin/python --no-deps -e …`).
  - `enqueue` checks `target_tag` against `targets.json`: the tag must be used
    by at least one episode, and the episode lengths must still match. It then
    passes `--policy.annotation_path` and `--policy.target_tag`.
  - The chunk parameters are not allowed.
  - `LogFollower` exposes the plugin's "usable frames" log line as
    `progress.note`.
  - The plugin's tests: `cd ../lerobot_policy_target_reach && uv run --no-project --python ../venv/bin/python --with pytest python -m pytest -q tests`.

## Target annotations (`backend/target_annotations.py`)

- File: `<root>/annotations/<org>/<dataset>/targets.json`, containing
  `{version, types, tags, episodes: {"<ep>": {length, targets: [{frame, type, tag}]}}}`.
  - Types are fixed: `lin`, `circ`, `coll`.
  - A tag is unique within an episode.
  - Writes are atomic.
  - `snapshot` adds `mismatched_episodes` (stored length ≠ dataset) and
    `num_episodes`.
- Endpoints:
  - `GET annotations/targets?repo_id=`
  - `PUT annotations/targets?repo_id=&episode= {targets}`
  - `GET annotations/tags/usage?repo_id=&tag=`
  - `POST annotations/tags?repo_id= {add}` or `{remove}`. Removing a tag deletes its targets everywhere.
- UI: `src/components/target-annotations.tsx`. The chart markers use the
  `markers` prop of `DataRecharts`. Frame ↔ time conversion uses the chart rows'
  timestamps, not fps.
- The plugin sampler patches `EpisodeAwareSampler` in `lerobot_train`, and in
  `__main__`, because training runs via `python -m`. It only acts when a
  `TargetReachConfig` with an annotation path and a tag has been created.
  The `load_policy` in the policy server calls `register_third_party_plugins()`
  so these checkpoints load.
- Evaluating a `target_reach` policy works differently from other policies.
  - The job result includes `target` (`{tag, key, frame, pose[]}`), built by
    `target_reference` from the dataset's **current** annotations.
  - `per_joint` compares the prediction with that pose over frames `0…t`
    instead of the recorded actions. An episode without the tag gets no stats.
  - The chart shows a `target | <joint>` group, defined in
    `withPredictedActions`.

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
