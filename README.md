---
title: Visualize Dataset (v2.0+ latest dataset format)
emoji: 💻
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
hf_oauth: true
hf_oauth_scopes:
  - read-repos
hf_oauth_expiration_minutes: 480
---

# LeRobot Dataset Visualizer

LeRobot Dataset Tool and Visualizer is a web application for interactive exploration and visualization of robotics datasets, particularly those in the LeRobot format. It enables users to browse, view, and analyze episodes from large-scale robotics datasets, combining synchronized video playback with rich, interactive data graphs.

## Project Overview

This tool is designed to help robotics researchers and practitioners quickly inspect and understand large, complex datasets. It fetches dataset metadata and episode data (including video and sensor/telemetry data), and provides a unified interface for:

- Navigating between organizations, datasets, and episodes
- Watching episode videos
- Exploring synchronized time-series data with interactive charts
- Analyzing action quality and identifying problematic episodes
- Visualizing robot poses in 3D using URDF models
- Paginating through large datasets efficiently

## Key Features

- **Dataset & Episode Navigation:** Quickly jump between organizations, datasets, and episodes using a sidebar and navigation controls.
- **Synchronized Video & Data:** Video playback is synchronized with interactive data graphs for detailed inspection of sensor and control signals.
- **Overview Panel:** At-a-glance summary of dataset metadata, camera info, and episode details.
- **Statistics Panel:** Dataset-level statistics including episode count, total recording time, frames-per-second, and an episode-length histogram.
- **Action Insights Panel:** Data-driven analysis tools to guide training configuration — includes autocorrelation, state-action alignment, speed distribution, and cross-episode variance heatmap.
- **Filtering Panel:** Identify and flag problematic episodes (low movement, jerky motion, outlier length) for removal. Exports flagged episode IDs as a ready-to-run LeRobot CLI command.
- **3D URDF Viewer:** Visualize robot joint poses frame-by-frame in an interactive 3D scene, with end-effector trail rendering. Supports SO-100, SO-101, and OpenArm bimanual robots.
- **Annotations Panel:** Hand-edit the v3.1 language schema (`language_persistent` + `language_events`) — subtask, plan, memory, interjection + paired speech, and VQA atoms with bounding-box / keypoint / count / attribute / spatial answers. VQA bboxes and keypoints render as overlays on the video player; drag or click on a camera to draw new ones. Backed by an optional FastAPI service (in `backend/`) for parquet rewrites and HF Hub push.
- **Efficient Data Loading:** Uses parquet and JSON loading for large dataset support, with pagination, chunking, and lazy-loaded panels for fast initial load.
- **Responsive UI:** Built with React, Next.js, and Tailwind CSS for a fast, modern user experience.

## Technologies Used

- **Next.js** (App Router)
- **React**
- **Recharts** (for data visualization)
- **Three.js** + **@react-three/fiber** + **@react-three/drei** (for 3D URDF visualization)
- **urdf-loader** (for parsing URDF robot models)
- **hyparquet** (for reading Parquet files)
- **Tailwind CSS** (styling)

## Getting Started

### Prerequisites

This project uses [Bun](https://bun.sh) as its package manager. If you don't have it installed:

```bash
# Install Bun
curl -fsSL https://bun.sh/install | bash
```

### Installation

Install dependencies:

```bash
bun install
```

### Development

Run the development server:

```bash
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `src/app/page.tsx` or other files in the `src/` directory. The app supports hot-reloading for rapid development.

### Other Commands

```bash
# Build for production
bun run build

# Start production server
bun start

# Run linter
bun run lint

# Format code
bun run format
```

### Environment Variables

- `DATASET_URL`: (optional) Base URL for dataset hosting (defaults to HuggingFace Datasets).
- `NEXT_PUBLIC_ANNOTATE_BACKEND_URL`: (optional) URL of the FastAPI annotation
  backend (`backend/app.py`). When set, the Annotations tab can save edits and
  rewrite parquet shards / push to the Hub. When unset the tab is read/edit
  only with sessionStorage persistence.

## Local datasets and policy evaluation

Set `LOCAL_LEROBOT_DATA_ROOT` and optionally `POLICY_PYTHON` in a local `.env`
file (see `.env.example`; `.env` is git-ignored).
`bun run dev:local` (see `scripts/dev-local.sh`) reads everything from
`LOCAL_LEROBOT_DATA_ROOT`, read-only and instead of the Hub: LeRobot datasets
from `datasets/<org>/<dataset>` and trained policies from
`train/<run>/checkpoints/last/pretrained_model`. It also starts
`backend/policy_server.py` with `POLICY_PYTHON` (a Python that has `lerobot`
and `torch` installed). In the episode view, **Run trained policy** then runs
the selected training run's latest checkpoint over the whole episode on the
GPU and charts
the predicted actions as a "predicted" group in the episode chart, next to
the recorded ones, with per-joint error stats in the policy panel.
"Actions used per chunk" = N queries the policy every N frames and uses the
first N actions of each predicted chunk. "Input remapping" picks which policy
input each dataset observation stream feeds (e.g. `observation.images.kinect`
→ `observation.images.camera1`); it starts from the `rename_map` the
checkpoint was trained with, else from matching names.

**Train policies** on the start page opens `/training`: it lists every run
under `train/` with the dataset it was trained on, and queues new ACT, SmolVLA
or pi0-fast trainings (`lerobot-train`, `push_to_hub` and wandb off) that run
one after another into `train/<name>`. The default name is
`<type>_<org>_<dataset>_<DDMMYYYY>_<HHMMSS>`. Each running job shows its
progress, loss curve and estimated finish time; queued jobs can be cancelled
and running ones stopped, keeping or deleting their checkpoints. The queue is
stored in `train/.visualizer/` and trainings run as separate processes, so
both survive a restart of `dev:local`. SmolVLA and pi0-fast fine-tune their
Hub base models, which are downloaded on first use; pi0-fast also needs access
to the gated `google/paligemma-3b-pt-224`.

Image augmentation has two separate switches. **Color augmentation** randomly
changes brightness, contrast, saturation, hue and sharpness. **Position
augmentation** randomly rotates (±5°) and shifts (±5%) the image. Both use
LeRobot's default transforms; each frame gets up to 3 of the enabled ones.
For `target_reach` the augmentation runs batched on the GPU after the images
are scaled to 240×320, instead of on full-size frames in the data loader
workers (with 4 workers this makes batch loading about 4× faster).

**Upload to Hugging Face.** Only on request: tick **Upload to Hugging Face
when done** in the New training form, or press **Upload** next to a trained
policy. The last checkpoint's `pretrained_model` files (weights, configs,
processors) and a generated model card go to the given model repository
(default `<user>/<run name>`, private by default); the dataset is never
uploaded. The login comes from `hf auth login` or `HF_TOKEN` on the machine
running the visualizer. `VISUALIZER_HF_DRY_RUN=1` does everything except
contacting the Hub, for testing.

**Target frames.** The episode view has a **Target annotations** panel for
marking target frames. Each target has a type (`lin`, `circ` or `coll`) and a
tag from an editable, dataset-wide list.

- Keys: `t` adds a target at the current frame; `[` and `]` jump to the
  previous and next target.
- Each tag can be used at most once per episode.
- Removing a tag (after a confirmation) also deletes every target that uses it.
- Targets are drawn as markers in the chart, and the sidebar counts them per
  episode.
- They are stored next to the dataset, never inside it, in
  `annotations/<org>/<dataset>/targets.json`.

**Target reach policy.** The `/training` page also offers **Target reach**, a
plugin policy from `../lerobot_policy_target_reach`. Install it into the
`POLICY_PYTHON` environment with `pip install -e`.

- Model: a frozen ImageNet ResNet18 plus the joint state feed a small MLP
  (about 400k trainable parameters), which predicts the joint pose at the
  episode's target frame for the selected tag.
- Training data: frames `0…t` of each episode that has the tag, where `t` is
  the tagged frame. Episodes without the tag are skipped. A filtered sampler
  makes every batch sample usable, and color and position augmentation still
  apply.
- The default run name includes the tag.

## Annotations backend (optional)

The Annotations tab edits LeRobot v3.1 language atoms — `language_persistent`
(broadcast subtask/plan/memory) and `language_events` (per-frame
interjection / vqa / speech) — and renders existing bbox/keypoint atoms over
the video player. Edits live in `sessionStorage` by default; to write the
new columns into `data/chunk-*/file-*.parquet` (matching the writer in
[lerobot#3471](https://github.com/huggingface/lerobot/pull/3471)) and push the
result to the Hub, run the bundled FastAPI service:

```bash
# 1. install + start the backend (port 7861 by default)
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --port 7861 --reload

# 2. start the visualizer with the backend URL configured
cd ..
NEXT_PUBLIC_ANNOTATE_BACKEND_URL=http://127.0.0.1:7861 bun run dev
```

The backend exposes:

- `POST /api/dataset/load` — load a dataset by `repo_id` or `local_path`
- `GET  /api/episodes/{ep}/atoms` — list atoms for an episode
- `POST /api/episodes/{ep}/atoms` — replace atoms (event timestamps are
  snapped to exact source-frame timestamps before persisting)
- `GET  /api/episodes/{ep}/frame_timestamps` — used client-side for snapping
- `POST /api/export` — rewrite parquet with the new language columns plus
  the dataset-level `tools` column (drops legacy `subtask_index`)
- `POST /api/push_to_hub` — export and push to a target repo

## Docker Deployment

This application can be deployed using Docker with bun for optimal performance and self-contained builds.

### Build the Docker image

```bash
docker build -t lerobot-visualizer .
```

### Run the container

```bash
docker run -p 7860:7860 lerobot-visualizer
```

The application will be available at [http://localhost:7860](http://localhost:7860).

### Run with custom environment variables

```bash
docker run -p 7860:7860 -e DATASET_URL=your-url lerobot-visualizer
```

## Contributing

Contributions, bug reports, and feature requests are welcome! Please open an issue or submit a pull request.

### Acknowledgement

The app was orignally created by [@Mishig25](https://github.com/mishig25) and taken from this PR [#1055](https://github.com/huggingface/lerobot/pull/1055)
