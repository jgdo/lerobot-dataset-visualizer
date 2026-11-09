"use client";

import Link from "next/link";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, POLICY_BACKEND_ENABLED } from "@/components/policy-panel";
import { useTargetFile } from "@/components/target-annotations";
import {
  dismissUpload,
  HfUpload,
  repoIdProblem,
  UploadDialog,
  UploadRequest,
  UploadStatus,
  UploadWhenDone,
  useUploads,
} from "@/components/hf-upload";
import { PageHeader, Spinner, ThemeToggle } from "@/components/ui";

// Talks to the training queue of backend/policy_server.py (training.py):
// lists runs under train/, queues lerobot-train jobs and follows their logs.

type PolicyTypeSpec = {
  type: string;
  label: string;
  base: string | null;
  cameras: string[] | null;
};
type LocalDataset = {
  repo_id: string;
  cameras: string[];
  episodes: number | null;
  frames: number | null;
  fps: number | null;
};
type TrainOptions = { types: PolicyTypeSpec[]; datasets: LocalDataset[] };
type TrainingRun = {
  name: string;
  type: string | null;
  dataset: string | null;
  steps: number | null;
  checkpoint: string | null;
  checkpoints: string[];
  has_policy: boolean;
  modified: number;
};
type JobProgress = {
  step: number;
  total: number;
  elapsed_s: number | null;
  remaining_s: number | null;
  losses: [number, number][];
  tail: string[];
  note?: string | null;
  phase?: "preparing" | "training";
};
type TrainingJob = {
  id: string;
  name: string;
  policy_type: string;
  repo_id: string;
  params: Record<string, number | boolean | string>;
  rename_map: Record<string, string>;
  empty_cameras: number;
  target_tag?: string;
  upload?: UploadRequest | null;
  upload_error?: string;
  status: "queued" | "running" | "stopping" | "done" | "failed" | "stopped";
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  error: string | null;
  stop_mode: "keep" | "delete" | null;
  deleted?: boolean;
  progress?: JobProgress;
};

type Field = {
  key: string;
  label: string;
  help: string;
  kind: "int" | "float" | "bool" | "choice";
  choices?: string[];
};

const COMMON_FIELDS: Field[] = [
  {
    key: "steps",
    label: "Training steps",
    help: "Total number of optimizer steps",
    kind: "int",
  },
  {
    key: "batch_size",
    label: "Batch size",
    help: "Samples per step (lower it if the GPU runs out of memory)",
    kind: "int",
  },
  {
    key: "save_freq",
    label: "Save checkpoint every",
    help: "Steps between saved checkpoints",
    kind: "int",
  },
  {
    key: "log_freq",
    label: "Log loss every",
    help: "Steps between loss values in the chart",
    kind: "int",
  },
  {
    key: "chunk_size",
    label: "Action chunk size",
    help: "Number of future actions the policy predicts at once",
    kind: "int",
  },
  {
    key: "n_action_steps",
    label: "Actions executed per chunk",
    help: "Actions of each chunk executed before predicting again (≤ chunk size)",
    kind: "int",
  },
  {
    key: "optimizer_lr",
    label: "Learning rate",
    help: "Empty = the policy's default",
    kind: "float",
  },
  {
    key: "num_workers",
    label: "Data loader workers",
    help: "Processes decoding video frames",
    kind: "int",
  },
  {
    key: "seed",
    label: "Seed",
    help: "Random seed",
    kind: "int",
  },
  {
    key: "color_augmentation",
    label: "Color augmentation",
    help: "Random brightness, contrast, saturation, hue and sharpness of camera images",
    kind: "bool",
  },
  {
    key: "position_augmentation",
    label: "Position augmentation",
    help: "Random rotation (±5°) and shift (±5%) of camera images",
    kind: "bool",
  },
];

const TYPE_FIELDS: Record<string, Field[]> = {
  act: [
    {
      key: "use_vae",
      label: "Use VAE",
      help: "Train ACT's variational objective",
      kind: "bool",
    },
  ],
  smolvla: [
    {
      key: "freeze_vision_encoder",
      label: "Freeze vision encoder",
      help: "Keep the pretrained image encoder fixed",
      kind: "bool",
    },
    {
      key: "train_expert_only",
      label: "Train action expert only",
      help: "Keep the pretrained VLM fixed and train only the action expert",
      kind: "bool",
    },
  ],
  pi0_fast: [
    {
      key: "dtype",
      label: "Precision",
      help: "bfloat16 is needed to fit the model on a 24 GB GPU",
      kind: "choice",
      choices: ["bfloat16", "float32"],
    },
    {
      key: "gradient_checkpointing",
      label: "Gradient checkpointing",
      help: "Saves GPU memory at the cost of speed",
      kind: "bool",
    },
  ],
  target_reach: [
    {
      key: "target_key",
      label: "Target pose",
      help: "observation.state = the reached joint position at the target frame, action = the commanded one",
      kind: "choice",
      choices: ["observation.state", "action"],
    },
  ],
};

// target_reach predicts a single pose, so it has no action chunk.
const CHUNK_KEYS = new Set(["chunk_size", "n_action_steps"]);

function fieldsFor(type: string) {
  const common =
    type === "target_reach"
      ? COMMON_FIELDS.filter((f) => !CHUNK_KEYS.has(f.key))
      : COMMON_FIELDS;
  return [...common, ...(TYPE_FIELDS[type] ?? [])];
}

type ParamValues = Record<string, string | boolean>;

const DEFAULTS: Record<string, ParamValues> = {
  act: {
    steps: "50000",
    batch_size: "16",
    save_freq: "5000",
    log_freq: "200",
    chunk_size: "100",
    n_action_steps: "100",
    optimizer_lr: "",
    num_workers: "4",
    seed: "1000",
    color_augmentation: false,
    position_augmentation: false,
    use_vae: true,
  },
  smolvla: {
    steps: "25000",
    batch_size: "16",
    save_freq: "5000",
    log_freq: "200",
    chunk_size: "50",
    n_action_steps: "50",
    optimizer_lr: "",
    num_workers: "4",
    seed: "1000",
    color_augmentation: false,
    position_augmentation: false,
    freeze_vision_encoder: true,
    train_expert_only: true,
  },
  pi0_fast: {
    steps: "25000",
    batch_size: "4",
    save_freq: "5000",
    log_freq: "200",
    chunk_size: "50",
    n_action_steps: "50",
    optimizer_lr: "",
    num_workers: "4",
    seed: "1000",
    color_augmentation: false,
    position_augmentation: false,
    dtype: "bfloat16",
    gradient_checkpointing: true,
  },
  target_reach: {
    steps: "5000",
    batch_size: "32",
    save_freq: "1000",
    log_freq: "100",
    optimizer_lr: "",
    num_workers: "4",
    seed: "1000",
    color_augmentation: true,
    position_augmentation: true,
    target_key: "observation.state",
  },
};

const INPUT_CLASS =
  "w-full rounded-md border border-line bg-[var(--surface-0)] px-2 py-1.5 text-xs text-fg outline-none focus:border-accent disabled:opacity-50";
const CARD_CLASS =
  "rounded-lg border border-line bg-[var(--surface-1)]/60 p-5 space-y-4";

function pad(n: number) {
  return String(n).padStart(2, "0");
}

function exampleName(type: string, repoId: string, tag?: string) {
  const d = new Date();
  const tagPart = tag ? `_${tag}` : "";
  return `${type}_${repoId.replace("/", "_")}${tagPart}_${pad(d.getDate())}${pad(
    d.getMonth() + 1,
  )}${d.getFullYear()}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(
    d.getSeconds(),
  )}`;
}

function formatDuration(s: number | null | undefined) {
  if (s == null || !isFinite(s)) return "–";
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return h > 0 ? `${h}h ${pad(m)}m` : m > 0 ? `${m}m ${pad(sec)}s` : `${sec}s`;
}

function formatClock(ms: number) {
  const d = new Date(ms);
  const sameDay = d.toDateString() === new Date().toDateString();
  return d.toLocaleString(undefined, {
    weekday: sameDay ? undefined : "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Default camera renames: keep matching names, else fill the base's slots in order. */
function defaultCameraMap(datasetCams: string[], baseCams: string[]) {
  const map: Record<string, string> = {};
  const free = baseCams.filter((c) => !datasetCams.includes(c));
  for (const cam of datasetCams) {
    map[cam] = baseCams.includes(cam) ? cam : (free.shift() ?? "");
  }
  return map;
}

export default function TrainingPage() {
  const [options, setOptions] = useState<TrainOptions | null>(null);
  const [runs, setRuns] = useState<TrainingRun[] | null>(null);
  const [jobs, setJobs] = useState<TrainingJob[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refreshRuns = useCallback(() => {
    api<TrainingRun[]>("/api/policy/train/runs")
      .then(setRuns)
      .catch((e) => setError(String(e.message ?? e)));
  }, []);

  useEffect(() => {
    if (!POLICY_BACKEND_ENABLED) return;
    api<TrainOptions>("/api/policy/train/options")
      .then(setOptions)
      .catch((e) => setError(String(e.message ?? e)));
    refreshRuns();
  }, [refreshRuns]);

  // Poll the queue; refresh the run list whenever a job changes state.
  const statusKey = (jobs ?? []).map((j) => `${j.id}:${j.status}`).join(",");
  useEffect(() => {
    if (!POLICY_BACKEND_ENABLED) return;
    let stopped = false;
    const poll = () =>
      api<{ jobs: TrainingJob[] }>("/api/policy/train/jobs")
        .then((d) => !stopped && setJobs(d.jobs))
        .catch((e) => !stopped && setError(String(e.message ?? e)));
    poll();
    const timer = setInterval(poll, 2000);
    return () => {
      stopped = true;
      clearInterval(timer);
    };
  }, []);
  const { uploads, refresh: refreshUploads } = useUploads();
  useEffect(() => {
    if (statusKey) {
      refreshRuns();
      refreshUploads(); // a finished job may have started its upload
    }
  }, [statusKey, refreshRuns, refreshUploads]);

  const activeNames = useMemo(
    () =>
      new Set(
        (jobs ?? [])
          .filter((j) => j.status === "running" || j.status === "stopping")
          .map((j) => j.name),
      ),
    [jobs],
  );

  return (
    <div className="min-h-screen bg-[var(--bg)] text-fg">
      <header className="flex items-center gap-3 border-b border-line-subtle bg-[var(--surface-0)] px-4 py-2">
        <Link
          href="/"
          className="text-xs font-medium tracking-wide text-fg-muted transition-colors hover:text-fg"
        >
          ← Datasets
        </Link>
        <span className="text-sm font-semibold">Training</span>
        <div className="ml-auto">
          <ThemeToggle />
        </div>
      </header>

      <main className="mx-auto w-full max-w-5xl space-y-6 px-4 py-6">
        <PageHeader
          title="Policy training"
          description="Train LeRobot policies on local datasets. Runs are written to LOCAL_LEROBOT_DATA_ROOT/train and run one after another."
        />

        {!POLICY_BACKEND_ENABLED && (
          <p className="text-sm text-fg-muted">
            The policy service is not running. Start the visualizer with{" "}
            <code>bun run dev:local</code>.
          </p>
        )}
        {error && <p className="text-sm text-flag-fg">{error}</p>}

        {POLICY_BACKEND_ENABLED && (
          <>
            <QueueSection
              jobs={jobs}
              uploads={uploads}
              onChanged={refreshRuns}
            />
            {options && (
              <NewTrainingForm
                options={options}
                existing={runs ?? []}
                onQueued={(job) =>
                  setJobs((js) => [
                    ...(js ?? []).filter((j) => j.id !== job.id),
                    job,
                  ])
                }
              />
            )}
            <RunsSection
              runs={runs}
              activeNames={activeNames}
              uploads={uploads}
              onUploadsChanged={refreshUploads}
            />
          </>
        )}
      </main>
    </div>
  );
}

function RunsSection({
  runs,
  activeNames,
  uploads,
  onUploadsChanged,
}: {
  runs: TrainingRun[] | null;
  activeNames: Set<string>;
  uploads: HfUpload[];
  onUploadsChanged: () => void;
}) {
  const [uploadRun, setUploadRun] = useState<string | null>(null);
  const latestUpload = (run: string) =>
    uploads.filter((u) => u.run === run).at(-1);
  return (
    <section className={CARD_CLASS}>
      <h3 className="text-sm font-semibold text-fg">Trained policies</h3>
      {!runs ? (
        <p className="flex items-center gap-2 text-xs text-fg-muted">
          <Spinner /> Loading…
        </p>
      ) : runs.length === 0 ? (
        <p className="text-xs text-fg-muted">No training runs yet.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="text-fg-faint">
              <tr className="border-b border-line-subtle">
                <th className="py-1.5 pr-4 font-medium">Name / dataset</th>
                <th className="py-1.5 pr-4 font-medium">Type</th>
                <th className="py-1.5 pr-4 font-medium">Last checkpoint</th>
                <th className="py-1.5 pr-4 font-medium">Modified</th>
                <th className="py-1.5 font-medium">Hugging Face</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.name} className="border-b border-line-subtle/60">
                  <td className="break-all py-1.5 pr-4 font-mono">
                    <div className="text-fg">{r.name}</div>
                    <div className="text-[11px] text-fg-muted">
                      {r.dataset ?? "–"}
                    </div>
                  </td>
                  <td className="py-1.5 pr-4 text-fg-soft">{r.type ?? "–"}</td>
                  <td className="py-1.5 pr-4 tabular text-fg-soft">
                    {activeNames.has(r.name) ? (
                      <span className="text-accent-fg">training…</span>
                    ) : r.checkpoint ? (
                      <>
                        {Number(r.checkpoint).toLocaleString()}
                        {r.steps ? ` / ${r.steps.toLocaleString()}` : ""}
                      </>
                    ) : (
                      <span className="text-fg-faint">no checkpoint</span>
                    )}
                  </td>
                  <td className="py-1.5 pr-4 tabular text-fg-muted">
                    {new Date(r.modified * 1000).toLocaleString()}
                  </td>
                  <td className="py-1.5">
                    <div className="flex flex-wrap items-center gap-2">
                      <UploadStatus
                        upload={latestUpload(r.name)}
                        onDismiss={() => {
                          const up = latestUpload(r.name);
                          if (up)
                            dismissUpload(up.id)
                              .catch(() => {})
                              .finally(onUploadsChanged);
                        }}
                      />
                      {r.has_policy &&
                        !activeNames.has(r.name) &&
                        latestUpload(r.name)?.status !== "running" && (
                          <button
                            className="btn px-2 py-0.5"
                            onClick={() => setUploadRun(r.name)}
                          >
                            Upload
                          </button>
                        )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {uploadRun && (
        <UploadDialog
          run={uploadRun}
          onClose={() => setUploadRun(null)}
          onStarted={onUploadsChanged}
        />
      )}
    </section>
  );
}

function NewTrainingForm({
  options,
  existing,
  onQueued,
}: {
  options: TrainOptions;
  existing: TrainingRun[];
  onQueued: (job: TrainingJob) => void;
}) {
  const [repoId, setRepoId] = useState(options.datasets[0]?.repo_id ?? "");
  const [type, setType] = useState(options.types[0]?.type ?? "act");
  const [name, setName] = useState("");
  const [params, setParams] = useState<ParamValues>(DEFAULTS[type] ?? {});
  const [cameraMap, setCameraMap] = useState<Record<string, string>>({});
  const [nameProblem, setNameProblem] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [uploadOn, setUploadOn] = useState(false);
  const [upload, setUpload] = useState<UploadRequest>({
    repo_id: "",
    private: true,
  });

  const spec = options.types.find((t) => t.type === type);
  const dataset = options.datasets.find((d) => d.repo_id === repoId);

  // target_reach: the tag whose annotated frame is the training target.
  const isTargetReach = type === "target_reach";
  const { file: targetFile, error: targetError } = useTargetFile(
    repoId,
    isTargetReach,
  );
  const [targetTag, setTargetTag] = useState("");
  const tagStats = useMemo(() => {
    const stats: Record<string, { episodes: number; frames: number }> = {};
    for (const tag of targetFile?.tags ?? [])
      stats[tag] = { episodes: 0, frames: 0 };
    for (const ep of Object.values(targetFile?.episodes ?? {}))
      for (const t of ep.targets) {
        if (!stats[t.tag]) continue;
        stats[t.tag].episodes += 1;
        stats[t.tag].frames += t.frame + 1;
      }
    return stats;
  }, [targetFile]);
  useEffect(() => {
    if (!targetFile) return;
    setTargetTag((cur) =>
      tagStats[cur]?.episodes
        ? cur
        : (targetFile.tags.find((t) => tagStats[t].episodes) ?? ""),
    );
  }, [targetFile, tagStats]);

  useEffect(() => setParams(DEFAULTS[type] ?? {}), [type]);
  useEffect(() => {
    if (spec?.cameras && dataset)
      setCameraMap(defaultCameraMap(dataset.cameras, spec.cameras));
    else setCameraMap({});
  }, [spec, dataset]);

  // Live check of a manually entered name (also re-checked on submit).
  useEffect(() => {
    const trimmed = name.trim();
    if (!trimmed) {
      setNameProblem(null);
      return;
    }
    const timer = setTimeout(() => {
      api<{ problem: string | null }>(
        `/api/policy/train/name?name=${encodeURIComponent(trimmed)}`,
      )
        .then((r) => setNameProblem(r.problem))
        .catch(() => {});
    }, 250);
    return () => clearTimeout(timer);
  }, [name, existing]);

  const mappedTargets = Object.values(cameraMap).filter(Boolean);
  const duplicateTarget = mappedTargets.find(
    (t, i) => mappedTargets.indexOf(t) !== i,
  );
  const emptyCameras = spec?.cameras
    ? spec.cameras.filter((c) => !mappedTargets.includes(c)).length
    : 0;

  const chunk = Number(params.chunk_size);
  const nSteps = Number(params.n_action_steps);
  const formProblem = !repoId
    ? "Select a dataset"
    : isTargetReach && !targetTag
      ? "Annotate target frames with a tag in the episode viewer first"
      : isTargetReach && targetFile?.mismatched_episodes.length
        ? "The annotations do not match the dataset (episode lengths changed)"
        : spec?.cameras && mappedTargets.length === 0
          ? "Map at least one dataset camera to a policy camera"
          : duplicateTarget
            ? `Two cameras are mapped to ${duplicateTarget}`
            : chunk && nSteps && nSteps > chunk
              ? "Actions executed per chunk must not exceed the chunk size"
              : !params.steps
                ? "Set the number of training steps"
                : uploadOn && repoIdProblem(upload.repo_id)
                  ? "Fix the Hugging Face repository name"
                  : null;

  const submit = async () => {
    setSubmitting(true);
    setSubmitError(null);
    try {
      const fields = fieldsFor(type);
      const body = {
        policy_type: type,
        repo_id: repoId,
        name: name.trim(),
        params: Object.fromEntries(fields.map((f) => [f.key, params[f.key]])),
        rename_map: Object.fromEntries(
          Object.entries(cameraMap).filter(([k, v]) => v && v !== k),
        ),
        empty_cameras: emptyCameras,
        ...(isTargetReach ? { target_tag: targetTag } : {}),
        ...(uploadOn
          ? { upload: { ...upload, repo_id: upload.repo_id.trim() } }
          : {}),
      };
      const job = await api<TrainingJob>("/api/policy/train/jobs", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
      onQueued(job);
      setName("");
    } catch (e) {
      setSubmitError(String((e as Error).message ?? e));
    } finally {
      setSubmitting(false);
    }
  };

  const fields = fieldsFor(type);

  return (
    <section className={CARD_CLASS}>
      <h3 className="text-sm font-semibold text-fg">New training</h3>

      <div className="grid gap-4 sm:grid-cols-2">
        <div className="space-y-1">
          <label
            htmlFor="train-dataset"
            className="block text-xs text-fg-muted"
          >
            Dataset
          </label>
          <select
            id="train-dataset"
            className={`${INPUT_CLASS} font-mono`}
            value={repoId}
            onChange={(e) => setRepoId(e.target.value)}
          >
            {options.datasets.map((d) => (
              <option key={d.repo_id} value={d.repo_id}>
                {d.repo_id}
              </option>
            ))}
          </select>
          {dataset && (
            <p className="text-[11px] text-fg-faint">
              {dataset.episodes} episodes · {dataset.frames?.toLocaleString()}{" "}
              frames · {dataset.fps} fps
            </p>
          )}
        </div>

        <div className="space-y-1">
          <p className="text-xs text-fg-muted">Policy type</p>
          <div className="flex gap-1.5">
            {options.types.map((t) => (
              <button
                key={t.type}
                type="button"
                className={`btn flex-1 py-1.5 ${t.type === type ? "btn-active" : ""}`}
                onClick={() => setType(t.type)}
              >
                {t.label}
              </button>
            ))}
          </div>
          {spec?.base && (
            <p className="text-[11px] text-fg-faint">
              Fine-tunes <code>{spec.base}</code> (downloaded on first use)
              {type === "pi0_fast" &&
                ". Needs access to the gated google/paligemma-3b-pt-224 on the Hub"}
            </p>
          )}
        </div>
      </div>

      <div className="space-y-1">
        <label htmlFor="train-name" className="block text-xs text-fg-muted">
          Model name
        </label>
        <input
          id="train-name"
          className={`${INPUT_CLASS} font-mono`}
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder={`${exampleName(type, repoId, isTargetReach ? targetTag : undefined)} (default, timestamped when queued)`}
        />
        {nameProblem && <p className="text-xs text-flag-fg">{nameProblem}</p>}
      </div>

      {isTargetReach && (
        <div className="space-y-1">
          <label htmlFor="train-tag" className="block text-xs text-fg-muted">
            Target tag
          </label>
          <select
            id="train-tag"
            className={`${INPUT_CLASS} font-mono`}
            value={targetTag}
            onChange={(e) => setTargetTag(e.target.value)}
            disabled={!targetFile?.tags.length}
          >
            {!targetFile?.tags.length && <option value="">(no tags)</option>}
            {targetFile?.tags.map((tag) => (
              <option key={tag} value={tag} disabled={!tagStats[tag]?.episodes}>
                {tag}
              </option>
            ))}
          </select>
          <p className="text-[11px] text-fg-faint">
            {targetError
              ? targetError
              : targetTag && targetFile
                ? `Used in ${tagStats[targetTag].episodes}/${targetFile.num_episodes} episodes · ${tagStats[targetTag].frames.toLocaleString()} training frames. Frames 0…t of each episode learn the pose at its target frame t; episodes without the tag are skipped.`
                : "Add targets in the episode viewer (Target annotations panel)."}
          </p>
        </div>
      )}

      <div className="grid gap-x-4 gap-y-3 sm:grid-cols-2 lg:grid-cols-3">
        {fields.map((f) => (
          <ParamInput
            key={f.key}
            field={f}
            value={params[f.key]}
            onChange={(v) => setParams((p) => ({ ...p, [f.key]: v }))}
          />
        ))}
      </div>

      {spec?.cameras && dataset && (
        <div className="space-y-2">
          <p className="text-xs text-fg-muted">
            Camera mapping (dataset → {spec.label} input)
          </p>
          {dataset.cameras.map((cam) => (
            <div key={cam} className="flex items-center gap-2">
              <code className="min-w-0 flex-1 truncate text-xs text-fg-soft">
                {cam}
              </code>
              <span className="text-fg-faint">→</span>
              <select
                className={`${INPUT_CLASS} flex-1 font-mono`}
                value={cameraMap[cam] ?? ""}
                onChange={(e) =>
                  setCameraMap((m) => ({ ...m, [cam]: e.target.value }))
                }
              >
                <option value="">(not used)</option>
                {spec.cameras!.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </div>
          ))}
          <p className="text-[11px] text-fg-faint">
            {emptyCameras > 0
              ? `${emptyCameras} unmapped policy camera${emptyCameras > 1 ? "s" : ""} will be fed blank images (empty_cameras=${emptyCameras}).`
              : "All policy cameras are fed."}
          </p>
        </div>
      )}

      <UploadWhenDone
        enabled={uploadOn}
        onEnabled={setUploadOn}
        value={upload}
        onChange={setUpload}
        defaultName={name.trim() || "<run name>"}
      />

      <div className="flex flex-wrap items-center gap-3">
        <button
          className="btn btn-primary px-4 py-1.5 text-sm"
          disabled={submitting || !!formProblem || !!nameProblem}
          onClick={submit}
        >
          {submitting ? <Spinner /> : null}
          Add to queue
        </button>
        {(formProblem || submitError) && (
          <p className="text-xs text-flag-fg">{submitError ?? formProblem}</p>
        )}
      </div>
    </section>
  );
}

function ParamInput({
  field,
  value,
  onChange,
}: {
  field: Field;
  value: string | boolean | undefined;
  onChange: (v: string | boolean) => void;
}) {
  if (field.kind === "bool") {
    return (
      <label
        className="flex items-center gap-2 self-end pb-1.5 text-xs text-fg-soft"
        title={field.help}
      >
        <input
          type="checkbox"
          className="accent-accent"
          checked={!!value}
          onChange={(e) => onChange(e.target.checked)}
        />
        {field.label}
      </label>
    );
  }
  return (
    <label className="block space-y-1" title={field.help}>
      <span className="block text-xs text-fg-muted">{field.label}</span>
      {field.kind === "choice" ? (
        <select
          className={INPUT_CLASS}
          value={String(value ?? "")}
          onChange={(e) => onChange(e.target.value)}
        >
          {field.choices!.map((c) => (
            <option key={c}>{c}</option>
          ))}
        </select>
      ) : (
        <input
          className={`${INPUT_CLASS} tabular`}
          type="number"
          min={field.key === "seed" || field.key === "num_workers" ? 0 : 1}
          step={field.kind === "float" ? "any" : 1}
          placeholder={field.kind === "float" ? "default" : undefined}
          value={String(value ?? "")}
          onChange={(e) => onChange(e.target.value)}
        />
      )}
    </label>
  );
}

type Confirm =
  | { kind: "cancel"; job: TrainingJob }
  | { kind: "stop"; job: TrainingJob };

function QueueSection({
  jobs,
  uploads,
  onChanged,
}: {
  jobs: TrainingJob[] | null;
  uploads: HfUpload[];
  onChanged: () => void;
}) {
  const [confirm, setConfirm] = useState<Confirm | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const act = async (url: string, init: RequestInit) => {
    setActionError(null);
    try {
      await api(url, init);
      onChanged();
    } catch (e) {
      setActionError(String((e as Error).message ?? e));
    }
  };
  const remove = (job: TrainingJob) =>
    act(`/api/policy/train/jobs/${job.id}`, { method: "DELETE" });
  const stop = (job: TrainingJob, mode: "keep" | "delete") =>
    act(`/api/policy/train/jobs/${job.id}/stop`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ mode }),
    });

  const active = (jobs ?? []).filter((j) =>
    ["running", "stopping", "queued"].includes(j.status),
  );
  const finished = (jobs ?? []).filter(
    (j) => !["running", "stopping", "queued"].includes(j.status),
  );

  return (
    <section className={CARD_CLASS}>
      <h3 className="text-sm font-semibold text-fg">Training queue</h3>
      {actionError && <p className="text-xs text-flag-fg">{actionError}</p>}
      {!jobs ? (
        <p className="flex items-center gap-2 text-xs text-fg-muted">
          <Spinner /> Loading…
        </p>
      ) : active.length === 0 ? (
        <p className="text-xs text-fg-muted">
          Nothing running or queued. Add a training below.
        </p>
      ) : (
        <div className="space-y-3">
          {active.map((job) => (
            <JobCard
              key={job.id}
              job={job}
              position={active
                .filter((j) => j.status === "queued")
                .indexOf(job)}
              onCancel={() => setConfirm({ kind: "cancel", job })}
              onStop={() => setConfirm({ kind: "stop", job })}
            />
          ))}
        </div>
      )}

      {finished.length > 0 && (
        <div className="space-y-3 border-t border-line-subtle pt-4">
          <h4 className="text-xs font-medium uppercase tracking-wide text-fg-faint">
            Finished
          </h4>
          {[...finished].reverse().map((job) => (
            <JobCard
              key={job.id}
              job={job}
              upload={uploads.find((u) => u.job_id === job.id)}
              onRemove={() => remove(job)}
            />
          ))}
        </div>
      )}

      {confirm && (
        <ConfirmDialog
          confirm={confirm}
          onClose={() => setConfirm(null)}
          onCancelJob={() => {
            remove(confirm.job);
            setConfirm(null);
          }}
          onStopJob={(mode) => {
            stop(confirm.job, mode);
            setConfirm(null);
          }}
        />
      )}
    </section>
  );
}

const STATUS_STYLE: Record<TrainingJob["status"], string> = {
  queued: "text-fg-muted",
  running: "text-accent-fg",
  stopping: "text-accent-fg",
  done: "text-fg-soft",
  failed: "text-red-500",
  stopped: "text-fg-muted",
};

function JobCard({
  job,
  upload,
  position,
  onCancel,
  onStop,
  onRemove,
}: {
  job: TrainingJob;
  upload?: HfUpload;
  position?: number | null;
  onCancel?: () => void;
  onStop?: () => void;
  onRemove?: () => void;
}) {
  const [showLog, setShowLog] = useState(false);
  const p = job.progress;
  const total = p?.total || Number(job.params.steps) || 0;
  const pct = total && p ? Math.min(100, (p.step / total) * 100) : 0;
  const running = job.status === "running" || job.status === "stopping";
  const lastLoss = p?.losses.length ? p.losses[p.losses.length - 1][1] : null;

  let statusText: string = job.status;
  if (job.status === "queued")
    statusText = position ? `queued (#${position + 1})` : "queued (next)";
  if (job.status === "running" && p?.phase === "preparing")
    statusText = "preparing (loading data / base model)";
  if (job.status === "stopping") statusText = "stopping…";
  if (job.status === "stopped")
    statusText = job.deleted ? "stopped, deleted" : "stopped, checkpoints kept";

  return (
    <div className="space-y-3 rounded-md border border-line-subtle bg-[var(--surface-0)] p-3">
      <div className="flex flex-wrap items-start gap-x-4 gap-y-1">
        <div className="min-w-0 flex-1">
          <p className="truncate font-mono text-sm text-fg">{job.name}</p>
          <p className="text-[11px] text-fg-muted">
            {job.policy_type} · <span className="font-mono">{job.repo_id}</span>{" "}
            {job.target_tag && (
              <>
                · tag <span className="font-mono">{job.target_tag}</span>{" "}
              </>
            )}
            · {Number(job.params.steps).toLocaleString()} steps · batch{" "}
            {String(job.params.batch_size ?? "default")}
            {job.upload && (
              <>
                {" "}
                · upload to{" "}
                <span className="font-mono">
                  {job.upload.repo_id || "<user>/" + job.name}
                </span>
                {job.upload.private ? " (private)" : ""}
              </>
            )}
          </p>
          {(upload || job.upload_error) && (
            <div className="mt-1 text-[11px]">
              {upload ? (
                <UploadStatus upload={upload} />
              ) : (
                <span className="text-red-500">
                  upload not started: {job.upload_error}
                </span>
              )}
            </div>
          )}
        </div>
        <span className={`text-xs ${STATUS_STYLE[job.status]}`}>
          {statusText}
        </span>
        {job.status === "queued" && onCancel && (
          <button className="btn" onClick={onCancel}>
            Cancel
          </button>
        )}
        {job.status === "running" && onStop && (
          <button className="btn" onClick={onStop}>
            Stop
          </button>
        )}
        {onRemove && (
          <button className="btn" onClick={onRemove} title="Remove from list">
            Dismiss
          </button>
        )}
      </div>

      {p && job.status !== "queued" && (
        <>
          <div className="space-y-1">
            <div className="h-1.5 w-full overflow-hidden rounded bg-fill">
              <div
                className={`h-full ${job.status === "failed" ? "bg-red-500" : "bg-accent"} transition-all`}
                style={{ width: `${pct}%` }}
              />
            </div>
            <div className="flex flex-wrap gap-x-4 text-[11px] tabular text-fg-muted">
              <span>
                step {p.step.toLocaleString()} / {total.toLocaleString()} (
                {pct.toFixed(1)}%)
              </span>
              {lastLoss != null && <span>loss {lastLoss.toFixed(4)}</span>}
              {p.elapsed_s != null && (
                <span>elapsed {formatDuration(p.elapsed_s)}</span>
              )}
              {running && p.remaining_s != null && p.step > 0 && (
                <span className="text-fg-soft">
                  ~{formatDuration(p.remaining_s)} left · done ≈{" "}
                  {formatClock(Date.now() + p.remaining_s * 1000)}
                </span>
              )}
              {job.finished_at && (
                <span>finished {formatClock(job.finished_at * 1000)}</span>
              )}
            </div>
          </div>
          {p.losses.length > 1 && <LossChart losses={p.losses} />}
        </>
      )}

      {p?.note && (
        <p className="font-mono text-[11px] text-fg-muted">{p.note}</p>
      )}
      {job.error && (
        <p className="break-words font-mono text-[11px] text-red-500">
          {job.error}
        </p>
      )}
      {p && p.tail.length > 0 && job.status !== "queued" && (
        <div>
          <button
            className="text-[11px] text-fg-faint hover:text-fg"
            onClick={() => setShowLog((s) => !s)}
          >
            {showLog ? "Hide log" : "Show log"}
          </button>
          {showLog && (
            <pre className="mt-1 max-h-60 overflow-auto rounded bg-fill p-2 text-[10px] leading-snug text-fg-soft">
              {p.tail.join("\n")}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

function LossChart({ losses }: { losses: [number, number][] }) {
  const data = losses.map(([step, loss]) => ({ step, loss }));
  const logScale = losses.every(([, l]) => l > 0);
  return (
    <div className="h-40 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={data}
          margin={{ top: 4, right: 8, bottom: 0, left: 0 }}
        >
          <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="3 3" />
          <XAxis
            dataKey="step"
            type="number"
            domain={["dataMin", "dataMax"]}
            tick={{ fontSize: 10, fill: "var(--fg-muted)" }}
            stroke="var(--chart-axis)"
          />
          <YAxis
            scale={logScale ? "log" : "auto"}
            domain={["auto", "auto"]}
            allowDataOverflow
            width={48}
            tick={{ fontSize: 10, fill: "var(--fg-muted)" }}
            tickFormatter={(v: number) => Number(v.toPrecision(3)).toString()}
            stroke="var(--chart-axis)"
          />
          <Tooltip
            contentStyle={{
              background: "var(--surface-1)",
              border: "1px solid var(--line)",
              fontSize: 11,
            }}
            formatter={(v: number) => [v.toFixed(4), "loss"]}
            labelFormatter={(s) => `step ${s}`}
          />
          <Line
            type="linear"
            dataKey="loss"
            stroke="var(--accent)"
            dot={false}
            strokeWidth={1.5}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function ConfirmDialog({
  confirm,
  onClose,
  onCancelJob,
  onStopJob,
}: {
  confirm: Confirm;
  onClose: () => void;
  onCancelJob: () => void;
  onStopJob: (mode: "keep" | "delete") => void;
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-modal="true"
        className="w-full max-w-md space-y-4 rounded-lg border border-line bg-[var(--surface-1)] p-5 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        {confirm.kind === "cancel" ? (
          <>
            <h4 className="text-sm font-semibold text-fg">
              Cancel queued training?
            </h4>
            <p className="text-xs text-fg-muted">
              <span className="font-mono">{confirm.job.name}</span> will be
              removed from the queue. Nothing has been trained for it yet.
            </p>
            <div className="flex flex-wrap justify-end gap-2">
              <button className="btn px-3 py-1.5" onClick={onClose} autoFocus>
                Keep in queue
              </button>
              <button
                className="btn btn-primary px-3 py-1.5"
                onClick={onCancelJob}
              >
                Cancel training
              </button>
            </div>
          </>
        ) : (
          <>
            <h4 className="text-sm font-semibold text-fg">
              Stop running training?
            </h4>
            <p className="text-xs text-fg-muted">
              <span className="font-mono">{confirm.job.name}</span> is at step{" "}
              {confirm.job.progress?.step.toLocaleString() ?? 0}. Checkpoints
              are saved every{" "}
              {Number(confirm.job.params.save_freq ?? 0).toLocaleString()}{" "}
              steps; progress since the last one is lost.
            </p>
            <div className="flex flex-col gap-2">
              <button className="btn px-3 py-1.5" onClick={onClose} autoFocus>
                Continue training
              </button>
              <button
                className="btn px-3 py-1.5"
                onClick={() => onStopJob("keep")}
              >
                Stop and keep checkpoints
              </button>
              <button
                className="inline-flex items-center justify-center rounded-md border border-red-700 bg-red-600 px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-red-700"
                onClick={() => onStopJob("delete")}
              >
                Stop and delete previous checkpoints
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
