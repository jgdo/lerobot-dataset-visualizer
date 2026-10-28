"use client";

import React, { useEffect, useState } from "react";
import { Spinner } from "@/components/ui";

// Talks to backend/policy_server.py through the /api/policy rewrite in
// next.config.ts; only rendered when POLICY_BACKEND_URL was set at startup.
export const POLICY_BACKEND_ENABLED = !!process.env.POLICY_BACKEND_ENABLED;

const POLICY_KEY = "lerobot-viz-policy";
const DEFAULT_ACTIONS_PER_CHUNK = 25;
const POLL_MS = 400;

/** A training run under train/ whose `last` checkpoint is used. */
type TrainedPolicy = { name: string; checkpoint: string };

type PolicyInfo = {
  type: string;
  chunk_size: number;
  input_features: { key: string; type: string }[];
  /** Renames the policy was trained with (dataset key -> policy key). */
  default_remap: Record<string, string>;
  dataset_observation_keys: string[];
};

/** Dataset observation key -> policy input key; "" leaves the stream out. */
type Remap = Record<string, string>;

// What each dataset stream feeds before any user choice: the saved training
// rename, else the policy input of the same name, else nothing.
function defaultRemap(info: PolicyInfo): Remap {
  const inputs = new Set(info.input_features.map((f) => f.key));
  return Object.fromEntries(
    info.dataset_observation_keys.map((key) => {
      const saved = info.default_remap[key];
      return [
        key,
        saved && inputs.has(saved) ? saved : inputs.has(key) ? key : "",
      ];
    }),
  );
}

type JointStats = {
  name: string;
  mae: number;
  rmse: number;
  max_abs: number;
  recorded_range: number;
};

export type PolicyResult = {
  episode: number;
  policy: string;
  checkpoint: string;
  policy_type: string;
  chunk_size: number;
  actions_per_chunk: number;
  num_policy_calls: number;
  inputs: Record<string, string>;
  names: string[];
  timestamps: number[];
  recorded: number[][];
  predicted: number[][];
  per_joint: JointStats[];
};

type Job = {
  id: string;
  status: "queued" | "loading" | "running" | "done" | "error" | "cancelled";
  message: string;
  progress: number;
  total: number;
  result?: PolicyResult;
};

async function api<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, { cache: "no-store", ...init });
  } catch {
    throw new Error("Policy service not reachable");
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    throw new Error(
      body?.error ??
        (res.status >= 500
          ? "Policy service not reachable (is backend/policy_server.py running?)"
          : `HTTP ${res.status}`),
    );
  }
  return body as T;
}

function readStored(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeStored(key: string, value: string) {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Private mode etc.; the value just isn't remembered.
  }
}

/** One row per dataset observation stream, choosing the policy input it feeds. */
function InputRemapping({
  info,
  remap,
  onChange,
  disabled,
}: {
  info: PolicyInfo;
  remap: Remap;
  onChange: (remap: Remap) => void;
  disabled: boolean;
}) {
  const fed = new Set(Object.values(remap).filter(Boolean));
  const unfed = info.input_features.filter((f) => !fed.has(f.key));
  const duplicates = info.input_features.filter(
    (f) => Object.values(remap).filter((v) => v === f.key).length > 1,
  );
  const reset = defaultRemap(info);
  const isDefault = info.dataset_observation_keys.every(
    (k) => remap[k] === reset[k],
  );
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <label className="text-xs text-fg-muted">
          Input remapping (dataset stream → policy input)
        </label>
        {!isDefault && (
          <button
            className="btn ml-auto py-0.5 text-xs"
            onClick={() => onChange(reset)}
            disabled={disabled}
          >
            Reset
          </button>
        )}
      </div>
      <table className="w-full text-xs">
        <tbody>
          {info.dataset_observation_keys.map((key) => (
            <tr key={key}>
              <td className="py-0.5 pr-2 font-mono text-fg-soft">{key}</td>
              <td className="w-6 text-fg-faint">→</td>
              <td className="py-0.5">
                <select
                  className="w-full rounded-md border border-line bg-[var(--surface-0)] px-2 py-1 font-mono text-xs text-fg outline-none focus:border-accent"
                  value={remap[key] ?? ""}
                  onChange={(e) =>
                    onChange({ ...remap, [key]: e.target.value })
                  }
                  disabled={disabled}
                >
                  <option value="">— not used —</option>
                  {info.input_features.map((f) => (
                    <option key={f.key} value={f.key}>
                      {f.key}
                    </option>
                  ))}
                </select>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {duplicates.length > 0 && (
        <p className="text-xs text-flag-fg">
          More than one stream feeds {duplicates.map((f) => f.key).join(", ")}.
        </p>
      )}
      {unfed.length > 0 && (
        <p className="text-xs text-fg-faint">
          Not fed from the dataset: {unfed.map((f) => f.key).join(", ")}
          {unfed.some((f) => f.type === "VISUAL") &&
            " (missing cameras are left to the policy, e.g. padded as empty)"}
        </p>
      )}
    </div>
  );
}

type ChartRow = Record<string, number | Record<string, number>>;

const SERIES_NAME_DELIMITER = " | ";

/** Joint names of the recorded `action` series in a chart row. */
function actionJoints(row: ChartRow | undefined): string[] {
  if (!row) return [];
  const prefix = `action${SERIES_NAME_DELIMITER}`;
  const nested = row.action;
  return [
    ...Object.keys(row)
      .filter((k) => k.startsWith(prefix))
      .map((k) => k.slice(prefix.length)),
    ...(nested && typeof nested === "object" ? Object.keys(nested) : []),
  ];
}

/**
 * Adds the predicted actions to the episode's chart data as a "predicted"
 * group, next to each joint's recorded `action` series (so they also share a
 * chart in the split view). Rows are matched by frame position, not
 * timestamp: the episode chart stretches its timestamps over the video
 * segment, so they differ slightly from the dataset's.
 */
export function withPredictedActions(
  groups: ChartRow[][],
  result: PolicyResult | null,
): ChartRow[][] {
  if (!result || result.predicted.length === 0 || groups.length === 0) {
    return groups;
  }
  const predictedFor = (
    names: string[],
    rowIndex: number,
    rowCount: number,
  ) => {
    const last = result.predicted.length - 1;
    const i = rowCount > 1 ? Math.round((rowIndex * last) / (rowCount - 1)) : 0;
    // Flat "predicted | <joint>" keys, like the recorded series, so merging
    // the charts in the combined view keeps every joint.
    return Object.fromEntries(
      names.map((name) => [
        `predicted${SERIES_NAME_DELIMITER}${name}`,
        result.predicted[i][result.names.indexOf(name)],
      ]),
    );
  };

  const placed = new Set<string>();
  const out = groups.map((group) => {
    const names = actionJoints(group[0]).filter((n) =>
      result.names.includes(n),
    );
    if (names.length === 0) return group;
    names.forEach((n) => placed.add(n));
    return group.map((row, i) => ({
      ...row,
      ...predictedFor(names, i, group.length),
    }));
  });
  // Joints without a recorded series of the same name get a chart of their own.
  const rest = result.names.filter((n) => !placed.has(n));
  if (rest.length === 0) return out;
  const base = groups[0];
  return [
    ...out,
    base.map((row, i) => ({
      timestamp: row.timestamp,
      ...predictedFor(rest, i, base.length),
    })),
  ];
}

const fmt = (v: number) =>
  Math.abs(v) >= 100
    ? v.toFixed(1)
    : Math.abs(v) >= 1
      ? v.toFixed(3)
      : v.toFixed(4);

export function PolicyPanel({
  repoId,
  episodeId,
  onResult,
}: {
  repoId: string;
  episodeId: number;
  /** The finished run's result, or null while there is none (shown in the episode charts). */
  onResult: (result: PolicyResult | null) => void;
}) {
  const [open, setOpen] = useState(false);
  const [policies, setPolicies] = useState<TrainedPolicy[] | null>(null);
  const [policy, setPolicy] = useState("");
  const [actionsPerChunk, setActionsPerChunk] = useState(
    DEFAULT_ACTIONS_PER_CHUNK,
  );
  const [info, setInfo] = useState<PolicyInfo | null>(null);
  const [remap, setRemap] = useState<Remap>({});
  const [infoError, setInfoError] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Results belong to one episode; navigating away cancels and clears them.
  useEffect(() => {
    setJob(null);
    setError(null);
  }, [episodeId, repoId]);

  // The trained policies, listed again each time the panel opens so new
  // training runs show up; preselects the last used one.
  useEffect(() => {
    if (!open) return;
    api<TrainedPolicy[]>("/api/policy/policies")
      .then((list) => {
        setPolicies(list);
        setPolicy((current) => {
          const names = list.map((p) => p.name);
          if (names.includes(current)) return current;
          const stored = readStored(POLICY_KEY) ?? "";
          return names.includes(stored) ? stored : (names[0] ?? "");
        });
      })
      .catch((e) => setInfoError(String(e.message ?? e)));
  }, [open]);

  useEffect(() => {
    setInfo(null);
    setInfoError(null);
    if (!open || !policy) return;
    api<PolicyInfo>(
      `/api/policy/info?policy=${encodeURIComponent(policy)}&repo_id=${encodeURIComponent(repoId)}`,
    )
      .then((i) => {
        setInfo(i);
        setRemap(defaultRemap(i));
        // Policies with shorter chunks can't use the default.
        setActionsPerChunk((k) => Math.min(k, i.chunk_size));
      })
      .catch((e) => setInfoError(String(e.message ?? e)));
  }, [policy, repoId, open]);

  const running =
    !!job &&
    (job.status === "queued" ||
      job.status === "loading" ||
      job.status === "running");

  // Poll the job until it settles.
  useEffect(() => {
    if (!job || !running) return;
    let stopped = false;
    const timer = setTimeout(async () => {
      try {
        const next = await api<Job>(`/api/policy/jobs/${job.id}`);
        if (!stopped) setJob(next);
      } catch (e) {
        if (!stopped) {
          setError(String((e as Error).message ?? e));
          setJob(null);
        }
      }
    }, POLL_MS);
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [job, running]);

  // Cancel a running job when leaving the episode or the page.
  useEffect(() => {
    if (!job || !running) return;
    const id = job.id;
    return () => {
      fetch(`/api/policy/jobs/${id}`, { method: "DELETE" }).catch(() => {});
    };
    // Only re-run when the job id changes, not on every progress update.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.id, running]);

  const maxK = info?.chunk_size ?? 1000;
  const kValid = actionsPerChunk >= 1 && actionsPerChunk <= maxK;

  const start = async () => {
    setError(null);
    writeStored(POLICY_KEY, policy);
    try {
      const created = await api<Job>("/api/policy/jobs", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          policy,
          repo_id: repoId,
          episode: episodeId,
          actions_per_chunk: actionsPerChunk,
          remap,
        }),
      });
      setJob(created);
    } catch (e) {
      setError(String((e as Error).message ?? e));
    }
  };

  const cancel = () => {
    if (!job) return;
    fetch(`/api/policy/jobs/${job.id}`, { method: "DELETE" }).catch(() => {});
  };

  const result = job?.status === "done" ? job.result : undefined;
  useEffect(() => {
    onResult(result ?? null);
  }, [result, onResult]);

  if (!open) {
    return (
      <div>
        <button className="btn" onClick={() => setOpen(true)}>
          Run trained policy
        </button>
      </div>
    );
  }

  const pct =
    job && job.total > 0 ? Math.round((job.progress / job.total) * 100) : 0;

  return (
    <div className="panel space-y-4 p-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-fg">
          Trained policy vs. recorded actions
        </h3>
        <button className="btn" onClick={() => setOpen(false)}>
          Hide
        </button>
      </div>

      <div className="space-y-2">
        <label className="block text-xs text-fg-muted">
          Trained policy (latest checkpoint)
        </label>
        <select
          className="w-full rounded-md border border-line bg-[var(--surface-0)] px-2 py-1.5 font-mono text-xs text-fg outline-none focus:border-accent"
          value={policy}
          onChange={(e) => setPolicy(e.target.value)}
          disabled={running || !policies?.length}
        >
          {!policies && <option value="">Loading…</option>}
          {policies?.length === 0 && (
            <option value="">No trained policies found</option>
          )}
          {policies?.map((p) => (
            <option key={p.name} value={p.name}>
              {p.name} (checkpoint {p.checkpoint})
            </option>
          ))}
        </select>
        {policies?.length === 0 && (
          <p className="text-xs text-fg-faint">
            Looked for{" "}
            <code>train/&lt;run&gt;/checkpoints/last/pretrained_model</code>{" "}
            under LOCAL_LEROBOT_DATA_ROOT.
          </p>
        )}
        {info && (
          <p className="text-xs text-fg-muted">
            {info.type} policy · predicts {info.chunk_size} actions per chunk
          </p>
        )}
        {infoError && <p className="text-xs text-flag-fg">{infoError}</p>}
      </div>

      {info && info.dataset_observation_keys.length > 0 && (
        <InputRemapping
          info={info}
          remap={remap}
          onChange={setRemap}
          disabled={running}
        />
      )}

      <div className="flex flex-wrap items-end gap-4">
        <div>
          <label className="block text-xs text-fg-muted">
            Actions used per chunk
          </label>
          <input
            type="number"
            min={1}
            max={maxK}
            value={actionsPerChunk}
            onChange={(e) => setActionsPerChunk(parseInt(e.target.value, 10))}
            disabled={running}
            className="mt-1 w-24 rounded-md border border-line bg-[var(--surface-0)] px-2 py-1.5 text-sm text-fg tabular outline-none focus:border-accent"
          />
        </div>
        <p className="max-w-md flex-1 text-xs text-fg-faint">
          {actionsPerChunk === 1 || !kValid
            ? "The policy runs on every frame of the episode and its first predicted action is compared with the recorded action."
            : `The policy runs on every ${actionsPerChunk}th frame of the episode and the first ${actionsPerChunk} predicted actions of each chunk are compared with the recorded actions.`}
        </p>
        {running ? (
          <button className="btn" onClick={cancel}>
            Cancel
          </button>
        ) : (
          <button
            className="btn btn-primary"
            onClick={start}
            disabled={!policy || !info || !kValid}
          >
            Run on episode {episodeId}
          </button>
        )}
      </div>
      {!kValid && info && (
        <p className="text-xs text-flag-fg">
          Choose between 1 and {info.chunk_size}.
        </p>
      )}

      {running && job && (
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-xs text-fg-muted">
            <Spinner className="h-3 w-3" />
            <span>
              {job.status === "queued"
                ? "Waiting for another run to finish"
                : job.message}
            </span>
            {job.total > 0 && (
              <span className="ml-auto tabular">
                {job.progress} / {job.total} frames ({pct}%)
              </span>
            )}
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-fill">
            <div
              className="h-full bg-accent transition-[width] duration-300"
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      )}

      {job?.status === "cancelled" && (
        <p className="text-xs text-fg-muted">Run cancelled.</p>
      )}
      {(error || job?.status === "error") && (
        <p className="whitespace-pre-wrap text-xs text-flag-fg">
          {error ?? job?.message}
        </p>
      )}

      {result && (
        <div className="space-y-4">
          <p className="text-xs text-fg-muted">
            {result.policy_type} · {result.num_policy_calls} policy calls ·{" "}
            {result.actions_per_chunk} action
            {result.actions_per_chunk === 1 ? "" : "s"} per chunk · predicted
            minus recorded action · predicted actions are shown in the chart
            above
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-xs tabular">
              <thead>
                <tr className="text-left text-fg-faint">
                  <th className="py-1 pr-4 font-medium">Joint</th>
                  <th className="py-1 pr-4 font-medium">Mean abs. error</th>
                  <th className="py-1 pr-4 font-medium">RMSE</th>
                  <th className="py-1 pr-4 font-medium">Max abs. error</th>
                  <th className="py-1 pr-4 font-medium">Recorded range</th>
                </tr>
              </thead>
              <tbody>
                {result.per_joint.map((j) => (
                  <tr key={j.name} className="border-t border-line-subtle">
                    <td className="py-1 pr-4 text-fg">{j.name}</td>
                    <td className="py-1 pr-4 text-fg-soft">{fmt(j.mae)}</td>
                    <td className="py-1 pr-4 text-fg-soft">{fmt(j.rmse)}</td>
                    <td className="py-1 pr-4 text-fg-soft">{fmt(j.max_abs)}</td>
                    <td className="py-1 pr-4 text-fg-soft">
                      {fmt(j.recorded_range)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
