"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTime, useTimeControls } from "@/context/time-context";
import { api } from "@/components/policy-panel";
import type { ChartMarker } from "@/components/data-recharts";

// Per-episode target frames, stored by backend/target_annotations.py in
// <root>/annotations/<org>/<dataset>/targets.json (never inside the dataset).

export type TargetType = "lin" | "circ" | "coll";
export type Target = { frame: number; type: TargetType; tag: string };
export type TargetFile = {
  version: number;
  types: TargetType[];
  tags: string[];
  episodes: Record<string, { length: number; targets: Target[] }>;
  mismatched_episodes: number[];
  num_episodes: number;
};

const BASE = "/api/policy/annotations";
const TYPE_COLORS: Record<TargetType, string> = {
  lin: "#22c55e",
  circ: "#3b82f6",
  coll: "#a855f7",
};
const DEFAULT_KEY = "lerobot-viz-target-defaults";

export function useTargetFile(repoId: string, enabled: boolean) {
  const [file, setFile] = useState<TargetFile | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    api<TargetFile>(`${BASE}/targets?repo_id=${encodeURIComponent(repoId)}`)
      .then((f) => !cancelled && (setFile(f), setError(null)))
      .catch((e: Error) => !cancelled && setError(e.message));
    return () => {
      cancelled = true;
    };
  }, [repoId, enabled]);
  return { file, setFile, error, setError };
}

/** Chart markers for one episode; `timestamps[i]` is chart row i's time. */
export function targetMarkers(
  targets: Target[],
  timestamps: number[],
): ChartMarker[] {
  return targets
    .filter((t) => t.frame < timestamps.length)
    .map((t) => ({
      time: timestamps[t.frame],
      label: `${t.tag} (${t.type})`,
      color: TYPE_COLORS[t.type],
    }));
}

export function nearestFrame(timestamps: number[], time: number): number {
  let best = 0;
  for (let i = 1; i < timestamps.length; i++) {
    if (Math.abs(timestamps[i] - time) < Math.abs(timestamps[best] - time)) {
      best = i;
    }
  }
  return best;
}

function isTextEntry(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return (
    target.isContentEditable ||
    ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)
  );
}

function CurrentFrame({ timestamps }: { timestamps: number[] }) {
  const { currentTime } = useTime();
  return <>{nearestFrame(timestamps, currentTime)}</>;
}

type Confirm =
  | { kind: "target"; frame: number }
  | { kind: "tag"; tag: string; targets: number; episodes: number };

export function TargetAnnotationsPanel({
  repoId,
  episodeId,
  timestamps,
  file,
  onFile,
}: {
  repoId: string;
  episodeId: number;
  timestamps: number[];
  file: TargetFile | null;
  onFile: (f: TargetFile) => void;
}) {
  const { seek, subscribe } = useTimeControls();
  const timeRef = useRef(0);
  useEffect(() => subscribe((t) => (timeRef.current = t)), [subscribe]);

  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [showTags, setShowTags] = useState(false);
  const [newTag, setNewTag] = useState("");
  const [confirm, setConfirm] = useState<Confirm | null>(null);

  const targets = useMemo(
    () => file?.episodes[String(episodeId)]?.targets ?? [],
    [file, episodeId],
  );
  const tags = useMemo(() => file?.tags ?? [], [file]);
  const types = file?.types ?? ["lin", "circ", "coll"];
  const mismatched = file?.mismatched_episodes.includes(episodeId) ?? false;

  const q = `repo_id=${encodeURIComponent(repoId)}`;

  const save = useCallback(
    async (next: Target[]) => {
      setBusy(true);
      try {
        onFile(
          await api<TargetFile>(`${BASE}/targets?${q}&episode=${episodeId}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ targets: next }),
          }),
        );
        setError(null);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setBusy(false);
      }
    },
    [q, episodeId, onFile],
  );

  const postTag = async (body: { add?: string; remove?: string }) => {
    setBusy(true);
    try {
      onFile(
        await api<TargetFile>(`${BASE}/tags?${q}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }),
      );
      setError(null);
      return true;
    } catch (e) {
      setError((e as Error).message);
      return false;
    } finally {
      setBusy(false);
    }
  };

  const currentFrame = useCallback(
    () => nearestFrame(timestamps, timeRef.current),
    [timestamps],
  );

  const addTarget = useCallback(() => {
    const frame = currentFrame();
    if (targets.some((t) => t.frame === frame)) {
      setError(`Frame ${frame} already has a target`);
      return;
    }
    const used = new Set(targets.map((t) => t.tag));
    let defaults: { type?: TargetType; tag?: string } = {};
    try {
      defaults = JSON.parse(localStorage.getItem(DEFAULT_KEY) ?? "{}");
    } catch {}
    const tag =
      defaults.tag && tags.includes(defaults.tag) && !used.has(defaults.tag)
        ? defaults.tag
        : tags.find((t) => !used.has(t));
    if (!tag) {
      setError(
        tags.length
          ? "Every tag is already used in this episode"
          : "Add a tag first (Tags below)",
      );
      if (!tags.length) setShowTags(true);
      return;
    }
    void save([...targets, { frame, type: defaults.type ?? "lin", tag }]);
  }, [currentFrame, targets, tags, save]);

  const update = (frame: number, patch: Partial<Target>) => {
    const next = targets.map((t) =>
      t.frame === frame ? { ...t, ...patch } : t,
    );
    const changed = next.find((t) => t.frame === (patch.frame ?? frame));
    if (changed) {
      try {
        localStorage.setItem(
          DEFAULT_KEY,
          JSON.stringify({ type: changed.type, tag: changed.tag }),
        );
      } catch {}
    }
    void save(next);
  };

  const setToCurrent = (frame: number) => {
    const to = currentFrame();
    if (to === frame) return;
    if (targets.some((t) => t.frame === to)) {
      setError(`Frame ${to} already has a target`);
      return;
    }
    update(frame, { frame: to });
  };

  const jump = useCallback(
    (frame: number) => {
      if (frame < timestamps.length) seek(timestamps[frame]);
    },
    [seek, timestamps],
  );

  const askRemoveTag = async (tag: string) => {
    try {
      const usage = await api<{ targets: number; episodes: number }>(
        `${BASE}/tags/usage?${q}&tag=${encodeURIComponent(tag)}`,
      );
      setConfirm({ kind: "tag", tag, ...usage });
    } catch (e) {
      setError((e as Error).message);
    }
  };

  // t: add target; [ / ]: previous / next target.
  const keyRef = useRef({ addTarget, jump, targets, currentFrame });
  keyRef.current = { addTarget, jump, targets, currentFrame };
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey || isTextEntry(e.target)) return;
      const s = keyRef.current;
      if (e.key === "t") {
        e.preventDefault();
        s.addTarget();
      } else if (e.key === "[" || e.key === "]") {
        e.preventDefault();
        const now = s.currentFrame();
        const frames = s.targets.map((t) => t.frame);
        const to =
          e.key === "]"
            ? frames.find((f) => f > now)
            : frames.filter((f) => f < now).at(-1);
        if (to !== undefined) s.jump(to);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const tagCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const ep of Object.values(file?.episodes ?? {})) {
      for (const t of ep.targets) {
        counts.set(t.tag, (counts.get(t.tag) ?? 0) + 1);
      }
    }
    return counts;
  }, [file]);

  return (
    <div className="panel space-y-3 p-4">
      <div className="flex flex-wrap items-center gap-3">
        <h3 className="text-sm font-semibold text-fg">Target positions</h3>
        <span className="text-xs text-fg-muted tabular">
          frame <CurrentFrame timestamps={timestamps} />
        </span>
        <button
          className="btn btn-primary ml-auto px-3 py-1"
          onClick={addTarget}
          disabled={busy || !file}
          title="Add a target at the current frame (t). [ and ] jump between targets."
        >
          + Target at current frame
        </button>
      </div>

      {mismatched && (
        <p className="text-xs text-fg-muted">
          ⚠ The episode length changed since these targets were saved; check
          their frames.
        </p>
      )}
      {error && <p className="text-xs text-red-500">{error}</p>}

      {targets.length === 0 ? (
        <p className="text-xs text-fg-muted">
          No targets in this episode. Press <kbd>t</kbd> to add one at the
          current frame.
        </p>
      ) : (
        <table className="w-full text-xs tabular">
          <thead className="text-left text-[10px] uppercase tracking-wide text-fg-faint">
            <tr>
              <th className="py-1 font-normal">Frame</th>
              <th className="py-1 font-normal">Type</th>
              <th className="py-1 font-normal">Tag</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {targets.map((t) => (
              <tr key={t.frame} className="border-t border-line-subtle">
                <td className="py-1.5 pr-3 text-fg">
                  <span
                    className="mr-2 inline-block h-2 w-2 rounded-full"
                    style={{ background: TYPE_COLORS[t.type] }}
                  />
                  {t.frame}
                </td>
                <td className="py-1.5 pr-3">
                  <select
                    className="rounded border border-line bg-[var(--surface-1)] px-1.5 py-0.5 text-fg"
                    value={t.type}
                    disabled={busy}
                    onChange={(e) =>
                      update(t.frame, { type: e.target.value as TargetType })
                    }
                  >
                    {types.map((ty) => (
                      <option key={ty} value={ty}>
                        {ty}
                      </option>
                    ))}
                  </select>
                </td>
                <td className="py-1.5 pr-3">
                  <select
                    className="rounded border border-line bg-[var(--surface-1)] px-1.5 py-0.5 text-fg"
                    value={t.tag}
                    disabled={busy}
                    onChange={(e) => update(t.frame, { tag: e.target.value })}
                  >
                    {tags.map((tag) => (
                      <option
                        key={tag}
                        value={tag}
                        disabled={
                          tag !== t.tag && targets.some((o) => o.tag === tag)
                        }
                      >
                        {tag}
                      </option>
                    ))}
                  </select>
                </td>
                <td className="py-1.5 text-right whitespace-nowrap">
                  <button
                    className="btn px-2 py-0.5"
                    onClick={() => jump(t.frame)}
                  >
                    Jump to
                  </button>{" "}
                  <button
                    className="btn px-2 py-0.5"
                    disabled={busy}
                    onClick={() => setToCurrent(t.frame)}
                  >
                    Set to current frame
                  </button>{" "}
                  <button
                    className="btn px-2 py-0.5"
                    disabled={busy}
                    onClick={() =>
                      setConfirm({ kind: "target", frame: t.frame })
                    }
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <div className="border-t border-line-subtle pt-2">
        <button
          className="text-xs text-fg-muted hover:text-fg"
          onClick={() => setShowTags((v) => !v)}
        >
          {showTags ? "▾" : "▸"} Tags ({tags.length}, whole dataset)
        </button>
        {showTags && (
          <div className="mt-2 space-y-2">
            <ul className="flex flex-wrap gap-2">
              {tags.map((tag) => (
                <li
                  key={tag}
                  className="flex items-center gap-1.5 rounded-md border border-line px-2 py-0.5 text-xs text-fg"
                >
                  <span className="font-mono">{tag}</span>
                  <span className="text-fg-faint tabular">
                    {tagCounts.get(tag) ?? 0}
                  </span>
                  <button
                    className="text-fg-faint hover:text-fg"
                    title={`Remove tag ${tag}`}
                    disabled={busy}
                    onClick={() => void askRemoveTag(tag)}
                  >
                    ×
                  </button>
                </li>
              ))}
              {tags.length === 0 && (
                <li className="text-xs text-fg-muted">No tags yet.</li>
              )}
            </ul>
            <form
              className="flex gap-2"
              onSubmit={async (e) => {
                e.preventDefault();
                const tag = newTag.trim();
                if (tag && (await postTag({ add: tag }))) setNewTag("");
              }}
            >
              <input
                className="w-48 rounded border border-line bg-[var(--surface-1)] px-2 py-1 text-xs text-fg"
                placeholder="new_tag"
                value={newTag}
                onChange={(e) => setNewTag(e.target.value)}
              />
              <button
                className="btn px-3 py-1"
                type="submit"
                disabled={busy || !newTag.trim()}
              >
                Add tag
              </button>
            </form>
          </div>
        )}
      </div>

      {confirm && (
        <ConfirmDialog onClose={() => setConfirm(null)}>
          {confirm.kind === "target" ? (
            <>
              <h4 className="text-sm font-semibold text-fg">
                Delete the target at frame {confirm.frame}?
              </h4>
              <div className="flex justify-end gap-2">
                <button
                  className="btn px-3 py-1.5"
                  onClick={() => setConfirm(null)}
                  autoFocus
                >
                  Cancel
                </button>
                <button
                  className={DANGER}
                  onClick={() => {
                    setConfirm(null);
                    void save(targets.filter((t) => t.frame !== confirm.frame));
                  }}
                >
                  Delete target
                </button>
              </div>
            </>
          ) : (
            <>
              <h4 className="text-sm font-semibold text-fg">
                Remove tag <span className="font-mono">{confirm.tag}</span>?
              </h4>
              <p className="text-xs text-fg-muted">
                {confirm.targets > 0
                  ? `This also deletes ${confirm.targets} target${confirm.targets === 1 ? "" : "s"} in ${confirm.episodes} episode${confirm.episodes === 1 ? "" : "s"} that use it.`
                  : "No target uses it."}
              </p>
              <div className="flex justify-end gap-2">
                <button
                  className="btn px-3 py-1.5"
                  onClick={() => setConfirm(null)}
                  autoFocus
                >
                  Cancel
                </button>
                <button
                  className={DANGER}
                  onClick={() => {
                    setConfirm(null);
                    void postTag({ remove: confirm.tag });
                  }}
                >
                  Remove tag and targets
                </button>
              </div>
            </>
          )}
        </ConfirmDialog>
      )}
    </div>
  );
}

const DANGER =
  "inline-flex items-center justify-center rounded-md border border-red-700 bg-red-600 px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-red-700";

function ConfirmDialog({
  onClose,
  children,
}: {
  onClose: () => void;
  children: React.ReactNode;
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
        className="w-full max-w-md space-y-3 rounded-lg border border-line bg-[var(--surface-1)] p-5"
        onClick={(e) => e.stopPropagation()}
      >
        {children}
      </div>
    </div>
  );
}
