"use client";

import React, { useCallback, useEffect, useState } from "react";
import { api } from "@/components/policy-panel";
import { Spinner } from "@/components/ui";

// Hugging Face uploads of trained policies (backend/hf_upload.py): the
// "upload when done" option of a new training and the upload of existing runs.

export type HfUpload = {
  id: string;
  run: string;
  repo_id: string | null;
  private: boolean;
  job_id: string | null;
  status: "running" | "done" | "failed";
  progress?: {
    phase: "preparing" | "uploading" | "committing";
    done?: number | null;
    total?: number | null;
    rate?: number | null;
  };
  url: string | null;
  error: string | null;
  started_at: number;
  finished_at: number | null;
};
type HfAccount = { user: string | null; error?: string; dry_run?: boolean };
export type UploadRequest = { repo_id: string; private: boolean };

const INPUT_CLASS =
  "w-full rounded-md border border-line bg-[var(--surface-0)] px-2 py-1.5 text-xs text-fg outline-none focus:border-accent disabled:opacity-50";
const REPO_ID_RE =
  /^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}(\/[A-Za-z0-9][A-Za-z0-9_.-]{0,95})?$/;

export function repoIdProblem(repoId: string): string | null {
  const id = repoId.trim();
  if (!id) return null;
  if (!REPO_ID_RE.test(id) || id.includes("..") || id.endsWith(".git"))
    return "Use user/name with letters, digits, '_', '-' and '.' only";
  return null;
}

/** The Hub account uploads go to (`hf auth login` on the backend machine). */
export function useHfAccount(enabled: boolean) {
  const [account, setAccount] = useState<HfAccount | null>(null);
  useEffect(() => {
    if (!enabled || account) return;
    api<HfAccount>("/api/policy/train/hf")
      .then(setAccount)
      .catch((e) => setAccount({ user: null, error: String(e.message ?? e) }));
  }, [enabled, account]);
  return account;
}

/** All uploads, polled while one is running. */
export function useUploads() {
  const [uploads, setUploads] = useState<HfUpload[]>([]);
  const refresh = useCallback(() => {
    api<HfUpload[]>("/api/policy/train/uploads")
      .then(setUploads)
      .catch(() => {});
  }, []);
  const running = uploads.some((u) => u.status === "running");
  useEffect(() => {
    refresh();
    const timer = setInterval(refresh, running ? 2000 : 10000);
    return () => clearInterval(timer);
  }, [refresh, running]);
  return { uploads, refresh };
}

function AccountLine({ account }: { account: HfAccount | null }) {
  if (!account)
    return (
      <span className="flex items-center gap-1">
        <Spinner /> Checking Hugging Face login…
      </span>
    );
  if (!account.user)
    return (
      <span className="text-flag-fg">
        {account.error ?? "Not logged in"}. Run <code>hf auth login</code> on
        the machine running the visualizer.
      </span>
    );
  return (
    <span>
      Logged in as <span className="font-mono">{account.user}</span>
      {account.dry_run && " (dry run: nothing is sent)"}
    </span>
  );
}

function RepoFields({
  value,
  onChange,
  defaultName,
  user,
}: {
  value: UploadRequest;
  onChange: (v: UploadRequest) => void;
  defaultName: string;
  user: string | null;
}) {
  const problem = repoIdProblem(value.repo_id);
  return (
    <div className="space-y-2">
      <label className="block space-y-1">
        <span className="block text-xs text-fg-muted">Model repository</span>
        <input
          className={`${INPUT_CLASS} font-mono`}
          value={value.repo_id}
          onChange={(e) => onChange({ ...value, repo_id: e.target.value })}
          placeholder={`${user ?? "<user>"}/${defaultName}`}
        />
      </label>
      {problem && <p className="text-xs text-flag-fg">{problem}</p>}
      <label className="flex items-center gap-2 text-xs text-fg-soft">
        <input
          type="checkbox"
          className="accent-accent"
          checked={value.private}
          onChange={(e) => onChange({ ...value, private: e.target.checked })}
        />
        Private repository
      </label>
    </div>
  );
}

/** "Upload to Hugging Face when done" in the New training form. */
export function UploadWhenDone({
  enabled,
  onEnabled,
  value,
  onChange,
  defaultName,
}: {
  enabled: boolean;
  onEnabled: (v: boolean) => void;
  value: UploadRequest;
  onChange: (v: UploadRequest) => void;
  defaultName: string;
}) {
  const account = useHfAccount(enabled);
  return (
    <div className="space-y-2">
      <label className="flex items-center gap-2 text-xs text-fg-soft">
        <input
          type="checkbox"
          className="accent-accent"
          checked={enabled}
          onChange={(e) => onEnabled(e.target.checked)}
        />
        Upload to Hugging Face when done
      </label>
      {enabled && (
        <div className="space-y-2 border-l-2 border-line pl-3">
          <RepoFields
            value={value}
            onChange={onChange}
            defaultName={defaultName}
            user={account?.user ?? null}
          />
          <p className="text-[11px] text-fg-faint">
            <AccountLine account={account} />. After a successful training the
            last checkpoint (weights, configs, processors) and a model card are
            uploaded. The dataset is not uploaded.
          </p>
        </div>
      )}
    </div>
  );
}

/** Upload state of a run (runs table) or of a finished training (queue). */
export function UploadStatus({
  upload,
  onDismiss,
}: {
  upload: HfUpload | undefined;
  onDismiss?: () => void;
}) {
  if (!upload) return null;
  if (upload.status === "running") return <UploadProgress upload={upload} />;
  if (upload.status === "done")
    return (
      <span className="inline-flex items-center gap-1.5">
        <a
          href={upload.url ?? undefined}
          target="_blank"
          rel="noreferrer"
          className="font-mono text-accent-fg hover:underline"
        >
          {upload.repo_id} ↗
        </a>
        {upload.repo_id && <CopyButton text={upload.repo_id} />}
      </span>
    );
  return (
    <span
      className="block max-w-xs truncate text-red-500"
      title={upload.error ?? undefined}
    >
      upload failed: {upload.error ?? "unknown error"}{" "}
      {onDismiss && (
        <button className="text-fg-faint hover:text-fg" onClick={onDismiss}>
          (dismiss)
        </button>
      )}
    </span>
  );
}

function formatBytes(n: number) {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GB`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  return `${Math.round(n / 1e3)} kB`;
}

function UploadProgress({ upload }: { upload: HfUpload }) {
  const p = upload.progress;
  const pct =
    p?.total && p.done != null ? Math.min(100, (p.done / p.total) * 100) : null;
  const label =
    p?.phase === "uploading"
      ? pct != null
        ? `uploading ${pct.toFixed(0)}%`
        : "uploading…"
      : p?.phase === "committing"
        ? "committing…"
        : "preparing…";
  return (
    <span className="inline-flex min-w-36 flex-col gap-1 text-accent-fg">
      <span className="flex items-center gap-1">
        <Spinner className="h-3 w-3" /> {label}
      </span>
      {pct != null && (
        <>
          <span className="block h-1 w-full overflow-hidden rounded bg-fill">
            <span
              className="block h-full bg-accent transition-all"
              style={{ width: `${pct}%` }}
            />
          </span>
          <span className="tabular text-[11px] text-fg-muted">
            {formatBytes(p!.done!)} / {formatBytes(p!.total!)}
            {p?.rate ? ` · ${formatBytes(p.rate)}/s` : ""}
          </span>
        </>
      )}
    </span>
  );
}

/** Copies the model repository id (e.g. for `--policy.path`). */
function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // Fallback for non-secure contexts (http on another host).
      const area = document.createElement("textarea");
      area.value = text;
      document.body.appendChild(area);
      area.select();
      document.execCommand("copy");
      area.remove();
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };
  return (
    <button
      className="btn px-1.5 py-0.5 text-[11px]"
      onClick={copy}
      title={`Copy ${text}`}
    >
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

export function dismissUpload(id: string) {
  return api(`/api/policy/train/uploads/${id}`, { method: "DELETE" });
}

/** Confirmation dialog for uploading an existing run. */
export function UploadDialog({
  run,
  onClose,
  onStarted,
}: {
  run: string;
  onClose: () => void;
  onStarted: () => void;
}) {
  const account = useHfAccount(true);
  const [value, setValue] = useState<UploadRequest>({
    repo_id: "",
    private: true,
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const start = async () => {
    setBusy(true);
    setError(null);
    try {
      await api("/api/policy/train/uploads", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ run, ...value, repo_id: value.repo_id.trim() }),
      });
      onStarted();
      onClose();
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setBusy(false);
    }
  };

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
        <h4 className="text-sm font-semibold text-fg">
          Upload to Hugging Face
        </h4>
        <p className="text-xs text-fg-muted">
          Uploads the last checkpoint of{" "}
          <span className="font-mono">{run}</span> (weights, configs,
          processors) and a model card. The dataset is not uploaded. An existing
          repository gets a new commit.
        </p>
        <RepoFields
          value={value}
          onChange={setValue}
          defaultName={run}
          user={account?.user ?? null}
        />
        <p className="text-[11px] text-fg-faint">
          <AccountLine account={account} />
        </p>
        {error && <p className="text-xs text-flag-fg">{error}</p>}
        <div className="flex flex-wrap justify-end gap-2">
          <button className="btn px-3 py-1.5" onClick={onClose} autoFocus>
            Cancel
          </button>
          <button
            className="btn btn-primary px-3 py-1.5"
            disabled={busy || !account?.user || !!repoIdProblem(value.repo_id)}
            onClick={start}
          >
            {busy ? <Spinner /> : null}
            Upload
          </button>
        </div>
      </div>
    </div>
  );
}
