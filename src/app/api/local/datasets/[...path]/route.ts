import { NextRequest } from "next/server";
import { createReadStream } from "node:fs";
import { readdir, stat } from "node:fs/promises";
import path from "node:path";
import { Readable } from "node:stream";

// Read-only stand-in for huggingface.co/datasets that serves LeRobot datasets
// from local disk. Point the app at it with
//   LOCAL_LEROBOT_DATA_ROOT=/path/to/root DATASET_URL=/api/local/datasets
// A request for `<org>/<dataset>/resolve/<rev>/<file>` is answered from
// `$LOCAL_LEROBOT_DATA_ROOT/datasets/<org>/<dataset>/<file>`; the revision is
// ignored.
// `_/_/list` is a small extension used by the home page: it returns the
// `<org>/<dataset>` ids of every dataset found under the root.

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const CONTENT_TYPES: Record<string, string> = {
  ".json": "application/json",
  ".jsonl": "application/x-ndjson",
  ".parquet": "application/octet-stream",
  ".mp4": "video/mp4",
  ".webm": "video/webm",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".yaml": "text/yaml",
};

function rootDir(): string | null {
  const root = process.env.LOCAL_LEROBOT_DATA_ROOT;
  return root ? path.resolve(root, "datasets") : null;
}

// Maps the request path to a file below the root, or null if it escapes it.
function resolveLocalFile(root: string, parts: string[]): string | null {
  const [org, dataset, resolve, , ...rest] = parts;
  if (!org || !dataset || resolve !== "resolve" || rest.length === 0) {
    return null;
  }
  const file = path.resolve(root, org, dataset, ...rest);
  return file.startsWith(root + path.sep) ? file : null;
}

async function listDirs(dir: string): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true });
  return entries
    .filter((e) => e.isDirectory() && !e.name.startsWith("."))
    .map((e) => e.name)
    .sort();
}

async function serve(req: NextRequest, parts: string[], head: boolean) {
  const root = rootDir();
  if (!root)
    return new Response("LOCAL_LEROBOT_DATA_ROOT not set", { status: 500 });

  // Dataset listing for the home page.
  if (parts.join("/") === "_/_/list") {
    const ids: string[] = [];
    for (const org of await listDirs(root).catch(() => [])) {
      for (const ds of await listDirs(path.join(root, org)).catch(() => [])) {
        const info = path.join(root, org, ds, "meta", "info.json");
        if (await stat(info).catch(() => null)) ids.push(`${org}/${ds}`);
      }
    }
    return Response.json(ids);
  }

  const file = resolveLocalFile(root, parts);
  if (!file) return new Response("Forbidden", { status: 403 });

  let size: number;
  try {
    const st = await stat(file);
    if (!st.isFile()) throw new Error("not a file");
    size = st.size;
  } catch {
    return new Response("Not found", { status: 404 });
  }

  const headers = new Headers({
    "content-type":
      CONTENT_TYPES[path.extname(file).toLowerCase()] ??
      "application/octet-stream",
    "accept-ranges": "bytes",
    "cache-control": "no-store",
  });

  let start = 0;
  let end = size - 1;
  let status = 200;
  const range = req.headers.get("range");
  if (range) {
    const m = /^bytes=(\d*)-(\d*)$/.exec(range.trim());
    if (!m || (m[1] === "" && m[2] === "")) {
      return new Response(null, {
        status: 416,
        headers: { "content-range": `bytes */${size}` },
      });
    }
    if (m[1] === "") {
      // Suffix range: the last N bytes.
      start = Math.max(0, size - parseInt(m[2], 10));
    } else {
      start = parseInt(m[1], 10);
      if (m[2] !== "") end = Math.min(end, parseInt(m[2], 10));
    }
    if (start > end || start >= size) {
      return new Response(null, {
        status: 416,
        headers: { "content-range": `bytes */${size}` },
      });
    }
    status = 206;
    headers.set("content-range", `bytes ${start}-${end}/${size}`);
  }
  headers.set("content-length", String(end - start + 1));

  if (head || size === 0) return new Response(null, { status, headers });

  const stream = createReadStream(file, { start, end });
  req.signal.addEventListener("abort", () => stream.destroy());
  return new Response(
    Readable.toWeb(stream) as unknown as ReadableStream<Uint8Array>,
    { status, headers },
  );
}

export async function GET(
  req: NextRequest,
  ctx: { params: Promise<{ path: string[] }> },
) {
  const { path: parts } = await ctx.params;
  return serve(req, parts, false);
}

export async function HEAD(
  req: NextRequest,
  ctx: { params: Promise<{ path: string[] }> },
) {
  const { path: parts } = await ctx.params;
  return serve(req, parts, true);
}
