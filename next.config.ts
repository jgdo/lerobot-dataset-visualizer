import type { NextConfig } from "next";
import packageJson from "./package.json";

const nextConfig: NextConfig = {
  typescript: {
    ignoreBuildErrors: true,
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  transpilePackages: ["three"],
  // Data fetching runs in the browser, so DATASET_URL has to be inlined into
  // the client bundle for a local dataset server to take effect there.
  env: {
    DATASET_URL: process.env.DATASET_URL ?? "",
    POLICY_BACKEND_ENABLED: process.env.POLICY_BACKEND_URL ? "1" : "",
  },
  // The policy evaluation service (backend/policy_server.py) is reached
  // same-origin through this rewrite, so the browser needs no CORS setup.
  async rewrites() {
    const policyBackend = process.env.POLICY_BACKEND_URL;
    return policyBackend
      ? [
          {
            source: "/api/policy/:path*",
            destination: `${policyBackend.replace(/\/$/, "")}/api/policy/:path*`,
          },
        ]
      : [];
  },
  // Avoid the 200-800ms cold-start cost of barrel-file imports.
  // react-icons re-exports thousands of icon components from /fa, etc.;
  // recharts and @huggingface/hub also have wide entry surfaces.
  experimental: {
    optimizePackageImports: ["react-icons", "recharts", "@huggingface/hub"],
  },
  generateBuildId: () => packageJson.version,
};

export default nextConfig;
