import { defineConfig, devices } from "@playwright/test";
import path from "node:path";

const runId = Date.now().toString();
export default defineConfig({
  testDir: "./e2e",
  globalSetup: "./e2e/setup.ts",
  timeout: 120_000,
  workers: 1,
  use: { baseURL: "http://127.0.0.1:3010", trace: "retain-on-failure" },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"] } },
    {
      name: "mobile",
      use: { ...devices["iPhone 13"], defaultBrowserType: "chromium" },
    },
  ],
  webServer: [
    {
      command: `"${path.resolve(process.platform === "win32" ? "../.venv/Scripts/python.exe" : "../.venv/bin/python")}" -m uvicorn app.main:app --host 127.0.0.1 --port 8010`,
      url: "http://127.0.0.1:8010/health",
      env: {
        SCENEMIND_DATA_DIR: `../data/e2e-${runId}/videos`,
        SCENEMIND_DATABASE_URL: `sqlite:///../data/e2e-${runId}/transcripts.db`,
        SCENEMIND_MODEL_CACHE: "../data/models",
        SCENEMIND_CORS_ORIGINS: '["http://127.0.0.1:3010"]',
      },
    },
    {
      command: "npm run dev -- --hostname 127.0.0.1 --port 3010",
      url: "http://127.0.0.1:3010",
      env: { NEXT_PUBLIC_API_URL: "http://127.0.0.1:8010" },
      timeout: 120_000,
    },
  ],
});
