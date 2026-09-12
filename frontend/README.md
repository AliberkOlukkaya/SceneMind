# SceneMind frontend

Next.js App Router, TypeScript and Tailwind. See the [root README](../README.md) for setup, models and limitations.

Commands: npm run dev, lint, typecheck, build, format, test:e2e.

Browser tests start isolated API/frontend servers and generate synthetic media. Install backend development dependencies and run `npx playwright install chromium`. Set SCENEMIND_MODEL_E2E=1 for real CLIP tests; those download missing model weights.
