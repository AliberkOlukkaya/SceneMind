# Current architecture

Next.js App Router with TypeScript and Tailwind provides the browser interface. FastAPI exposes GET /health. Pydantic Settings reads SCENEMIND_ environment variables. CORS allows localhost:3000 by default.

No database, model or retrieval pipeline exists yet. Local runtime data belongs in ignored data/. Backend code is in backend/app, tests in tests/. Later ML code will be added when a working feature needs it.
