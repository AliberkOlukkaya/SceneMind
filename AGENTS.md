# SceneMind engineering instructions

Build a local-first multimodal video search product. Read PROJECT_STATUS.md and TASKS.md before each phase. Verify prior acceptance criteria, implement a bounded phase, test, document, and commit stable milestones. Continue safe work autonomously.

No mandatory paid APIs, cloud resources or training. No facial identity recognition. Never commit secrets, user media, weights, datasets, embeddings or local databases. Keep models configurable and cached; unit tests must mock inference. Explain major ML components in docs/learning with inputs, outputs, preprocessing, parameters, source, license, CPU/GPU behavior and alternatives.

Run backend pytest and Ruff, frontend ESLint, TypeScript and production build for relevant changes. Never report unexecuted checks as passing. Update PROJECT_STATUS.md, TASKS.md, ARCHITECTURE.md and DECISIONS.md as behavior evolves. Record exact blockers and next actions. Do not force push or rewrite history.
