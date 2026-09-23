import { expect, test } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

test("connection warning clears when the backend recovers", async ({
  page,
}) => {
  await page.route("http://127.0.0.1:8010/videos", (route) => route.abort());
  await page.goto("/");
  await expect(
    page.getByText("Cannot reach the video service.", { exact: false }),
  ).toBeVisible();
  await page.unroute("http://127.0.0.1:8010/videos");
  await expect(
    page.getByText("Cannot reach the video service.", { exact: false }),
  ).not.toBeVisible({ timeout: 10_000 });
});

test("upload, browse frames and seek on desktop and mobile", async ({
  page,
}) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Video library" }),
  ).toBeVisible();
  await expect(
    page.getByText("Up to 1024 MiB", { exact: false }),
  ).toBeVisible();
  await expect(page.getByText("60 minutes", { exact: false })).toBeVisible();
  await page
    .getByLabel("Choose video", { exact: true })
    .setInputFiles(path.resolve("../data/e2e-fixture.mp4"));
  await expect(
    page.getByRole("heading", { name: "e2e-fixture.mp4" }),
  ).toBeVisible();
  await expect(page.getByLabel("Seek to 0:05")).toBeVisible({
    timeout: 30_000,
  });
  await page.getByLabel("Seek to 0:05").click();
  await expect
    .poll(() =>
      page
        .locator("video")
        .evaluate((video: HTMLVideoElement) => video.currentTime),
    )
    .toBeCloseTo(5, 0);
  await page.getByLabel("Search mode").selectOption("visual");
  await page.getByLabel("Describe a moment").fill("a blue screen");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  await expect(
    page.locator(".semantic-search").getByRole("alert"),
  ).toContainText("Build the visual index before searching.");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
  await page.screenshot({
    path: `../data/workspace-${test.info().project.name}.png`,
    fullPage: true,
  });
});

test("search presents possible moments without false certainty", async ({
  page,
}) => {
  await page.goto("/");
  await page
    .getByLabel("Choose video", { exact: true })
    .setInputFiles(path.resolve("../data/e2e-fixture.mp4"));
  await expect(page.getByLabel("Seek to 0:05")).toBeVisible({
    timeout: 30_000,
  });
  await expect(page.getByLabel("Search mode")).toHaveValue("hybrid");
  await expect(page.getByLabel("Search mode")).not.toHaveValue("auto");
  await expect(
    page.getByText("Search across both spoken and visual content."),
  ).toBeVisible();
  await page.route("**/videos/*/search?*", async (route) => {
    expect(new URL(route.request().url()).searchParams.get("mode")).toBe(
      "hybrid",
    );
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        score_type: "reciprocal_rank_fusion",
        requested_mode: "hybrid",
        selected_route: "hybrid",
        modalities_used: ["speech", "visual"],
        results: [
          {
            timestamp: 5,
            thumbnail: "/fixture/frame.jpg",
            score: 8.74219,
            modality: "speech",
            text: "The speaker explains the project architecture.",
          },
        ],
      }),
    });
  });
  await page.route("**/fixture/frame.jpg", (route) =>
    route.fulfill({ status: 200, contentType: "image/jpeg", body: "" }),
  );
  await page.getByLabel("Describe a moment").fill("project architecture");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Most relevant moments" }),
  ).toBeVisible();
  await expect(page.getByText("1 possible match")).toBeVisible();
  await expect(
    page.getByText("Results are ranked by relevance", { exact: false }),
  ).toBeVisible();
  await expect(page.locator(".result-context")).toContainText("Smart Search");
  await expect(
    page.getByText("The speaker explains the project architecture."),
  ).toBeVisible();
  await expect(page.getByText("8.742", { exact: false })).not.toBeVisible();
  await page.locator(".result").click();
  await expect
    .poll(() =>
      page
        .locator("video")
        .evaluate((video: HTMLVideoElement) => video.currentTime),
    )
    .toBeCloseTo(5, 0);
});

test("product search modes map directly to existing retrieval modes", async ({
  page,
}) => {
  await page.goto("/");
  await page
    .getByLabel("Choose video", { exact: true })
    .setInputFiles(path.resolve("../data/e2e-fixture.mp4"));
  await expect(page.getByLabel("Seek to 0:05")).toBeVisible({
    timeout: 30_000,
  });
  const selector = page.getByLabel("Search mode");
  await expect(selector.locator("option")).toHaveText([
    "Smart Search",
    "Spoken Content",
    "Visual Content",
  ]);
  await selector.selectOption({ label: "Spoken Content" });
  await expect(selector).toHaveValue("speech");
  await expect(
    page.getByText("Search what is said in the video."),
  ).toBeVisible();
  await selector.selectOption({ label: "Visual Content" });
  await expect(selector).toHaveValue("visual");
  await expect(
    page.getByText("Search what appears in the video."),
  ).toBeVisible();
  await expect(selector.locator('option[value="auto"]')).toHaveCount(0);
});

test("URL import joins the normal library, search, and seek flow", async ({
  page,
}) => {
  const id = "00000000-0000-0000-0000-000000000099";
  let imported = false;
  let polls = 0;
  const queued = {
    id,
    filename: "Importing video",
    status: "processing",
    stage: "fetching",
    source_type: "url",
    source_provider: "direct",
    source_url: "https://media.example/test.mp4",
    frames: [],
  };
  const ready = {
    ...queued,
    filename: "Public test video.mp4",
    status: "ready",
    stage: "ready",
    metadata: { duration: 6, width: 320, height: 240, fps: 10 },
    frames: [
      { timestamp: 0, thumbnail: `/videos/${id}/frames/000001.jpg` },
      { timestamp: 5, thumbnail: `/videos/${id}/frames/000002.jpg` },
    ],
  };
  await page.route("http://127.0.0.1:8010/videos", async (route) => {
    if (route.request().method() !== "GET") return route.fallback();
    polls += 1;
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify(imported ? (polls > 1 ? [ready] : [queued]) : []),
    });
  });
  await page.route("**/videos/import-url", async (route) => {
    expect(route.request().postDataJSON()).toEqual({
      url: "https://media.example/test.mp4",
    });
    imported = true;
    polls = 0;
    await route.fulfill({
      status: 202,
      contentType: "application/json",
      body: JSON.stringify(queued),
    });
  });
  await page.route(`**/videos/${id}/index`, (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({ status: "ready" }),
    }),
  );
  await page.route(`**/videos/${id}/transcript?*`, (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({ status: "ready", segments: [] }),
    }),
  );
  const media = fs.readFileSync(path.resolve("../data/e2e-fixture.mp4"));
  await page.route(`**/videos/${id}/media`, (route) => {
    const range = route.request().headers().range;
    if (!range)
      return route.fulfill({
        status: 200,
        contentType: "video/mp4",
        headers: {
          "Content-Length": String(media.length),
          "Accept-Ranges": "bytes",
        },
        body: media,
      });
    const match = /bytes=(\d+)-(\d*)/.exec(range);
    const start = Number(match?.[1] || 0);
    const end = Math.min(
      Number(match?.[2] || media.length - 1),
      media.length - 1,
    );
    return route.fulfill({
      status: 206,
      contentType: "video/mp4",
      headers: {
        "Accept-Ranges": "bytes",
        "Content-Range": `bytes ${start}-${end}/${media.length}`,
        "Content-Length": String(end - start + 1),
      },
      body: media.subarray(start, end + 1),
    });
  });
  await page.route(`**/videos/${id}/frames/*.jpg`, (route) =>
    route.fulfill({ status: 200, contentType: "image/jpeg", body: "" }),
  );
  await page.route(`**/videos/${id}/search?*`, (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        selected_route: "hybrid",
        modalities_used: ["visual", "speech"],
        results: [
          {
            timestamp: 5,
            thumbnail: `/videos/${id}/frames/000002.jpg`,
            score: 0.1,
            modality: "visual+speech",
          },
        ],
      }),
    }),
  );

  await page.goto("/");
  await page.getByLabel("Video URL").fill("https://media.example/test.mp4");
  await page.getByRole("button", { name: "Import video" }).click();
  await expect(page.getByRole("status")).toContainText("Fetching video");
  await expect(page.getByLabel("Seek to 0:05")).toBeVisible({
    timeout: 10_000,
  });
  await expect(page.getByRole("link", { name: "Open source" })).toHaveAttribute(
    "href",
    "https://media.example/test.mp4",
  );
  await page.getByLabel("Describe a moment").fill("public test moment");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  await expect
    .poll(() =>
      page
        .locator("video")
        .evaluate((video: HTMLVideoElement) => video.duration),
    )
    .toBeGreaterThan(0);
  await page.locator(".result").click();
  await expect
    .poll(() =>
      page
        .locator("video")
        .evaluate((video: HTMLVideoElement) => video.currentTime),
    )
    .toBeCloseTo(5, 0);
});

test("Ask Video answers with timestamp evidence and supports abstention", async ({
  page,
}) => {
  await page.route("**/videos/*/ask/status", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({ enabled: true, configured: true }),
    }),
  );
  let abstain = false;
  await page.route("**/videos/*/ask", async (route) => {
    const question = route.request().postDataJSON().question;
    abstain = question.includes("World Cup");
    await route.fulfill({
      contentType: "application/json",
      body: JSON.stringify(
        abstain
          ? {
              answerable: false,
              answer:
                "I couldn't find enough evidence in this video to answer that reliably.",
              citations: [],
            }
          : {
              answerable: true,
              answer:
                "The speaker recommends retrieval for factual grounding [1].",
              citations: [
                {
                  evidence_id: "E001",
                  start_seconds: 5,
                  end_seconds: 6,
                  text: "Retrieval improves factual grounding.",
                },
              ],
            },
      ),
    });
  });
  await page.goto("/");
  await page
    .getByLabel("Choose video", { exact: true })
    .setInputFiles(path.resolve("../data/e2e-fixture.mp4"));
  await expect(page.getByLabel("Seek to 0:05")).toBeVisible({
    timeout: 30_000,
  });
  const question = page.getByLabel("Ask something about this video");
  await question.fill("Why use retrieval?");
  await page.getByRole("button", { name: "Ask", exact: true }).click();
  await expect(
    page.getByText("The speaker recommends retrieval", { exact: false }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Source 1, seek to 0:05" }).click();
  await expect
    .poll(() =>
      page
        .locator("video")
        .evaluate((video: HTMLVideoElement) => video.currentTime),
    )
    .toBeCloseTo(5, 0);
  await question.fill("Who won the World Cup?");
  await page.getByRole("button", { name: "Ask", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Not enough evidence" }),
  ).toBeVisible();
  expect(abstain).toBeTruthy();
});

test("Ask Video explains missing provider configuration", async ({ page }) => {
  await page.route("**/videos/*/ask/status", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({ enabled: true, configured: false }),
    }),
  );
  await page.goto("/");
  await page
    .getByLabel("Choose video", { exact: true })
    .setInputFiles(path.resolve("../data/e2e-fixture.mp4"));
  await expect(
    page.getByText("Ask Video is not configured", { exact: false }),
  ).toBeVisible({
    timeout: 30_000,
  });
});

test("Ask Video explains bounded scope without exposing internal labels", async ({
  page,
}) => {
  await page.route("**/videos/*/ask/status", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({ enabled: true, configured: true }),
    }),
  );
  await page.route("**/videos/*/ask", (route) =>
    route.fulfill({
      contentType: "application/json",
      body: JSON.stringify({
        answerable: false,
        answer:
          "I can't answer that type of question reliably yet. Try asking about a specific fact or explanation spoken in the video.",
        citations: [],
        scope: { supported: false, category: "TEMPORAL_ORDERING" },
      }),
    }),
  );
  await page.goto("/");
  await page
    .getByLabel("Choose video", { exact: true })
    .setInputFiles(path.resolve("../data/e2e-fixture.mp4"));
  await expect(page.getByLabel("Ask something about this video")).toBeVisible({
    timeout: 30_000,
  });
  await page
    .getByLabel("Ask something about this video")
    .fill("What happens after the introduction?");
  await page.getByRole("button", { name: "Ask", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Question not supported yet" }),
  ).toBeVisible();
  await expect(page.getByText("TEMPORAL_ORDERING")).toHaveCount(0);
});

test("real model search seeks to the matching scene", async ({ page }) => {
  test.skip(
    !process.env.SCENEMIND_MODEL_E2E,
    "Opt-in: requires visual extra and downloads CLIP if uncached.",
  );
  await page.goto("/");
  await page
    .getByLabel("Choose video", { exact: true })
    .setInputFiles(path.resolve("../data/e2e-fixture.mp4"));
  await expect(page.getByLabel("Seek to 0:05")).toBeVisible({
    timeout: 30_000,
  });
  await page.getByRole("button", { name: "Build visual index" }).click();
  await expect(page.getByText("Visual index: ready")).toBeVisible({
    timeout: 90_000,
  });
  await page.getByLabel("Describe a moment").fill("a blue screen");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  const first = page.locator(".result").first();
  await expect(first).toContainText("0:05");
  await first.click();
  await expect
    .poll(() =>
      page
        .locator("video")
        .evaluate((video: HTMLVideoElement) => video.currentTime),
    )
    .toBeCloseTo(5, 0);
  await page.screenshot({
    path: `../data/search-${test.info().project.name}.png`,
    fullPage: true,
  });
});
