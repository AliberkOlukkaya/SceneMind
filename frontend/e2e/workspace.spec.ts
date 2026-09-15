import { expect, test } from "@playwright/test";
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
  await expect(
    page.locator(".result-context"),
  ).toContainText("Smart Search");
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
  await expect(page.getByText("Search what is said in the video.")).toBeVisible();
  await selector.selectOption({ label: "Visual Content" });
  await expect(selector).toHaveValue("visual");
  await expect(page.getByText("Search what appears in the video.")).toBeVisible();
  await expect(selector.locator('option[value="auto"]')).toHaveCount(0);
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
