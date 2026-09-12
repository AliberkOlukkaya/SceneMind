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
  await page.getByLabel("Describe a moment").fill("a blue screen");
  await page.getByRole("button", { name: "Search", exact: true }).click();
  await expect(
    page.locator(".semantic-search").getByRole("alert"),
  ).toContainText("Build a visual index or transcribe");
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
