import { execFileSync } from "node:child_process";
import path from "node:path";

export default function setup() {
  const python = path.resolve(
    process.platform === "win32"
      ? "../.venv/Scripts/python.exe"
      : "../.venv/bin/python",
  );
  execFileSync(python, [path.resolve("../scripts/create_fixture.py")], {
    stdio: "inherit",
  });
}
