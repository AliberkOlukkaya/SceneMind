"""Disposable Linux/PostgreSQL verification with no published ports or user volumes."""

import subprocess
import time
from uuid import uuid4


def run(*args, **kwargs):
    return subprocess.run(["docker", *args], check=True, **kwargs)


def main():
    name = "scenemind-check-" + uuid4().hex[:12]
    run("build", "-t", "scenemind-validation", ".")
    run(
        "run",
        "--rm",
        "-e",
        "SCENEMIND_DURABLE_JOBS=false",
        "scenemind-validation",
        "python",
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-c",
        "backend/pyproject.toml",
        "-q",
        "--basetemp=/app/data/tests",
    )
    run("network", "create", "--internal", name, capture_output=True)
    try:
        # Trust is confined to a disposable internal test network, with no host port.
        run(
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            "--network",
            name,
            "-e",
            "POSTGRES_HOST_AUTH_METHOD=trust",
            "-e",
            "POSTGRES_DB=scenemind",
            "postgres:17",
            capture_output=True,
        )
        for _ in range(60):
            ready = subprocess.run(
                ["docker", "exec", name, "pg_isready", "-U", "postgres"], capture_output=True
            )
            if ready.returncode == 0:
                break
            time.sleep(1)
        else:
            raise RuntimeError("PostgreSQL did not become ready")
        run(
            "run",
            "--rm",
            "--network",
            name,
            "-e",
            f"SCENEMIND_VALIDATION_DATABASE_URL=postgresql+psycopg://postgres@{name}/scenemind",
            "scenemind-validation",
            "python",
            "scripts/validate_postgres.py",
        )
    finally:
        subprocess.run(["docker", "stop", name], capture_output=True)
        run("network", "rm", name, capture_output=True)


if __name__ == "__main__":
    main()
