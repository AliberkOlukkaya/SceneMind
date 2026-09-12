"""Record installed dependency closures, excluding unrelated local packages.

This is a platform-specific snapshot, not a dependency resolver. Run pip check first.
"""

import tomllib
from importlib.metadata import distribution
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

root = Path(__file__).resolve().parents[1]
project = tomllib.loads((root / "backend/pyproject.toml").read_text(encoding="utf-8"))["project"]


def closure(extras):
    pending = [*project["dependencies"]]
    for extra in extras:
        pending.extend(project["optional-dependencies"][extra])
    found = {}
    while pending:
        requirement = Requirement(pending.pop())
        if requirement.marker and not requirement.marker.evaluate():
            continue
        name = canonicalize_name(requirement.name)
        if name in found:
            continue
        installed = distribution(name)
        found[name] = installed.version
        for dependency in installed.requires or []:
            parsed = Requirement(dependency)
            if parsed.marker is None or any(
                parsed.marker.evaluate({"extra": extra}) for extra in (requirement.extras or {""})
            ):
                parsed.marker = None
                pending.append(str(parsed))
    return "\n".join(f"{name}=={version}" for name, version in sorted(found.items())) + "\n"


(root / "backend/requirements.lock").write_text(
    "# Windows Python 3.13 base + development dependency snapshot\n" + closure(["dev"]),
    encoding="utf-8",
)
(root / "backend/requirements-ml.lock").write_text(
    "# Windows Python 3.13 full CPU inference environment\n"
    "--extra-index-url https://download.pytorch.org/whl/cpu\n"
    + closure(["dev", "speech", "visual"]),
    encoding="utf-8",
)
