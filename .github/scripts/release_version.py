"""Print the project version after validating a stable GitHub release tag."""

import argparse
from pathlib import Path
import re
import tomllib


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    tag = parser.parse_args().tag

    number = r"(?:0|[1-9][0-9]*)"
    match = re.fullmatch(rf"v({number}\.{number}\.{number})", tag)
    if match is None:
        parser.error(f"Expected a stable SemVer tag (vX.Y.Z), received {tag!r}")

    version = match[1]
    with Path("pyproject.toml").open("rb") as package:
        project_version = tomllib.load(package)["project"]["version"]
    if version != project_version:
        parser.error(f"{tag} does not match pyproject.toml version {project_version}")

    print(version)
