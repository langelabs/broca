"""Check release version validation and safe image publishing retries."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / ".github" / "scripts"
IMAGE = "docker.io/lange-labs/mesh:0.1.0"
REVISION = "a" * 40
DIGEST = "sha256:" + "b" * 64


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("v0.1.0", "0.1.0"),
        ("0.1.0", "Expected a stable SemVer tag"),
        ("v00.1.0", "Expected a stable SemVer tag"),
        ("v0.1.0-rc.1", "Expected a stable SemVer tag"),
        ("v0.1.0+build", "Expected a stable SemVer tag"),
        ("--help", "Expected a stable SemVer tag"),
        ("v0.1.1", "does not match pyproject.toml"),
    ],
)
def test_release_version(tmp_path: Path, tag: str, expected: str) -> None:
    """Require a stable GitHub release tag that matches the project version.

    :param tmp_path: Isolated project directory.
    :param tag: Tag passed to the release validator.
    :param expected: Expected version or validation error.
    """
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "0.1.0"\n')
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "release_version.py"), "--", tag],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    if tag == "v0.1.0":
        assert result.stdout.strip() == expected
        assert result.returncode == 0
    else:
        assert expected in result.stderr
        assert result.returncode != 0


@pytest.mark.parametrize(
    ("registry_state", "should_push", "success"),
    [
        ("missing", True, True),
        ("matching", False, True),
        ("wrong-version", False, False),
        ("wrong-revision", False, False),
        ("unavailable", False, False),
    ],
)
def test_image_publish_retry(
    tmp_path: Path, registry_state: str, should_push: bool, success: bool
) -> None:
    """Publish a missing image tag and preserve existing tags on retries.

    :param tmp_path: Isolated directory for the Docker command and registry state.
    :param registry_state: Docker registry response to the first inspection.
    :param should_push: Whether this run may upload an image.
    :param success: Whether the publish can return a verified digest.
    """
    published = tmp_path / "pushed"
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        'case "$1 $2" in\n'
        '  "image tag") exit 0 ;;\n'
        '  "image push") touch "$PUBLISHED"; exit 0 ;;\n'
        '  "buildx imagetools")\n'
        '    if [[ "$5" != "{{json .}}" ]]; then\n'
        '      printf "Name: %s\\nDigest: %s\\n" "${@: -1}" "$DIGEST"; exit 0\n'
        '    fi\n'
        '    if [[ "$STATE" == missing && ! -e "$PUBLISHED" ]]; then\n'
        '      printf "ERROR: %s: not found\\n" "${@: -1}" >&2; exit 1\n'
        '    fi\n'
        '    if [[ "$STATE" == unavailable ]]; then\n'
        '      echo "ERROR: registry connection failed" >&2; exit 1\n'
        '    fi\n'
        '    printf "%s\\n" "$INSPECTION" ;;\n'
        '  *) exit 64 ;;\n'
        "esac\n"
    )
    docker.chmod(0o755)
    labels = {
        "org.opencontainers.image.version": (
            "0.1.1" if registry_state == "wrong-version" else "0.1.0"
        ),
        "org.opencontainers.image.revision": (
            "c" * 40 if registry_state == "wrong-revision" else REVISION
        ),
    }
    result = subprocess.run(
        ["bash", str(SCRIPTS / "push_mesh.sh"), IMAGE, REVISION],
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "STATE": registry_state,
            "PUBLISHED": str(published),
            "DIGEST": DIGEST,
            "INSPECTION": json.dumps(
                {
                    "manifest": {"digest": DIGEST},
                    "image": {"config": {"Labels": labels}},
                },
                indent=2,
            ),
        },
        capture_output=True,
        text=True,
    )

    assert published.exists() == should_push
    assert (result.returncode == 0) == success
    assert result.stdout.strip() == (DIGEST if success else "")
