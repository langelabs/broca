#!/usr/bin/env bash
# Publish a validated Mesh image and print the verified registry digest.
set -euo pipefail

image=$1
revision=$2
version=${image##*:}

# Read the digest and OCI labels from a single-platform registry manifest.
inspect_mesh() {
  docker buildx imagetools inspect --format '{{json .}}' "$1"
}

if manifest=$(inspect_mesh "$image" 2>&1); then
  echo "Checking existing image $image for this release" >&2
elif [[ "$manifest" == "ERROR: $image: not found" ]]; then
  docker image tag broca-mesh:ci "$image"
  docker image push "$image" >&2
  manifest=$(inspect_mesh "$image")
else
  echo "$manifest" >&2
  exit 1
fi

if ! digest=$(jq -er --arg version "$version" --arg revision "$revision" '
  select(.image.config.Labels."org.opencontainers.image.version" == $version)
  | select(.image.config.Labels."org.opencontainers.image.revision" == $revision)
  | .manifest.digest | select(test("^sha256:[0-9a-f]{64}$"))
' <<< "$manifest"); then
  echo "Could not verify that $image belongs to version $version at revision $revision" >&2
  exit 1
fi

printf '%s\n' "$digest"
