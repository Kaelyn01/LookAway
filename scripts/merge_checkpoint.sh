#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_BASE_DIR="${PROJECT_ROOT}/cache/lookaway/checkpoints/LookAway-Qwen3.5-4B/global_step_65/"
BASE_DIR="${BASE_DIR:-${1:-${DEFAULT_BASE_DIR}}}"
BASE_DIR="${BASE_DIR%/}"
ACTOR_DIR="${BASE_DIR}/actor"

if [ ! -d "${ACTOR_DIR}" ]; then
  echo "Actor checkpoint directory not found: ${ACTOR_DIR}" >&2
  exit 1
fi

echo "Merging ${ACTOR_DIR} -> ${BASE_DIR}"

STAGING_DIR="$(mktemp -d "${BASE_DIR}.merge.XXXXXX")"
BACKUP_DIR="$(mktemp -d "${BASE_DIR}.backup.XXXXXX")"
restore_needed=0
staged_names=()
backed_up_names=()
installed_names=()

cleanup() {
  exit_code=$?
  trap - EXIT
  set +u
  if [ "${restore_needed}" -eq 1 ] && [ "${exit_code}" -ne 0 ]; then
    for name in "${installed_names[@]}"; do
      target="${BASE_DIR}/${name}"
      if [ -d "${target}" ] && [ ! -L "${target}" ]; then
        find "${target}" -depth -delete
      elif [ -e "${target}" ] || [ -L "${target}" ]; then
        rm -f "${target}"
      fi
    done
    for name in "${backed_up_names[@]}"; do
      mv "${BACKUP_DIR}/${name}" "${BASE_DIR}/"
    done
  fi
  find "${STAGING_DIR}" -depth -delete
  find "${BACKUP_DIR}" -depth -delete
  exit "${exit_code}"
}
trap cleanup EXIT

python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${ACTOR_DIR}" \
  --target_dir "${STAGING_DIR}"

while IFS= read -r -d '' staged_entry; do
  staged_name="${staged_entry##*/}"
  if [ "${staged_name}" = "actor" ]; then
    echo "Checkpoint merge produced a reserved actor directory." >&2
    exit 1
  fi
  staged_names+=("${staged_name}")
done < <(find "${STAGING_DIR}" -mindepth 1 -maxdepth 1 -print0)

if [ "${#staged_names[@]}" -eq 0 ]; then
  echo "Checkpoint merge produced no files." >&2
  exit 1
fi

# Replace merged entries only after staging succeeds. Preserve actor shards and
# restore prior top-level entries if installation fails midway.
restore_needed=1
while IFS= read -r -d '' existing_file; do
  name="${existing_file##*/}"
  mv "${existing_file}" "${BACKUP_DIR}/"
  backed_up_names+=("${name}")
done < <(find "${BASE_DIR}" -mindepth 1 -maxdepth 1 -type f -print0)
for name in "${staged_names[@]}"; do
  target="${BASE_DIR}/${name}"
  if [ -e "${target}" ] || [ -L "${target}" ]; then
    mv "${target}" "${BACKUP_DIR}/"
    backed_up_names+=("${name}")
  fi
  mv "${STAGING_DIR}/${name}" "${BASE_DIR}/"
  installed_names+=("${name}")
done
restore_needed=0

echo "Merge completed."
