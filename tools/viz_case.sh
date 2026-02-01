#!/usr/bin/env bash
set -euo pipefail

case_dir="${1:-}"
if [[ -z "${case_dir}" ]]; then
  echo "Usage: $0 <synthetic_case_dir_name> [topk]" >&2
  echo "Example: $0 rotation_fine_22_5deg 1" >&2
  exit 2
fi

topk="${2:-1}"

python tools/viz_detect.py \
  --image "synthetic_cases/${case_dir}/image.png" \
  --template "synthetic_cases/${case_dir}/template.png" \
  --match-json "synthetic_cases/${case_dir}/cli_config.json" \
  --topk "${topk}" \
  --save-overlay "${case_dir}.png"

