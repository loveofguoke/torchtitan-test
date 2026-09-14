#!/usr/bin/env bash
set -euo pipefail

diagnostic_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "${diagnostic_dir}/../../.." && pwd)"
cd "${repository_root}"
exec python "${diagnostic_dir}/diagnostic_benchmark.py" "$@"
