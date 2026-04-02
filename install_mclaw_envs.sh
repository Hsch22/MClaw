#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_DIR}}"
ENV_BASE="${ENV_BASE:-/mnt/kangshijia/wangbinyu/conda_envs}"
PTH_NAME="${PTH_NAME:-mclaw-local-dev.pth}"
ENV_NAMES=("mclaw-train" "mclaw-envserver")

usage() {
    cat <<EOF
Usage: bash install_mclaw_envs.sh

Optional environment variables:
  PROJECT_ROOT  Override the project path written into .pth
  ENV_BASE      Override the conda env base directory
  PTH_NAME      Override the .pth file name
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ ! -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
    echo "[error] pyproject.toml not found under PROJECT_ROOT=${PROJECT_ROOT}" >&2
    exit 1
fi

for env_name in "${ENV_NAMES[@]}"; do
    env_dir="${ENV_BASE}/${env_name}"
    python_bin="${env_dir}/bin/python"

    if [[ ! -x "${python_bin}" ]]; then
        echo "[error] Missing python executable: ${python_bin}" >&2
        exit 1
    fi

    site_packages="$("${python_bin}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
    pth_path="${site_packages}/${PTH_NAME}"

    printf '%s\n' "${PROJECT_ROOT}" > "${pth_path}"

    imported_path="$("${python_bin}" -c 'import mclaw; print(mclaw.__file__)')"
    if [[ "${imported_path}" != "${PROJECT_ROOT}/mclaw/__init__.py" ]]; then
        echo "[error] ${env_name} resolved mclaw to ${imported_path}, expected ${PROJECT_ROOT}/mclaw/__init__.py" >&2
        exit 1
    fi

    echo "[ok] ${env_name}: wrote ${pth_path}"
    echo "     mclaw -> ${imported_path}"
done

echo "[done] Added ${PROJECT_ROOT} to both MClaw conda environments."
