#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_DIR}}"
AGENTGYM_DIR="${AGENTGYM_DIR:-/mnt/kangshijia/husicheng/AgentGym}"
UV_BIN="${UV_BIN:-uv}"

usage() {
    cat <<EOF
Usage: bash install_agentenv_uv_envs.sh [babyai|weather|lmrlgym|all]

This script creates dedicated uv environments for the additional AgentGym envservers
used by MClaw examples:
  babyai  -> ${PROJECT_ROOT}/.venv-agentenv-babyai
  weather -> ${PROJECT_ROOT}/.venv-agentenv-tool-weather
  lmrlgym -> ${PROJECT_ROOT}/.venv-agentenv-lmrlgym

Optional environment variables:
  PROJECT_ROOT  Override the MClaw project root
  AGENTGYM_DIR  Override the AgentGym source tree
  UV_BIN        Override the uv executable
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

ensure_file() {
    local path="${1}"
    if [[ ! -f "${path}" ]]; then
        echo "[error] Missing file: ${path}" >&2
        exit 1
    fi
}

create_venv() {
    local venv_dir="${1}"
    local python_version="${2}"
    echo "[setup] Creating ${venv_dir} with Python ${python_version}"
    "${UV_BIN}" venv --python "${python_version}" "${venv_dir}"
}

install_frozen_requirements() {
    local python_bin="${1}"
    local freeze_file="${2}"
    echo "[setup] Installing frozen requirements from ${freeze_file}"
    "${UV_BIN}" pip install --python "${python_bin}" -r "${freeze_file}"
}

install_editable_no_deps() {
    local python_bin="${1}"
    local package_dir="${2}"
    echo "[setup] Installing editable package ${package_dir}"
    "${UV_BIN}" pip install --python "${python_bin}" --no-deps -e "${package_dir}"
}

install_babyai() {
    local venv_dir="${PROJECT_ROOT}/.venv-agentenv-babyai"
    local python_bin="${venv_dir}/bin/python"
    local freeze_file="${PROJECT_ROOT}/envs/agentenv-babyai-uv-freeze.txt"

    ensure_file "${freeze_file}"
    ensure_file "${AGENTGYM_DIR}/agentenv-babyai/pyproject.toml"
    create_venv "${venv_dir}" "3.12"
    install_frozen_requirements "${python_bin}" "${freeze_file}"
    install_editable_no_deps "${python_bin}" "${AGENTGYM_DIR}/agentenv-babyai"
    echo "[done] babyai -> ${python_bin}"
}

install_weather() {
    local venv_dir="${PROJECT_ROOT}/.venv-agentenv-tool-weather"
    local python_bin="${venv_dir}/bin/python"
    local freeze_file="${PROJECT_ROOT}/envs/agentenv-tool-weather-uv-freeze.txt"

    ensure_file "${freeze_file}"
    ensure_file "${AGENTGYM_DIR}/agentenv-tool/pyproject.toml"
    ensure_file "${AGENTGYM_DIR}/agentenv-tool/Toolusage/toolusage/setup.py"
    create_venv "${venv_dir}" "3.8.13"
    install_frozen_requirements "${python_bin}" "${freeze_file}"
    install_editable_no_deps "${python_bin}" "${AGENTGYM_DIR}/agentenv-tool/Toolusage/toolusage"
    install_editable_no_deps "${python_bin}" "${AGENTGYM_DIR}/agentenv-tool"
    echo "[done] weather -> ${python_bin}"
}

install_lmrlgym() {
    local venv_dir="${PROJECT_ROOT}/.venv-agentenv-lmrlgym"
    local python_bin="${venv_dir}/bin/python"
    local freeze_file="${PROJECT_ROOT}/envs/agentenv-lmrlgym-uv-freeze.txt"

    ensure_file "${freeze_file}"
    ensure_file "${AGENTGYM_DIR}/agentenv-lmrlgym/pyproject.toml"
    ensure_file "${AGENTGYM_DIR}/agentenv-lmrlgym/lmrlgym/setup.py"
    create_venv "${venv_dir}" "3.9.12"
    install_frozen_requirements "${python_bin}" "${freeze_file}"
    install_editable_no_deps "${python_bin}" "${AGENTGYM_DIR}/agentenv-lmrlgym/lmrlgym"
    install_editable_no_deps "${python_bin}" "${AGENTGYM_DIR}/agentenv-lmrlgym"
    echo "[done] lmrlgym -> ${python_bin}"
}

target="${1:-all}"
case "${target}" in
    babyai)
        install_babyai
        ;;
    weather)
        install_weather
        ;;
    lmrlgym)
        install_lmrlgym
        ;;
    all)
        install_babyai
        install_weather
        install_lmrlgym
        ;;
    *)
        usage >&2
        exit 1
        ;;
esac
