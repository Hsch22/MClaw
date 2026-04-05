#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: $0 <output_root>" >&2
    exit 2
fi

OUTPUT_ROOT="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_FILE="${OUTPUT_ROOT}/orchestrator.log"

LIMIT="${LIMIT:-50}"
METHODS="${METHODS:-action,hidden_state,output_grad,logprob,logit_distribution}"
REVIEW_ITEMS="${REVIEW_ITEMS:-16}"

log() {
    printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${LOG_FILE}"
}

wait_for_exit_code() {
    local task_name="$1"
    local exit_file="${OUTPUT_ROOT}/${task_name}/audit.exit_code"
    while [ ! -f "${exit_file}" ]; do
        sleep 15
    done
    local status
    status="$(cat "${exit_file}")"
    log "${task_name} finished with exit_code=${status}"
}

start_audit_session() {
    local session_name="$1"
    local cuda_devices="$2"
    local task_name="$3"
    local data_file="$4"
    local env_addr="$5"
    local output_dir="${OUTPUT_ROOT}/${task_name}"
    tmux new-session -d -s "${session_name}" \
        "bash -lc 'cd ${PROJECT_ROOT} && export CUDA_VISIBLE_DEVICES=${cuda_devices} LIMIT=${LIMIT} METHODS=${METHODS} && exec bash ${PROJECT_ROOT}/tools/run_root_audit_task.sh ${task_name} ${data_file} ${env_addr} ${output_dir}'"
    log "started ${session_name} (${task_name}) on CUDA_VISIBLE_DEVICES=${cuda_devices}"
}

render_manual_review() {
    local task_name="$1"
    local jsonl_path="${OUTPUT_ROOT}/${task_name}/root_cluster_audit.jsonl"
    local output_path="${OUTPUT_ROOT}/${task_name}/manual_review.md"
    if [ ! -f "${jsonl_path}" ]; then
        log "skip manual review for ${task_name}: missing ${jsonl_path}"
        return 0
    fi
    python "${PROJECT_ROOT}/tools/render_manual_review_markdown.py" \
        --input "${jsonl_path}" \
        --output "${output_path}" \
        --max-items "${REVIEW_ITEMS}" \
        --title "Manual Review: ${task_name} trainlike" \
        >> "${LOG_FILE}" 2>&1
    log "rendered manual review for ${task_name}"
}

write_completion_summary() {
    local summary_file="${OUTPUT_ROOT}/completion_summary.md"
    {
        echo "# Train-like Root Audit"
        echo
        echo "- output_root: \`${OUTPUT_ROOT}\`"
        echo "- limit_per_task: \`${LIMIT}\`"
        echo "- methods: \`${METHODS}\`"
        echo
        for task_name in textcraft babyai maze weather; do
            local exit_code_file="${OUTPUT_ROOT}/${task_name}/audit.exit_code"
            local exit_code="missing"
            if [ -f "${exit_code_file}" ]; then
                exit_code="$(cat "${exit_code_file}")"
            fi
            echo "## ${task_name}"
            echo
            echo "- exit_code: \`${exit_code}\`"
            echo "- summary_json: \`${OUTPUT_ROOT}/${task_name}/summary.json\`"
            echo "- manual_review: \`${OUTPUT_ROOT}/${task_name}/manual_review.md\`"
            echo "- audit_log: \`${OUTPUT_ROOT}/${task_name}/audit.log\`"
            echo
        done
    } > "${summary_file}"
    log "wrote ${summary_file}"
}

log "waiting for wave1"
wait_for_exit_code textcraft
wait_for_exit_code babyai

log "starting wave2"
start_audit_session "mclaw_r3_audit_maze" "2,4" "maze" \
    "/mnt/kangshijia/husicheng/AgentGym-RL-Data-ID/train/lmrlgym_maze_train.json" \
    "http://127.0.0.1:39416/maze"
start_audit_session "mclaw_r3_audit_weather" "5,7" "weather" \
    "/mnt/kangshijia/husicheng/AgentGym-RL-Data-ID/train/tool_weather_train.json" \
    "http://127.0.0.1:39410"

wait_for_exit_code maze
wait_for_exit_code weather

for task_name in textcraft babyai maze weather; do
    render_manual_review "${task_name}"
done

write_completion_summary

cc-connect send -m "train-like root-audit 已完成: ${OUTPUT_ROOT}"
log "notification sent"
