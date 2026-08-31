#!/usr/bin/env bash
set -euo pipefail

# Resolve paths relative to this repository.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${ORDERING_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
INPUT_DIR="${ORDERING_INPUT_DIR:-${PROJECT_ROOT}/OBELIX/cifs}"
RESULT_DIR="${ORDERING_RESULT_DIR:-${PROJECT_ROOT}/results/obelix_random10_321}"
DEVICE="${ORDERING_DEVICE:-cpu}"

ORDERING_SCRIPT="${SCRIPT_DIR}/order_disordered_cifs.py"
OUTPUT_DIR="${RESULT_DIR}/cifs"
REPORT_PATH="${RESULT_DIR}/report.csv"
ANOMALY_PATH="${RESULT_DIR}/anomalies.csv"

COMMON_ARGS=(
  --input-dir "${INPUT_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --report-path "${REPORT_PATH}"
  --anomaly-path "${ANOMALY_PATH}"
  --candidate-method random
  --num-candidates 10
  --ranker mattersim
  --keep-top 1
  --no-relax-cell
  --fmax 0.2
  --relax-steps 300
  --relax-timeout 600
  --step-progress-every 25
  --max-natoms-per-batch 256
  --include-ordered
  --progress-every 1
  --device "${DEVICE}"
)

usage() {
  printf '%s\n' \
    "Usage: $0 {start|retry|status|log|stop}" \
    "" \
    "  start   Resume the full 321-CIF run in the background." \
    "  retry   Retry only unresolved anomalies with new random seeds." \
    "  status  Show process state and completed report rows." \
    "  log     Follow the active or most recent log." \
    "  stop    Stop active start/retry jobs." \
    "" \
    "Optional environment variables:" \
    "  ORDERING_DEVICE=cpu|cuda" \
    "  ORDERING_RESULT_DIR=/path/to/results" \
    "  ORDERING_INPUT_DIR=/path/to/cifs" \
    "  ORDERING_PYTHON=/path/to/python"
}

require_files() {
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    printf 'Python executable not found: %s\n' "${PYTHON_BIN}" >&2
    exit 1
  fi
  if [[ ! -f "${ORDERING_SCRIPT}" ]]; then
    printf 'Ordering script not found: %s\n' "${ORDERING_SCRIPT}" >&2
    exit 1
  fi
  if [[ ! -d "${INPUT_DIR}" ]]; then
    printf 'Input directory not found: %s\n' "${INPUT_DIR}" >&2
    exit 1
  fi
  mkdir -p "${RESULT_DIR}" "${OUTPUT_DIR}"
}

pid_is_running() {
  local pid_file="$1"
  local pid=""
  [[ -f "${pid_file}" ]] || return 1
  pid="$(<"${pid_file}")"
  [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
  kill -0 "${pid}" 2>/dev/null
}

ensure_idle() {
  local pid_file=""
  for pid_file in "${RESULT_DIR}/resume.pid" "${RESULT_DIR}/retry.pid"; do
    if pid_is_running "${pid_file}"; then
      printf 'A job is already running: PID %s (%s)\n' "$(<"${pid_file}")" "${pid_file}" >&2
      exit 1
    fi
  done
}

start_job() {
  local job_name="$1"
  local mode_arg="$2"
  local log_path="${RESULT_DIR}/${job_name}.log"
  local pid_path="${RESULT_DIR}/${job_name}.pid"

  require_files
  ensure_idle
  printf '\n[%s] Starting %s job on device=%s\n' "$(date --iso-8601=seconds)" "${job_name}" "${DEVICE}" >> "${log_path}"
  nohup "${PYTHON_BIN}" "${ORDERING_SCRIPT}" \
    "${COMMON_ARGS[@]}" "${mode_arg}" >> "${log_path}" 2>&1 &
  local job_pid=$!
  printf '%s\n' "${job_pid}" > "${pid_path}"
  printf '%s job started: PID=%s\nLog: %s\n' "${job_name}" "${job_pid}" "${log_path}"
}

show_status() {
  local pid_file=""
  local found=0
  for pid_file in "${RESULT_DIR}/resume.pid" "${RESULT_DIR}/retry.pid"; do
    if pid_is_running "${pid_file}"; then
      found=1
      ps -p "$(<"${pid_file}")" -o pid,etime,stat,%cpu,%mem,args
    fi
  done
  if (( found == 0 )); then
    printf 'No ordering job is running.\n'
  fi

  if [[ -f "${REPORT_PATH}" ]]; then
    local report_lines
    report_lines="$(wc -l < "${REPORT_PATH}")"
    printf 'Completed report rows: %s\n' "$(( report_lines > 0 ? report_lines - 1 : 0 ))"
  else
    printf 'Report not created yet: %s\n' "${REPORT_PATH}"
  fi
  printf 'Results: %s\n' "${RESULT_DIR}"
}

follow_log() {
  local log_path="${RESULT_DIR}/resume.log"
  if pid_is_running "${RESULT_DIR}/retry.pid" || \
    [[ -f "${RESULT_DIR}/retry.log" && ( ! -f "${log_path}" || "${RESULT_DIR}/retry.log" -nt "${log_path}" ) ]]; then
    log_path="${RESULT_DIR}/retry.log"
  fi
  if [[ ! -f "${log_path}" ]]; then
    printf 'Log not found: %s\n' "${log_path}" >&2
    exit 1
  fi
  tail -n 50 -f "${log_path}"
}

stop_jobs() {
  local pid_file=""
  local stopped=0
  for pid_file in "${RESULT_DIR}/resume.pid" "${RESULT_DIR}/retry.pid"; do
    if pid_is_running "${pid_file}"; then
      local pid
      pid="$(<"${pid_file}")"
      kill "${pid}"
      printf 'Stop signal sent: PID=%s\n' "${pid}"
      stopped=1
    fi
  done
  if (( stopped == 0 )); then
    printf 'No active ordering job found.\n'
  fi
}

case "${1:-}" in
  start)
    start_job resume --resume
    ;;
  retry)
    start_job retry --retry-anomalies
    ;;
  status)
    show_status
    ;;
  log)
    follow_log
    ;;
  stop)
    stop_jobs
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
