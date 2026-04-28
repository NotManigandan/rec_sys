#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/movielens}"
DATASET_VARIANT="${DATASET_VARIANT:-ml-32m}"
OUT_DIR="${OUT_DIR:-${ROOT}/artifacts/ml32m_ablations}"
LOG_DIR="${OUT_DIR}/logs"
BUNDLE_DIR="${OUT_DIR}/bundles"

mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${BUNDLE_DIR}"

NUM_SHARDS="${NUM_SHARDS:-16}"
MALICIOUS_SHARD_ID="${MALICIOUS_SHARD_ID:-0}"
DEVICE="${DEVICE:-cuda}"
NUM_GPUS="${NUM_GPUS:-8}"
SEED="${SEED:-42}"
DISABLE_PROGRESS="${DISABLE_PROGRESS:-0}"

LOCAL_EPOCHS="${LOCAL_EPOCHS:-2}"
FEDERATED_ROUNDS="${FEDERATED_ROUNDS:-12}"
TOP_K="${TOP_K:-10}"
NUM_EVAL_NEGATIVES="${NUM_EVAL_NEGATIVES:-150}"
MIN_POSITIVE_RATING="${MIN_POSITIVE_RATING:-4.0}"
MAX_POSITIVE_INTERACTIONS="${MAX_POSITIVE_INTERACTIONS:-200}"

MF_BATCH_SIZE="${MF_BATCH_SIZE:-8192}"
MF_EMBEDDING_DIM="${MF_EMBEDDING_DIM:-128}"
MF_EVAL_BATCH_SIZE="${MF_EVAL_BATCH_SIZE:-8192}"

NEURAL_BATCH_SIZE="${NEURAL_BATCH_SIZE:-4096}"
NEURAL_EMBEDDING_DIM="${NEURAL_EMBEDDING_DIM:-128}"
NEURAL_EVAL_BATCH_SIZE="${NEURAL_EVAL_BATCH_SIZE:-512}"
NEURAL_HIDDEN_DIM="${NEURAL_HIDDEN_DIM:-2048}"
NEURAL_MLP_LAYERS="${NEURAL_MLP_LAYERS:-4}"
NEURAL_DROPOUT="${NEURAL_DROPOUT:-0.1}"

ATTACK_BUDGETS="${ATTACK_BUDGETS:-0.01,0.5,1.0,2.0}"
ATTACK_FILLER_ITEMS_PER_USER="${ATTACK_FILLER_ITEMS_PER_USER:-32}"
ATTACK_NEUTRAL_ITEMS_PER_USER="${ATTACK_NEUTRAL_ITEMS_PER_USER:-0}"
ATTACK_TARGET_WEIGHT="${ATTACK_TARGET_WEIGHT:-40}"
ATTACK_FILLER_WEIGHT="${ATTACK_FILLER_WEIGHT:-0.5}"
ATTACK_NEUTRAL_WEIGHT="${ATTACK_NEUTRAL_WEIGHT:-0.0}"
ATTACK_FILLER_POOL_SIZE="${ATTACK_FILLER_POOL_SIZE:-128}"
ATTACK_NEUTRAL_POOL_SIZE="${ATTACK_NEUTRAL_POOL_SIZE:-128}"

FOCUS_TOP_K="${FOCUS_TOP_K:-3}"
FOCUS_FACTOR="${FOCUS_FACTOR:-2.0}"
CLIP_FACTOR="${CLIP_FACTOR:-1.0}"
TRIM_RATIO="${TRIM_RATIO:-0.25}"

SYSTEM_ATTACK_BUDGETS="${SYSTEM_ATTACK_BUDGETS:-0.5}"
SYSTEM_FEDERATED_ROUNDS="${SYSTEM_FEDERATED_ROUNDS:-8}"
SYSTEM_LOCAL_EPOCHS="${SYSTEM_LOCAL_EPOCHS:-2}"
SYSTEM_SINGLE_GPU="${SYSTEM_SINGLE_GPU:-1}"
SYSTEM_MULTI_GPU="${SYSTEM_MULTI_GPU:-8}"

TARGET_GENRE="${TARGET_GENRE:-}"
TARGET_ITEM_ID="${TARGET_ITEM_ID:-}"

MF_BUNDLE="${BUNDLE_DIR}/mf_bpr_attack_bundle.json"
NEURAL_BUNDLE="${BUNDLE_DIR}/neural_bpr_attack_bundle.json"
SYSTEM_BUNDLE="${BUNDLE_DIR}/system_focus_bundle.json"
SUMMARY_TSV="${OUT_DIR}/run_times.tsv"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  RESOLVED_PYTHON_BIN="$(command -v "${PYTHON_BIN}" 2>/dev/null || true)"
  if [[ -n "${RESOLVED_PYTHON_BIN}" && -x "${RESOLVED_PYTHON_BIN}" ]]; then
    PYTHON_BIN="${RESOLVED_PYTHON_BIN}"
  else
    echo "Python binary not found or not executable: ${PYTHON_BIN}" >&2
    exit 1
  fi
fi

if [[ "${DEVICE}" == "cuda" ]] && [[ "${NUM_GPUS}" -lt 1 ]]; then
  echo "NUM_GPUS must be positive when DEVICE=cuda" >&2
  exit 1
fi

COMMON_ARGS=(
  --data-root "${DATA_ROOT}"
  --dataset-variant "${DATASET_VARIANT}"
  --num-shards "${NUM_SHARDS}"
  --malicious-shard-id "${MALICIOUS_SHARD_ID}"
  --local-epochs "${LOCAL_EPOCHS}"
  --federated-rounds "${FEDERATED_ROUNDS}"
  --top-k "${TOP_K}"
  --num-eval-negatives "${NUM_EVAL_NEGATIVES}"
  --min-positive-rating "${MIN_POSITIVE_RATING}"
  --seed "${SEED}"
  --device "${DEVICE}"
)

if [[ "${DEVICE}" == "cuda" ]]; then
  COMMON_ARGS+=(--num-gpus "${NUM_GPUS}")
fi

if [[ "${DISABLE_PROGRESS}" == "1" ]]; then
  COMMON_ARGS+=(--disable-progress)
fi

TARGET_ARGS=()
if [[ -n "${TARGET_GENRE}" ]]; then
  TARGET_ARGS+=(--target-genre "${TARGET_GENRE}")
fi
if [[ -n "${TARGET_ITEM_ID}" ]]; then
  TARGET_ARGS+=(--target-item-id "${TARGET_ITEM_ID}")
fi

ATTACK_ARGS=(
  --attack-budgets "${ATTACK_BUDGETS}"
  --attack-filler-items-per-user "${ATTACK_FILLER_ITEMS_PER_USER}"
  --attack-neutral-items-per-user "${ATTACK_NEUTRAL_ITEMS_PER_USER}"
  --attack-target-weight "${ATTACK_TARGET_WEIGHT}"
  --attack-filler-weight "${ATTACK_FILLER_WEIGHT}"
  --attack-neutral-weight "${ATTACK_NEUTRAL_WEIGHT}"
  --attack-filler-pool-size "${ATTACK_FILLER_POOL_SIZE}"
  --attack-neutral-pool-size "${ATTACK_NEUTRAL_POOL_SIZE}"
)

FOCUS_ARGS=(
  --defense-method focus_clip_mean
  --focus-top-k "${FOCUS_TOP_K}"
  --focus-factor "${FOCUS_FACTOR}"
)

usage() {
  cat <<EOF
Usage:
  $(basename "$0") [stage ...]

Stages:
  main       Run the main MF-BPR attack-budget sweep and save an exact attack bundle.
  defenses   Reuse the saved MF-BPR bundle and sweep defense methods.
  models     Run model-comparison sweeps for the supported federated models.
  systems    Run runtime / systems ablations for Table 3.
  all        Run every stage above.

Examples:
  $(basename "$0") main defenses
  $(basename "$0") models
  $(basename "$0") systems

Environment overrides:
  DATA_ROOT, OUT_DIR, NUM_GPUS, NUM_SHARDS, SEED, TARGET_GENRE, TARGET_ITEM_ID
  ATTACK_BUDGETS, MAX_POSITIVE_INTERACTIONS, MF_EMBEDDING_DIM, NEURAL_HIDDEN_DIM, ...

Notes:
  - Leave TARGET_GENRE and TARGET_ITEM_ID empty to use the heuristic target selector.
  - The current federated benchmark supports two learned models: mf_bpr and neural_bpr.
    If you want a third model row for the poster, add it later or pair these runs with a
    simpler external baseline when you build Table 4.
EOF
}

record_header() {
  if [[ ! -f "${SUMMARY_TSV}" ]]; then
    printf "label\telapsed_seconds\texit_code\toutput_json\n" > "${SUMMARY_TSV}"
  fi
}

run_logged() {
  local label="$1"
  local output_json="$2"
  shift 2
  local log_path="${LOG_DIR}/${label}.log"
  local start_ts
  local end_ts
  local elapsed
  local rc

  record_header
  start_ts="$(date +%s)"
  {
    echo "[$(date -Iseconds)] START ${label}"
    printf 'CMD:'
    for arg in "$@"; do
      printf ' %q' "${arg}"
    done
    printf '\n'
  } | tee "${log_path}"

  set +e
  "$@" 2>&1 | tee -a "${log_path}"
  rc="${PIPESTATUS[0]}"
  set -e

  end_ts="$(date +%s)"
  elapsed="$((end_ts - start_ts))"
  {
    echo "[$(date -Iseconds)] END ${label} rc=${rc} elapsed_seconds=${elapsed}"
    printf "%s\t%s\t%s\t%s\n" "${label}" "${elapsed}" "${rc}" "${output_json}" >> "${SUMMARY_TSV}"
  } | tee -a "${log_path}"

  if [[ "${rc}" -ne 0 ]]; then
    return "${rc}"
  fi
}

ensure_file() {
  local path="$1"
  local hint="$2"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required file: ${path}" >&2
    echo "Hint: ${hint}" >&2
    exit 1
  fi
}

run_main() {
  local output_json="${OUT_DIR}/mf_bpr_attack_baseline.json"
  run_logged \
    "mf_bpr_attack_baseline" \
    "${output_json}" \
    "${PYTHON_BIN}" -m recsys.federated.benchmark \
    "${COMMON_ARGS[@]}" \
    "${TARGET_ARGS[@]}" \
    --model mf_bpr \
    --batch-size "${MF_BATCH_SIZE}" \
    --embedding-dim "${MF_EMBEDDING_DIM}" \
    --eval-every 2 \
    --eval-batch-size "${MF_EVAL_BATCH_SIZE}" \
    --max-positive-interactions "${MAX_POSITIVE_INTERACTIONS}" \
    "${ATTACK_ARGS[@]}" \
    --defense-method none \
    --save-attack-bundle "${MF_BUNDLE}" \
    --output "${output_json}"
}

run_defenses() {
  ensure_file "${MF_BUNDLE}" "Run '$(basename "$0") main' first."

  local defenses=(
    "clip_mean"
    "clip_trimmed_mean"
    "focus_clip_mean"
    "focus_clip_trimmed_mean"
  )

  local defense
  for defense in "${defenses[@]}"; do
    local output_json="${OUT_DIR}/mf_bpr_${defense}.json"
    local extra_args=()
    case "${defense}" in
      clip_mean)
        extra_args=(--defense-method clip_mean --clip-factor "${CLIP_FACTOR}")
        ;;
      clip_trimmed_mean)
        extra_args=(--defense-method clip_trimmed_mean --clip-factor "${CLIP_FACTOR}" --trim-ratio "${TRIM_RATIO}")
        ;;
      focus_clip_mean)
        extra_args=(--defense-method focus_clip_mean --focus-top-k "${FOCUS_TOP_K}" --focus-factor "${FOCUS_FACTOR}")
        ;;
      focus_clip_trimmed_mean)
        extra_args=(--defense-method focus_clip_trimmed_mean --focus-top-k "${FOCUS_TOP_K}" --focus-factor "${FOCUS_FACTOR}" --trim-ratio "${TRIM_RATIO}")
        ;;
      *)
        echo "Unknown defense: ${defense}" >&2
        exit 1
        ;;
    esac

    run_logged \
      "mf_bpr_${defense}" \
      "${output_json}" \
      "${PYTHON_BIN}" -m recsys.federated.benchmark \
      "${COMMON_ARGS[@]}" \
      --model mf_bpr \
      --batch-size "${MF_BATCH_SIZE}" \
      --embedding-dim "${MF_EMBEDDING_DIM}" \
      --eval-every 2 \
      --eval-batch-size "${MF_EVAL_BATCH_SIZE}" \
      --max-positive-interactions "${MAX_POSITIVE_INTERACTIONS}" \
      --load-attack-bundle "${MF_BUNDLE}" \
      "${extra_args[@]}" \
      --output "${output_json}"
  done
}

run_models() {
  if [[ ! -f "${MF_BUNDLE}" ]]; then
    run_main
  fi

  local mf_focus_output="${OUT_DIR}/mf_bpr_focus_clip_mean.json"
  if [[ ! -f "${mf_focus_output}" ]]; then
    run_logged \
      "mf_bpr_focus_clip_mean" \
      "${mf_focus_output}" \
      "${PYTHON_BIN}" -m recsys.federated.benchmark \
      "${COMMON_ARGS[@]}" \
      --model mf_bpr \
      --batch-size "${MF_BATCH_SIZE}" \
      --embedding-dim "${MF_EMBEDDING_DIM}" \
      --eval-every 2 \
      --eval-batch-size "${MF_EVAL_BATCH_SIZE}" \
      --max-positive-interactions "${MAX_POSITIVE_INTERACTIONS}" \
      --load-attack-bundle "${MF_BUNDLE}" \
      "${FOCUS_ARGS[@]}" \
      --output "${mf_focus_output}"
  fi

  local neural_attack_output="${OUT_DIR}/neural_bpr_attack_baseline.json"
  run_logged \
    "neural_bpr_attack_baseline" \
    "${neural_attack_output}" \
    "${PYTHON_BIN}" -m recsys.federated.benchmark \
    "${COMMON_ARGS[@]}" \
    "${TARGET_ARGS[@]}" \
    --model neural_bpr \
    --batch-size "${NEURAL_BATCH_SIZE}" \
    --embedding-dim "${NEURAL_EMBEDDING_DIM}" \
    --hidden-dim "${NEURAL_HIDDEN_DIM}" \
    --mlp-layers "${NEURAL_MLP_LAYERS}" \
    --dropout "${NEURAL_DROPOUT}" \
    --eval-every 2 \
    --eval-batch-size "${NEURAL_EVAL_BATCH_SIZE}" \
    --max-positive-interactions "${MAX_POSITIVE_INTERACTIONS}" \
    "${ATTACK_ARGS[@]}" \
    --defense-method none \
    --save-attack-bundle "${NEURAL_BUNDLE}" \
    --output "${neural_attack_output}"

  local neural_focus_output="${OUT_DIR}/neural_bpr_focus_clip_mean.json"
  run_logged \
    "neural_bpr_focus_clip_mean" \
    "${neural_focus_output}" \
    "${PYTHON_BIN}" -m recsys.federated.benchmark \
    "${COMMON_ARGS[@]}" \
    --model neural_bpr \
    --batch-size "${NEURAL_BATCH_SIZE}" \
    --embedding-dim "${NEURAL_EMBEDDING_DIM}" \
    --hidden-dim "${NEURAL_HIDDEN_DIM}" \
    --mlp-layers "${NEURAL_MLP_LAYERS}" \
    --dropout "${NEURAL_DROPOUT}" \
    --eval-every 2 \
    --eval-batch-size "${NEURAL_EVAL_BATCH_SIZE}" \
    --max-positive-interactions "${MAX_POSITIVE_INTERACTIONS}" \
    --load-attack-bundle "${NEURAL_BUNDLE}" \
    "${FOCUS_ARGS[@]}" \
    --output "${neural_focus_output}"
}

run_systems() {
  local system_target_args=("${TARGET_ARGS[@]}")
  local system_attack_args=(
    --attack-budgets "${SYSTEM_ATTACK_BUDGETS}"
    --attack-filler-items-per-user "${ATTACK_FILLER_ITEMS_PER_USER}"
    --attack-neutral-items-per-user "${ATTACK_NEUTRAL_ITEMS_PER_USER}"
    --attack-target-weight "${ATTACK_TARGET_WEIGHT}"
    --attack-filler-weight "${ATTACK_FILLER_WEIGHT}"
    --attack-neutral-weight "${ATTACK_NEUTRAL_WEIGHT}"
    --attack-filler-pool-size "${ATTACK_FILLER_POOL_SIZE}"
    --attack-neutral-pool-size "${ATTACK_NEUTRAL_POOL_SIZE}"
  )

  local base_system_args=(
    --data-root "${DATA_ROOT}"
    --dataset-variant "${DATASET_VARIANT}"
    --num-shards "${NUM_SHARDS}"
    --malicious-shard-id "${MALICIOUS_SHARD_ID}"
    --model mf_bpr
    --local-epochs "${SYSTEM_LOCAL_EPOCHS}"
    --federated-rounds "${SYSTEM_FEDERATED_ROUNDS}"
    --batch-size "${MF_BATCH_SIZE}"
    --embedding-dim "${MF_EMBEDDING_DIM}"
    --eval-every 2
    --eval-batch-size "${MF_EVAL_BATCH_SIZE}"
    --top-k "${TOP_K}"
    --num-eval-negatives "${NUM_EVAL_NEGATIVES}"
    --min-positive-rating "${MIN_POSITIVE_RATING}"
    --seed "${SEED}"
    --device "${DEVICE}"
  )
  if [[ "${DISABLE_PROGRESS}" == "1" ]]; then
    base_system_args+=(--disable-progress)
  fi

  run_logged \
    "systems_single_gpu_no_cap_focus" \
    "${OUT_DIR}/systems_single_gpu_no_cap_focus.json" \
    "${PYTHON_BIN}" -m recsys.federated.benchmark \
    "${base_system_args[@]}" \
    "${system_target_args[@]}" \
    --num-gpus "${SYSTEM_SINGLE_GPU}" \
    "${system_attack_args[@]}" \
    "${FOCUS_ARGS[@]}" \
    --output "${OUT_DIR}/systems_single_gpu_no_cap_focus.json"

  run_logged \
    "systems_multi_gpu_no_cap_focus" \
    "${OUT_DIR}/systems_multi_gpu_no_cap_focus.json" \
    "${PYTHON_BIN}" -m recsys.federated.benchmark \
    "${base_system_args[@]}" \
    "${system_target_args[@]}" \
    --num-gpus "${SYSTEM_MULTI_GPU}" \
    "${system_attack_args[@]}" \
    "${FOCUS_ARGS[@]}" \
    --output "${OUT_DIR}/systems_multi_gpu_no_cap_focus.json"

  run_logged \
    "systems_multi_gpu_cap_focus" \
    "${OUT_DIR}/systems_multi_gpu_cap_focus.json" \
    "${PYTHON_BIN}" -m recsys.federated.benchmark \
    "${base_system_args[@]}" \
    "${system_target_args[@]}" \
    --num-gpus "${SYSTEM_MULTI_GPU}" \
    --max-positive-interactions "${MAX_POSITIVE_INTERACTIONS}" \
    "${system_attack_args[@]}" \
    "${FOCUS_ARGS[@]}" \
    --output "${OUT_DIR}/systems_multi_gpu_cap_focus.json"

  run_logged \
    "systems_multi_gpu_cap_bundle_seed" \
    "${OUT_DIR}/systems_multi_gpu_cap_bundle_seed.json" \
    "${PYTHON_BIN}" -m recsys.federated.benchmark \
    "${base_system_args[@]}" \
    "${system_target_args[@]}" \
    --num-gpus "${SYSTEM_MULTI_GPU}" \
    --max-positive-interactions "${MAX_POSITIVE_INTERACTIONS}" \
    "${system_attack_args[@]}" \
    --defense-method none \
    --save-attack-bundle "${SYSTEM_BUNDLE}" \
    --output "${OUT_DIR}/systems_multi_gpu_cap_bundle_seed.json"

  run_logged \
    "systems_multi_gpu_cap_bundle_focus" \
    "${OUT_DIR}/systems_multi_gpu_cap_bundle_focus.json" \
    "${PYTHON_BIN}" -m recsys.federated.benchmark \
    "${base_system_args[@]}" \
    --num-gpus "${SYSTEM_MULTI_GPU}" \
    --max-positive-interactions "${MAX_POSITIVE_INTERACTIONS}" \
    --load-attack-bundle "${SYSTEM_BUNDLE}" \
    "${FOCUS_ARGS[@]}" \
    --output "${OUT_DIR}/systems_multi_gpu_cap_bundle_focus.json"
}

run_stage() {
  local stage="$1"
  case "${stage}" in
    main)
      run_main
      ;;
    defenses)
      run_defenses
      ;;
    models)
      run_models
      ;;
    systems)
      run_systems
      ;;
    all)
      run_main
      run_defenses
      run_models
      run_systems
      ;;
    help|-h|--help)
      usage
      ;;
    *)
      echo "Unknown stage: ${stage}" >&2
      usage
      exit 1
      ;;
  esac
}

if [[ "$#" -eq 0 ]]; then
  usage
  exit 0
fi

for stage in "$@"; do
  run_stage "${stage}"
done
