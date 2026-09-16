#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_REVISED_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${CODE_REVISED_DIR}/.." && pwd)"

BASELINE_MANIFEST="${CODE_REVISED_DIR}/baseline_checkpoints.tsv"
STRUCTURE_MD="${PROJECT_ROOT}/structure.md"
LOG_ROOT="${CODE_REVISED_DIR}/logs_pth_sync"
OLD_CODE_RESULTS_ROOT="/home/cq/code/AD-MT/code/results"
REMOTE=""
DRY_RUN=0
VERIFY_ONLY=0
LIST_ONLY=0
YES=0
OVERWRITE=0
INCLUDE_EXTRA_STRUCTURE=0
EVAL_MODEL_TYPES="ema"

RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${LOG_ROOT}/sync_${RUN_TIMESTAMP}_pid$$"
MASTER_LOG="${RUN_DIR}/sync.log"
LIST1="${RUN_DIR}/list1_stage_pths.tsv"
LIST2="${RUN_DIR}/list2_checkpoint_call_paths.tsv"
LIST3="${RUN_DIR}/list3_structure_3070_pths.tsv"
SYNC_MANIFEST_RAW="${RUN_DIR}/sync_manifest.raw.tsv"
SYNC_MANIFEST="${RUN_DIR}/sync_manifest.tsv"
SYNC_RESULT="${RUN_DIR}/sync_result.tsv"
REPORT_MD="${RUN_DIR}/sync_report.md"

usage() {
  cat <<'EOF'
Usage:
  bash tools/sync_required_pths_to_4090.sh [options]

Purpose:
  Run this script on the 3070 server. It builds pth dependency lists for all
  major revision stages and rsyncs old external checkpoints needed on 4090.

Options:
  --remote USER@HOST             4090 SSH target, e.g. cq@192.168.203.58.
  --baseline-manifest FILE       TSV used by eval-baselines. Default:
                                 ../baseline_checkpoints.tsv.
  --structure FILE               structure.md path. Default: ../../structure.md.
  --old-code-results-root DIR    Root used to map /results/... paths. Default:
                                 /home/cq/code/AD-MT/code/results.
  --eval-model-types LIST        Comma list used by test stages. Default: ema.
  --include-extra-structure      Also copy 3070 pths listed in structure.md even
                                 if no current stage consumes them.
  --dry-run                      Generate lists and print planned sync only.
  --verify-only                  Do not rsync; only check whether remote files exist.
  --list-only                    Only generate the three inventory lists.
  --overwrite                    Rsync even if the remote pth already exists.
  --yes                          Skip interactive confirmation.
  -h, --help                     Show this help.

Examples:
  bash tools/sync_required_pths_to_4090.sh --list-only
  bash tools/sync_required_pths_to_4090.sh --dry-run --remote cq@192.168.203.58
  bash tools/sync_required_pths_to_4090.sh --remote cq@192.168.203.58 --yes
  bash tools/sync_required_pths_to_4090.sh --remote cq@192.168.203.58 --verify-only
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote)
      REMOTE="$2"
      shift 2
      ;;
    --baseline-manifest)
      BASELINE_MANIFEST="$2"
      shift 2
      ;;
    --structure)
      STRUCTURE_MD="$2"
      shift 2
      ;;
    --old-code-results-root)
      OLD_CODE_RESULTS_ROOT="${2%/}"
      shift 2
      ;;
    --eval-model-types)
      EVAL_MODEL_TYPES="$2"
      shift 2
      ;;
    --include-extra-structure)
      INCLUDE_EXTRA_STRUCTURE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --verify-only)
      VERIFY_ONLY=1
      shift
      ;;
    --list-only)
      LIST_ONLY=1
      shift
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --yes)
      YES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

mkdir -p "$RUN_DIR"
: > "$MASTER_LOG"

log() {
  local message="$*"
  printf '[%s] %s\n' "$(date '+%F %T')" "$message" | tee -a "$MASTER_LOG"
}

die() {
  log "[ERROR] $*"
  exit 1
}

shell_quote() {
  printf '%q' "$1"
}

normalize_path() {
  local path="$1"
  printf '%s' "$path" | sed -E 's#/{2,}#/#g'
}

map_structure_path() {
  local path
  path="$(normalize_path "$1")"
  if [[ "$path" == /results/* ]]; then
    printf '%s%s' "$OLD_CODE_RESULTS_ROOT" "${path#/results}"
  else
    printf '%s' "$path"
  fi
}

append_tsv() {
  local file="$1"
  shift
  local first=1
  {
    for field in "$@"; do
      if [[ "$first" -eq 1 ]]; then
        first=0
      else
        printf '\t'
      fi
      printf '%s' "$field"
    done
    printf '\n'
  } >> "$file"
}

csv_to_words() {
  printf '%s' "$1" | tr ',' ' '
}

add_stage_pth() {
  local stage="$1" role="$2" source_kind="$3" checkpoint_path="$4" should_sync="$5" prerequisite="$6" notes="$7"
  append_tsv "$LIST1" "$stage" "$role" "$source_kind" "$checkpoint_path" "$should_sync" "$prerequisite" "$notes"
}

add_call_path() {
  local stage="$1" script="$2" checkpoint_path="$3" resolved_by="$4" notes="$5"
  append_tsv "$LIST2" "$stage" "$script" "--checkpoint_path" "$checkpoint_path" "$resolved_by" "$notes"
}

add_sync_manifest_row() {
  local reason="$1" source_kind="$2" source_hint="$3" destination_path="$4" required_by_stage="$5" notes="$6"
  append_tsv "$SYNC_MANIFEST_RAW" "$reason" "$source_kind" "$source_hint" "$destination_path" "$required_by_stage" "$notes"
}

add_generated_test_rows() {
  local model_type="$1"
  local path

  path="./results_revised/LA/4_labeled_mr_la_main/vnet/vnet_best_${model_type}_model.pth"
  add_stage_pth "test" "input_checkpoint" "generated_on_4090" "$path" "no" "main:train_la_main" "test stage reads this pth after main finishes"
  add_call_path "test_la_main_${model_type}" "test_performance_3d.py" "$path" "checkpoint_for()" "generated by main, not copied from 3070"

  path="./results_revised/Pancreas/6_labeled_mr_pancreas_main/vnet/vnet_best_${model_type}_model.pth"
  add_stage_pth "test" "input_checkpoint" "generated_on_4090" "$path" "no" "main:train_pancreas_main" "test stage reads this pth after main finishes"
  add_call_path "test_pancreas_main_${model_type}" "test_performance_3d.py" "$path" "checkpoint_for()" "generated by main, not copied from 3070"

  path="./results_revised/Prostate/7_labeled_mr_prostate_main/unet/unet_best_${model_type}_model.pth"
  add_stage_pth "test" "input_checkpoint" "generated_on_4090" "$path" "no" "main:train_prostate_main" "test stage reads this pth after main finishes"
  add_call_path "test_prostate_main_${model_type}" "test_performance_2d.py" "$path" "checkpoint_for()" "generated by main, not copied from 3070"

  path="./results_revised/Prostate/4_labeled_mr_prostate_4l_main/unet/unet_best_${model_type}_model.pth"
  add_stage_pth "test" "input_checkpoint" "generated_on_4090" "$path" "no" "main:train_prostate_4l_main" "test stage reads this pth after main finishes"
  add_call_path "test_prostate_4l_main_${model_type}" "test_performance_2d.py" "$path" "checkpoint_for()" "generated by main, not copied from 3070"
}

add_generated_ablation_rows() {
  local model_type="$1"
  local exp path
  for exp in mr_la_ablate_no_ml mr_la_ablate_hinge mr_la_ablate_l1_alpha05; do
    path="./results_revised/LA/4_labeled_${exp}/vnet/vnet_best_${model_type}_model.pth"
    add_stage_pth "ablation-test" "input_checkpoint" "generated_on_4090" "$path" "no" "ablation:${exp}" "ablation-test reads this pth after ablation finishes"
    add_call_path "test_${exp}_${model_type}" "test_performance_3d.py" "$path" "checkpoint_for()" "generated by ablation, not copied from 3070"
  done
  for exp in mr_prostate_4l_ablate_no_ml mr_prostate_4l_ablate_hinge mr_prostate_4l_ablate_l1_alpha05; do
    path="./results_revised/Prostate/4_labeled_${exp}/unet/unet_best_${model_type}_model.pth"
    add_stage_pth "ablation-test" "input_checkpoint" "generated_on_4090" "$path" "no" "ablation:${exp}" "ablation-test reads this pth after ablation finishes"
    add_call_path "test_${exp}_${model_type}" "test_performance_2d.py" "$path" "checkpoint_for()" "generated by ablation, not copied from 3070"
  done
}

add_generated_multiclass_rows() {
  local model_type="$1"
  local path="./results_revised/ACDC/3_labeled_mr_acdc_multiclass/unet/unet_best_${model_type}_model.pth"
  add_stage_pth "multiclass" "input_checkpoint" "generated_on_4090" "$path" "no" "multiclass:train_acdc_multiclass" "multiclass test reads this pth after its own training finishes"
  add_call_path "test_acdc_multiclass_${model_type}" "test_performance_2d.py" "$path" "checkpoint_for()" "generated inside multiclass stage, not copied from 3070"
}

build_stage_lists() {
  append_tsv "$LIST1" "stage" "role" "source_kind" "checkpoint_path" "should_sync_from_3070" "prerequisite" "notes"
  append_tsv "$LIST2" "stage_or_substage" "test_script" "argument_name" "checkpoint_path" "resolved_by" "notes"
  append_tsv "$LIST3" "line_no" "raw_path" "normalized_4090_destination_path" "source_note"
  append_tsv "$SYNC_MANIFEST_RAW" "reason" "source_kind" "source_hint" "destination_path" "required_by_stage" "notes"

  add_stage_pth "smoke" "no_checkpoint_input" "none" "" "no" "none" "smoke trains briefly and does not require old pth"
  add_stage_pth "main" "no_checkpoint_input" "none" "" "no" "none" "main trains from initialization and writes new pths under results_revised"
  add_stage_pth "ablation" "no_checkpoint_input" "none" "" "no" "none" "ablation trains from initialization and writes new pths under results_revised"
  add_stage_pth "stats" "no_checkpoint_input" "csv_only" "" "no" "test/eval-baselines/ablation-test CSV outputs" "stats reads CSV metrics, not pth files"
  add_stage_pth "summary" "no_checkpoint_input" "csv_and_logs_only" "" "no" "previous result CSV/logs" "summary reads result CSV/logs, not pth files"

  local model_type
  for model_type in $(csv_to_words "$EVAL_MODEL_TYPES"); do
    [[ -z "$model_type" ]] && continue
    add_generated_test_rows "$model_type"
    add_generated_ablation_rows "$model_type"
    add_generated_multiclass_rows "$model_type"
  done

  [[ -f "$BASELINE_MANIFEST" ]] || die "Cannot find baseline manifest: $BASELINE_MANIFEST"
  local line_no=0
  local name dataset root_path data_path res_path exp model num_classes labeled_num model_type checkpoint_path
  while IFS=$'\t' read -r name dataset root_path data_path res_path exp model num_classes labeled_num model_type checkpoint_path; do
    line_no=$((line_no + 1))
    [[ "$line_no" -eq 1 ]] && continue
    [[ -z "${name:-}" || "$name" =~ ^# ]] && continue
    local normalized_path
    normalized_path="$(normalize_path "$checkpoint_path")"
    add_stage_pth "eval-baselines" "input_checkpoint" "external_old_checkpoint" "$normalized_path" "yes" "baseline_checkpoints.tsv:${name}" "old pth must exist on 4090 before eval-baselines"
    if [[ "$model" == "vnet" ]]; then
      add_call_path "eval_baseline_${name}" "test_performance_3d.py" "$normalized_path" "baseline_checkpoints.tsv checkpoint_path" "external old pth"
    else
      add_call_path "eval_baseline_${name}" "test_performance_2d.py" "$normalized_path" "baseline_checkpoints.tsv checkpoint_path" "external old pth"
    fi
    add_sync_manifest_row "baseline_checkpoints.tsv:${name}" "required_external_old_checkpoint" "$normalized_path" "$normalized_path" "eval-baselines" "required by current run_major_revision_experiments.sh"
  done < "$BASELINE_MANIFEST"

  if [[ -f "$STRUCTURE_MD" ]]; then
    awk '/3070/ && /\.pth/ {
      raw_line=$0
      line=$0
      while (match(line, /\/[^ \t`，。]*\.pth/)) {
        path=substr(line, RSTART, RLENGTH)
        print NR "\t" path "\t" raw_line
        line=substr(line, RSTART + RLENGTH)
      }
    }' "$STRUCTURE_MD" | while IFS=$'\t' read -r md_line raw_path raw_line; do
      local mapped
      mapped="$(map_structure_path "$raw_path")"
      append_tsv "$LIST3" "$md_line" "$raw_path" "$mapped" "$raw_line"
      if [[ "$INCLUDE_EXTRA_STRUCTURE" -eq 1 ]]; then
        add_sync_manifest_row "structure.md:${md_line}" "extra_structure_3070_checkpoint" "$mapped" "$mapped" "optional-extra-structure" "not required by current stages unless you add it to baseline_checkpoints.tsv"
      fi
    done
  else
    log "[WARN] structure.md not found: $STRUCTURE_MD"
  fi

  awk -F'\t' '
    NR == 1 { print; next }
    !seen[$4]++ { print }
  ' "$SYNC_MANIFEST_RAW" > "$SYNC_MANIFEST"
}

emit_source_candidates() {
  local hint="$1"
  local normalized rel
  normalized="$(normalize_path "$hint")"
  printf '%s\n' "$normalized"
  if [[ "$normalized" == "$OLD_CODE_RESULTS_ROOT"/* ]]; then
    rel="${normalized#"$OLD_CODE_RESULTS_ROOT"}"
    printf '/results%s\n' "$rel"
  fi
  if [[ "$normalized" == /home/cq/code/AD-MT/code/results/* ]]; then
    rel="${normalized#/home/cq/code/AD-MT/code/results}"
    printf '/results%s\n' "$rel"
  fi
  if [[ "$normalized" == /results/* ]]; then
    printf '%s%s\n' "$OLD_CODE_RESULTS_ROOT" "${normalized#/results}"
  fi
}

select_existing_source() {
  local hint="$1"
  local candidate
  while IFS= read -r candidate; do
    [[ -z "$candidate" ]] && continue
    if [[ -f "$candidate" ]]; then
      printf '%s' "$candidate"
      return 0
    fi
  done < <(emit_source_candidates "$hint" | awk '!seen[$0]++')
  return 1
}

remote_file_exists() {
  local path="$1"
  local q
  q="$(shell_quote "$path")"
  ssh -n "$REMOTE" "test -f $q" >/dev/null 2>&1
}

remote_mkdir_parent() {
  local path="$1"
  local parent q
  parent="$(dirname "$path")"
  q="$(shell_quote "$parent")"
  ssh -n "$REMOTE" "mkdir -p $q"
}

rsync_one() {
  local source_path="$1"
  local destination_path="$2"
  local qdest
  qdest="$(shell_quote "$destination_path")"
  rsync -avz --progress -- "$source_path" "${REMOTE}:${qdest}"
}

count_manifest_rows() {
  awk 'END { print (NR > 0 ? NR - 1 : 0) }' "$SYNC_MANIFEST"
}

sync_or_verify() {
  append_tsv "$SYNC_RESULT" "destination_path" "source_path" "local_source_status" "remote_status_before" "action" "rsync_exit" "remote_status_after" "final_status" "reason"

  if [[ "$LIST_ONLY" -eq 1 ]]; then
    log "List-only mode. No remote check or rsync will be performed."
    return 0
  fi

  if [[ "$DRY_RUN" -eq 0 && -z "$REMOTE" ]]; then
    die "--remote USER@HOST is required unless you use --dry-run or --list-only"
  fi
  if [[ "$VERIFY_ONLY" -eq 1 && -z "$REMOTE" ]]; then
    die "--remote USER@HOST is required for --verify-only"
  fi

  local total
  total="$(count_manifest_rows)"
  log "Sync manifest rows: $total"
  if [[ "$total" -eq 0 ]]; then
    log "No pth needs syncing."
    return 0
  fi

  if [[ "$DRY_RUN" -eq 0 && "$VERIFY_ONLY" -eq 0 && "$YES" -eq 0 ]]; then
    echo
    echo "About to rsync required pth files to: $REMOTE"
    echo "Manifest: $SYNC_MANIFEST"
    read -r -p "Continue? [y/N] " answer
    if [[ ! "$answer" =~ ^[Yy]$ ]]; then
      log "Cancelled by user."
      return 1
    fi
  fi

  local failures=0
  local line_no=0
  local reason source_kind source_hint destination_path required_by_stage notes
  while IFS=$'\t' read -r reason source_kind source_hint destination_path required_by_stage notes <&3; do
    line_no=$((line_no + 1))
    [[ "$line_no" -eq 1 ]] && continue
    [[ -z "${destination_path:-}" ]] && continue

    local source_path="" local_status="missing" remote_before="not_checked" action="none" rsync_exit="NA" remote_after="not_checked" final_status="unknown"
    if source_path="$(select_existing_source "$source_hint")"; then
      local_status="exists"
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
      action="dry_run_no_remote_action"
      final_status="dry_run"
      log "[DRY-RUN] ${reason}: source=${source_path:-MISSING} -> ${destination_path}"
      append_tsv "$SYNC_RESULT" "$destination_path" "${source_path:-}" "$local_status" "$remote_before" "$action" "$rsync_exit" "$remote_after" "$final_status" "$reason"
      continue
    fi

    if remote_file_exists "$destination_path"; then
      remote_before="exists"
    else
      remote_before="missing"
    fi

    if [[ "$VERIFY_ONLY" -eq 1 ]]; then
      action="verify_only"
      remote_after="$remote_before"
      if [[ "$remote_before" == "exists" ]]; then
        final_status="ok_remote_exists"
      else
        final_status="missing_remote"
        failures=$((failures + 1))
      fi
      append_tsv "$SYNC_RESULT" "$destination_path" "${source_path:-}" "$local_status" "$remote_before" "$action" "$rsync_exit" "$remote_after" "$final_status" "$reason"
      continue
    fi

    if [[ "$remote_before" == "exists" && "$OVERWRITE" -eq 0 ]]; then
      action="skip_remote_already_exists"
      remote_after="exists"
      final_status="ok_remote_already_exists"
      log "[OK] Remote already has ${destination_path}"
      append_tsv "$SYNC_RESULT" "$destination_path" "${source_path:-}" "$local_status" "$remote_before" "$action" "$rsync_exit" "$remote_after" "$final_status" "$reason"
      continue
    fi

    if [[ "$local_status" != "exists" ]]; then
      action="cannot_sync_local_missing"
      remote_after="$remote_before"
      final_status="missing_on_3070_and_remote_${remote_before}"
      failures=$((failures + 1))
      log "[MISSING] ${reason}: no local source found for ${destination_path}"
      append_tsv "$SYNC_RESULT" "$destination_path" "" "$local_status" "$remote_before" "$action" "$rsync_exit" "$remote_after" "$final_status" "$reason"
      continue
    fi

    log "[SYNC] ${source_path} -> ${REMOTE}:${destination_path}"
    action="rsync"
    if ! remote_mkdir_parent "$destination_path"; then
      rsync_exit="remote_mkdir_failed"
      remote_after="$remote_before"
      final_status="sync_failed"
      failures=$((failures + 1))
      append_tsv "$SYNC_RESULT" "$destination_path" "$source_path" "$local_status" "$remote_before" "$action" "$rsync_exit" "$remote_after" "$final_status" "$reason"
      continue
    fi
    if rsync_one "$source_path" "$destination_path"; then
      rsync_exit="0"
    else
      rsync_exit="$?"
      failures=$((failures + 1))
    fi
    if remote_file_exists "$destination_path"; then
      remote_after="exists"
    else
      remote_after="missing"
      failures=$((failures + 1))
    fi
    if [[ "$rsync_exit" == "0" && "$remote_after" == "exists" ]]; then
      final_status="ok_synced"
    else
      final_status="sync_failed"
    fi
    append_tsv "$SYNC_RESULT" "$destination_path" "$source_path" "$local_status" "$remote_before" "$action" "$rsync_exit" "$remote_after" "$final_status" "$reason"
  done 3< "$SYNC_MANIFEST"

  return "$failures"
}

write_report() {
  local exit_status="$1"
  local missing_count ok_count dry_count
  missing_count="$(awk -F'\t' 'NR > 1 && $8 !~ /^ok/ && $8 != "dry_run" { c++ } END { print c + 0 }' "$SYNC_RESULT" 2>/dev/null || echo 0)"
  ok_count="$(awk -F'\t' 'NR > 1 && $8 ~ /^ok/ { c++ } END { print c + 0 }' "$SYNC_RESULT" 2>/dev/null || echo 0)"
  dry_count="$(awk -F'\t' 'NR > 1 && $8 == "dry_run" { c++ } END { print c + 0 }' "$SYNC_RESULT" 2>/dev/null || echo 0)"

  {
    echo "# PTH Sync Report"
    echo
    echo "## Summary"
    echo
    echo "- Time: \`$(date '+%F %T')\`"
    echo "- Run dir: \`${RUN_DIR}\`"
    echo "- Remote: \`${REMOTE:-not provided}\`"
    echo "- Dry run: \`${DRY_RUN}\`"
    echo "- Verify only: \`${VERIFY_ONLY}\`"
    echo "- Include extra structure pths: \`${INCLUDE_EXTRA_STRUCTURE}\`"
    echo "- OK rows: \`${ok_count}\`"
    echo "- Dry-run rows: \`${dry_count}\`"
    echo "- Problem rows: \`${missing_count}\`"
    echo "- Exit status: \`${exit_status}\`"
    echo
    echo "## Generated Lists"
    echo
    echo "- List 1, all stage pth dependencies: \`${LIST1}\`"
    echo "- List 2, checkpoint_path call sites: \`${LIST2}\`"
    echo "- List 3, 3070 pths from structure.md: \`${LIST3}\`"
    echo "- Sync manifest: \`${SYNC_MANIFEST}\`"
    echo "- Sync result: \`${SYNC_RESULT}\`"
    echo "- Full log: \`${MASTER_LOG}\`"
    echo
    echo "## How To Interpret"
    echo
    echo "- \`generated_on_4090\`: this pth is produced by an earlier stage on 4090; do not copy it from 3070."
    echo "- \`external_old_checkpoint\`: this pth is an old result used by \`eval-baselines\`; it must exist on 4090."
    echo "- \`ok_remote_already_exists\`: 4090 already has the file, so the stage can use it."
    echo "- \`ok_synced\`: the file was copied to 4090 and verified with \`ssh test -f\`."
    echo "- \`missing_remote\` or \`missing_on_3070...\`: the file is still not available on 4090."
    echo
    if [[ -f "$SYNC_RESULT" ]]; then
      echo "## Problem Rows"
      echo
      echo '```text'
      awk -F'\t' 'NR == 1 || (NR > 1 && $8 !~ /^ok/ && $8 != "dry_run") { print }' "$SYNC_RESULT"
      echo '```'
      echo
    fi
    echo "## Next Step"
    echo
    if [[ "$exit_status" -eq 0 ]]; then
      echo "The required old pths are available or the script was run in list/dry mode. On 4090, run:"
      echo
      echo '```bash'
      echo "cd /home/cq/code/AD-MT-revision/code_revised"
      echo "bash run_major_revision_experiments.sh eval-baselines --gpu 0"
      echo '```'
    else
      echo "Fix the problem rows before running \`eval-baselines\`. Usual fixes:"
      echo
      echo "- If local source is missing on 3070, search the old 4090 backup or rerun that baseline."
      echo "- If remote SSH failed, check \`--remote\`, password/key login, and network reachability."
      echo "- If rsync failed, rerun this script; rsync can resume partial transfers."
      echo "- If a path exists on 3070 under \`/results/...\`, rerun with the same command; the script already tries that fallback."
    fi
  } > "$REPORT_MD"
  log "Report written to: $REPORT_MD"
}

main() {
  log "Run dir: $RUN_DIR"
  log "Code revised dir: $CODE_REVISED_DIR"
  log "Baseline manifest: $BASELINE_MANIFEST"
  log "Structure md: $STRUCTURE_MD"
  log "Remote: ${REMOTE:-not provided}"
  log "Eval model types: $EVAL_MODEL_TYPES"

  build_stage_lists
  log "List 1 written: $LIST1"
  log "List 2 written: $LIST2"
  log "List 3 written: $LIST3"
  log "Sync manifest written: $SYNC_MANIFEST"

  local sync_status=0
  sync_or_verify || sync_status="$?"
  write_report "$sync_status"

  if [[ "$sync_status" -ne 0 ]]; then
    log "[ERROR] Some required pths are still missing or failed to sync. See: $REPORT_MD"
    exit "$sync_status"
  fi
  log "Done."
}

main "$@"
