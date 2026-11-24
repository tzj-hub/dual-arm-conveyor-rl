#!/usr/bin/env bash
# Opt-in workaround for a reproduced native glibc cos/sin FMA-path SIGFPE.
# This changes IFUNC selection at process startup, not robot dynamics or math
# error handling. Preserve all user tunables, including existing CPU exclusions.
case "${CONVEYOR_LIBM_COMPAT:-0}" in
  0) ;;
  1)
    IFS=: read -r -a CONVEYOR_GLIBC_PARTS <<< "${GLIBC_TUNABLES:-}"
    CONVEYOR_GLIBC_OPTIONS=()
    CONVEYOR_FOUND_HWCAPS=0
    for CONVEYOR_GLIBC_PART in "${CONVEYOR_GLIBC_PARTS[@]}"; do
      [[ -z "$CONVEYOR_GLIBC_PART" ]] && continue
      if [[ "$CONVEYOR_GLIBC_PART" == glibc.cpu.hwcaps=* ]]; then
        CONVEYOR_GLIBC_PART="${CONVEYOR_GLIBC_PART},-FMA"
        CONVEYOR_FOUND_HWCAPS=1
      fi
      CONVEYOR_GLIBC_OPTIONS+=("$CONVEYOR_GLIBC_PART")
    done
    if [[ "$CONVEYOR_FOUND_HWCAPS" == 0 ]]; then
      CONVEYOR_GLIBC_OPTIONS+=("glibc.cpu.hwcaps=-FMA")
    fi
    printf -v GLIBC_TUNABLES '%s:' "${CONVEYOR_GLIBC_OPTIONS[@]}"
    export GLIBC_TUNABLES="${GLIBC_TUNABLES%:}"
    ;;
  *)
    echo "CONVEYOR_LIBM_COMPAT must be 0 or 1" >&2
    return 2
    ;;
esac
