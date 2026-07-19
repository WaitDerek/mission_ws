#!/usr/bin/env zsh

# Source this file so the environment changes remain in the current shell:
#   source /path/to/mission_ws/src/setup_all.zsh

_mission_setup_script="${${(%):-%N}:A}"
_mission_repo_root="${_mission_setup_script:h}"
_mission_ws_root="${_mission_repo_root:h}"
_changan_root="${_mission_ws_root:h}"

_mission_source_setup() {
  local setup_file="$1"
  local label="$2"
  if [[ ! -r "${setup_file}" ]]; then
    print -u2 -- "[mission setup] missing ${label}: ${setup_file}"
    return 1
  fi
  source "${setup_file}"
}

_mission_source_setup "${_changan_root}/dual_arm_ws/install/setup.zsh" \
  "dual_arm_ws" || return 1
_mission_source_setup "${_changan_root}/grasp_ws/install/setup.zsh" \
  "grasp_ws" || return 1
_mission_source_setup "${_changan_root}/vision_ws/install/setup.zsh" \
  "vision_ws" || return 1
_mission_source_setup "${_mission_ws_root}/install/setup.zsh" \
  "mission_ws" || return 1

export CHANGAN_ROOT="${_changan_root}"
export MISSION_WS_ROOT="${_mission_ws_root}"
export MISSION_CONDA_ROOT="${MISSION_CONDA_ROOT:-/home/dekc/anaconda3}"
export MISSION_GRASP_CONDA_ENV="${MISSION_GRASP_CONDA_ENV:-changan}"

if [[ -z "${MISSION_VISION_CONDA_ENV:-}" ]]; then
  if [[ -x "${MISSION_CONDA_ROOT}/envs/foundationpose/bin/python" ]]; then
    export MISSION_VISION_CONDA_ENV="foundationpose"
  elif [[ -x "${MISSION_CONDA_ROOT}/envs/foundationpose310/bin/python" ]]; then
    # Compatibility with the currently installed environment name.
    export MISSION_VISION_CONDA_ENV="foundationpose310"
  else
    export MISSION_VISION_CONDA_ENV="foundationpose"
  fi
fi

print -- "[mission setup] sourced dual_arm_ws, grasp_ws, vision_ws, mission_ws"
print -- "[mission setup] grasp=${MISSION_GRASP_CONDA_ENV}, vision=${MISSION_VISION_CONDA_ENV}"

unset _mission_setup_script _mission_repo_root _mission_ws_root _changan_root
unfunction _mission_source_setup
