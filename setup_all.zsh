#!/usr/bin/env zsh

# Source this file so the environment changes remain in the current shell:
#   source src/setup_all.zsh

_mission_setup_script="${${(%):-%N}:A}"
_mission_repo_root="${_mission_setup_script:h}"
_mission_ws_root="${_mission_repo_root:h}"
_changan_root="${_mission_ws_root:h}"
_moveit_ws_root="${MOVEIT2_WS:-${_changan_root}/../../libraries/ws_moveit2}"

_mission_source_setup() {
  local setup_file="$1"
  local label="$2"
  if [[ ! -r "${setup_file}" ]]; then
    print -u2 -- "[mission setup] missing ${label}: ${setup_file}"
    return 1
  fi
  source "${setup_file}"
}

if [[ -n "${ROS_SETUP:-}" ]]; then
  _mission_source_setup "${ROS_SETUP}" "ROS 2" || return 1
fi
if [[ -r "${_moveit_ws_root}/install/setup.zsh" ]]; then
  _mission_source_setup "${_moveit_ws_root}/install/setup.zsh" \
    "ws_moveit2" || return 1
else
  print -u2 -- "[mission setup] ws_moveit2 not found; set MOVEIT2_WS if needed"
fi

_mission_source_setup "${_changan_root}/dual_arm_ws/install/setup.zsh" \
  "dual_arm_ws" || return 1
if [[ -r "${_mission_ws_root}/install/setup.zsh" ]]; then
  _mission_source_setup "${_mission_ws_root}/install/setup.zsh" \
    "mission_ws" || return 1
else
  print -u2 -- "[mission setup] mission_ws is not built yet; continuing"
fi
_mission_source_setup "${_changan_root}/vision_ws/install/setup.zsh" \
  "vision_ws" || return 1

print -- "[mission setup] sourced ws_moveit2, dual_arm_ws, mission_ws, vision_ws"

unset _mission_setup_script _mission_repo_root _mission_ws_root _changan_root _moveit_ws_root
unfunction _mission_source_setup
