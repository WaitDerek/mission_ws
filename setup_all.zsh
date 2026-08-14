#!/usr/bin/env zsh

# Source this file so the environment changes remain in the current shell:
#   source ./setup_all.zsh

# The Git repository lives directly under the workspace's src directory.
_mission_setup_script="${${(%):-%N}:A}"
_mission_repo_root="${_mission_setup_script:h}"
_mission_ws_root="${_mission_repo_root:h}"
_code_root="${_mission_ws_root:h}"
_home_root="${_code_root:h}"

_mission_source_setup() {
  local setup_file="$1"
  local label="$2"
  if [[ ! -r "${setup_file}" ]]; then
    print -u2 -- "[mission setup] missing ${label}: ${setup_file}"
    return 1
  fi
  source "${setup_file}"
}

_mission_ros_setup="${ROS_SETUP_FILE:-}"
if [[ -z "${_mission_ros_setup}" && -n "${ROS_PREFIX:-}" ]]; then
  _mission_ros_setup="${ROS_PREFIX}/setup.zsh"
fi
if [[ -z "${_mission_ros_setup}" && ${commands[ros2]+_} ]]; then
  _mission_ros_setup="$(cd -- "${commands[ros2]:h}/.." && pwd -P)/setup.zsh"
fi
if [[ -z "${_mission_ros_setup}" ]]; then
  for _mission_candidate in /opt/ros/*/setup.zsh(N); do
    _mission_ros_setup="${_mission_candidate}"
    break
  done
fi
if [[ -z "${_mission_ros_setup}" ]]; then
  print -u2 -- "[mission setup] set ROS_SETUP_FILE to the ROS 2 setup.zsh path"
  return 1
fi
_mission_source_setup "${_mission_ros_setup}" \
  "ROS 2" || return 1
_mission_source_setup "${_home_root}/workspace/rm_robot_ws/install/setup.zsh" \
  "rm_robot_ws" || return 1
_mission_source_setup "${_code_root}/dual_arm_ws/install/setup.zsh" \
  "dual_arm_ws" || return 1
_mission_source_setup "${_code_root}/vision_ws/install/setup.zsh" \
  "vision_ws" || return 1
_mission_source_setup "${_mission_ws_root}/install/setup.zsh" \
  "mission_ws" || return 1

print -- "[mission setup] sourced ROS 2, rm_robot_ws, dual_arm_ws, vision_ws, mission_ws"

unset _mission_setup_script _mission_repo_root _mission_ws_root _code_root _home_root \
  _mission_ros_setup _mission_candidate
unfunction _mission_source_setup
