#!/usr/bin/env bash
# Compatibility launcher; each layer runs one continuous DragBox Action.
set -eo pipefail
source /rm_nvme/recordings/code/setup_mission_env.sh
exec ros2 run mission_controller calibrate_drag_bigbox_all_layers "$@"
