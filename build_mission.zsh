#!/usr/bin/env zsh

set -e

# Initialize conda for this zsh shell
source /home/unitree/miniconda3/etc/profile.d/conda.sh

# Activate the required Python environment
conda activate changan

echo "========== Python environment =========="
which python
python --version
python -c "import sys; print('Python:', sys.executable)"
python -c "import scipy; print('SciPy:', scipy.__version__)"
python -c "import scipy; print('SciPy:', scipy.__file__)"
echo "========================================="

source /opt/ros/foxy/setup.zsh
source /home/unitree/code/driver_ws/install_foxy/setup.zsh
source /home/unitree/code/dual_arm_ws/install/setup.zsh
source /home/unitree/code/vision_ws/src/hangcha-perception/install/setup.zsh

_script_dir="${0:A}"
_src_root="${_script_dir:h}"
_ws_root="${_src_root:h}"

cd "${_ws_root}"

echo "[mission build] Workspace: ${_ws_root}"
echo "[mission build] Removing legacy build artifacts..."

rm -rf build install log

echo "[mission build] Building mission workspace..."

colcon build --symlink-install

unset _script_dir _src_root _ws_root

echo "[mission build] Build completed successfully."
