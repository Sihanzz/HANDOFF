#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../.."
source /opt/ros/jazzy/setup.bash
ROS_PYTHON_PATHS=$(python3 -c "import sys; print(':'.join([p for p in sys.path if 'ros' in p]))")
export PYTHONPATH="${PYTHONPATH:-}:${ROS_PYTHON_PATHS}:$(pwd):$(pwd)/src"
exec uv run --python 3.12 python deploy/controller/prompt_node.py
