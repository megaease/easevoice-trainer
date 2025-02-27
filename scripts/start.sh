#!/bin/bash

echo "Starting trainer..."

uv run src/main.py &

echo "Starting Jupyter Lab..."

jupyter lab --allow-root --no-browser --port=8888 --ip=0.0.0.0 --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' --ServerApp.token="$JUPYTER_TOKEN" --ServerApp.allow_origin=* --ServerApp.preferred_dir="/root"  --app-dir="/usr/local/share/jupyter/lab"
echo "Jupyter Lab Started"
