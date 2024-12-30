#!/bin/bash

# Function to activate a conda environment and run a script as a specific user
run_in_env() {
    local env_name=$1
    local script=$2
    local role=$3
    local user=$4

    echo "Starting $role in conda environment: $env_name as user: $user"
    echo "545545" | sudo -S -u $user bash -c "
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate $env_name
        if [ $? -ne 0 ]; then
            echo \"Error: Failed to activate conda environment $env_name\"
            exit 1
        fi

        python $script

        if [ $? -ne 0 ]; then
            echo \"Error: $role encountered an error while running.\"
            exit 1
        else
            echo \"$role completed successfully.\"
        fi
    "
}

# Define the environments, scripts, and corresponding users
declare -A configs
configs=(
    ["ultralytics_christmas,frank"]="/home/frank/Frank/code/ultralytics_christmas/robot_control/SocketServer.py"
    ["Whisper,yuwenjia"]="/home/yuwenjia/code/server_script.py"
)

# Run the scripts in the respective environments under the specified users
for key in "${!configs[@]}"; do
    IFS=',' read -r env user <<< "$key"
    script=${configs[$key]}
    role=$(basename $script | cut -d'_' -f1) # Extract role from script name
    run_in_env $env $script $role $user &
done

# Wait for all background processes to finish
wait

echo "All processes completed."
