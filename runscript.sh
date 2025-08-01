#!/bin/bash

# Array of configuration values
vals=(
  "f,c,n,h,p = False,False,False,False,False"
  "f,c,n,h,p = True,True,False,False,False"
  "f,c,n,h,p = True,True,True,True,False"
  "f,c,n,h,p = True,True,True,True,True"
)

# Print current directory
pwd

# Loop over each config
for val in "${vals[@]}"
do
  # Write the config line to src/fcnhp.py
  echo "$val" > src/fcnhp.py

  # Clean up Docker artifacts
  docker rmi $(docker images -q) -f
  docker volume prune -f
  docker system prune -a --volumes -f

  # Rebuild and run the Docker container
  docker build -t llm-benchmarking -f docker/Dockerfile .
  docker run --rm -v "$(pwd)/output:/app/output" -v "$(pwd)/logs:/app/logs" -v "$(pwd)/src:/app/src" llm-benchmarking
done

