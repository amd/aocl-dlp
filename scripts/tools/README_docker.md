# AOCL-DLP Docker Development Environment

This directory contains scripts and configuration for creating a unified Docker-based development environment for AOCL-DLP. The environment provides a consistent setup across different host systems with all necessary development tools pre-installed.

## Overview

The Docker tool consists of:
- **Configuration**: `docker_config.yaml` - Centralized configuration for all Docker settings
- **Template**: `Dockerfile.in` - Dockerfile template that gets processed with user-specific values
- **Build Script**: `build.py` - Builds the Docker image with proper user configuration
- **Run Script**: `run.py` - Runs containers with workspace mounting and user setup
- **Clean Script**: `clean.py` - Cleans up Docker resources created by this tool

## Quick Start

1. **Build the development image**:
   ```bash
   cd scripts/tools
   ./build.py
   ```

2. **Run a development container**:
   ```bash
   ./run.py --workspace=$HOME/aocl-dlp-workspace
   ```

3. **Clean up when done**:
   ```bash
   ./clean.py
   ```

## Configuration

All settings are centralized in `docker_config.yaml`. Key configurable items include:

### Base Image
```yaml
base_image:
  name: "ubuntu"
  tag: "25.04"
```

### Packages
```yaml
packages:
  system:
    - lcov
    - gcc
    - gcov
    - cmake
    - ninja-build
    # ... more packages
```

### Container Settings
```yaml
container:
  name: "aocl-dlp-dev"
  privileged: true

network:
  ssh_port: 8022
```

## Detailed Usage

### Building the Image

```bash
./build.py [OPTIONS]
```

**Options:**
- `--config, -c`: Path to configuration file (default: `docker_config.yaml`)
- `--dockerfile-template, -t`: Path to Dockerfile template (default: `Dockerfile.in`)
- `--no-pull`: Skip pulling the base image

**What it does:**
1. Reads configuration from YAML file
2. Gets current user information (UID, GID, username)
3. Processes `Dockerfile.in` template with configuration values
4. Pulls the latest base image (unless `--no-pull` is specified)
5. Builds the Docker image with proper user setup

### Running Containers

```bash
./run.py --workspace=<path> [OPTIONS]
```

**Required Arguments:**
- `--workspace, -w`: Path to workspace directory to mount in container

**Options:**
- `--config, -c`: Path to configuration file (default: `docker_config.yaml`)
- `--ssh-service, -s`: Run container as SSH service in background

**What it does:**
1. Creates workspace directory if it doesn't exist
2. Sets proper ownership on workspace directory
3. Removes existing container (if configured to do so)
4. Mounts workspace to `/workspace` in container
5. Forwards SSH keys (if available)
6. Starts container with proper user context

**Interactive Mode (default):**
Drops you into a bash shell inside the container.

**SSH Service Mode:**
```bash
./run.py --workspace=$HOME/workspace --ssh-service
```
Runs container in background with SSH server. Connect via:
```bash
ssh -p 8022 username@localhost
```

### Cleaning Up

```bash
./clean.py [OPTIONS]
```

**Options:**
- `--config, -c`: Path to configuration file
- `--all, -a`: Clean everything including dangling images
- `--container-only`: Only remove container, keep image
- `--cache-only`: Only clean build cache
- `--dry-run`: Show what would be cleaned without doing it

**What it cleans:**
- AOCL-DLP container (stops and removes)
- AOCL-DLP Docker image
- Docker build cache
- Generated Dockerfile
- Local buildx cache directory

## Features

### User Management
- Creates container user with same UID/GID as host user
- Preserves username and shell preferences
- Enables passwordless sudo access
- Forwards SSH keys for git operations

### Development Tools
Pre-installed packages include:
- **Compilers**: GCC, Clang
- **Build Tools**: CMake, Ninja
- **Testing**: lcov, gcov
- **Debugging**: GDB, Valgrind
- **Utilities**: Git, Vim, htop
- **Libraries**: OpenMP development files

### Workspace Management
- Automatic workspace directory creation
- Proper ownership management
- Persistent data across container restarts
- Working directory set to `/workspace`

### SSH Support
- SSH server for remote development
- Key forwarding from host
- Configurable port mapping
- Secure configuration (no root login, key-based auth only)

## Docker vs Podman Support

The scripts automatically detect and work with both Docker and Podman:
- Docker is preferred if both are available
- Buildx support for enhanced caching (Docker only)
- Equivalent functionality with Podman

## Advanced Usage

### Custom Configuration

Create a custom configuration file:
```bash
cp docker_config.yaml my_config.yaml
# Edit my_config.yaml as needed
./build.py --config my_config.yaml
./run.py --config my_config.yaml --workspace=/my/workspace
```

### Multiple Environments

You can maintain multiple configurations for different purposes:
```bash
# Build with different base images
./build.py --config ubuntu_22_04_config.yaml
./build.py --config fedora_config.yaml

# Run with specific configurations
./run.py --config ubuntu_22_04_config.yaml --workspace=/ubuntu/workspace
```

### Development Workflow

1. **Initial Setup**:
   ```bash
   ./build.py
   ./run.py --workspace=$HOME/aocl-dlp-dev
   ```

2. **Daily Development**:
   ```bash
   # Container persists your workspace
   ./run.py --workspace=$HOME/aocl-dlp-dev
   # Work on code, run tests, etc.
   exit
   ```

3. **SSH Development**:
   ```bash
   # Start SSH service
   ./run.py --workspace=$HOME/aocl-dlp-dev --ssh-service

   # Connect from another terminal or IDE
   ssh -p 8022 $(whoami)@localhost

   # Stop when done
   ./clean.py --container-only
   ```

4. **Cleanup**:
   ```bash
   # Remove everything
   ./clean.py --all

   # Or just the container
   ./clean.py --container-only
   ```

## Troubleshooting

### Permission Issues
If you encounter permission issues with the workspace:
```bash
# Fix ownership manually
sudo chown -R $(id -u):$(id -g) $HOME/aocl-dlp-workspace
```

### Image Build Failures
```bash
# Clean and rebuild
./clean.py
./build.py --no-pull  # Skip base image pull if network issues
```

### Container Won't Start
```bash
# Check if image exists
docker images | grep aocl-dlp-dev

# Rebuild if missing
./build.py

# Check for conflicting containers
docker ps -a | grep aocl-dlp-dev
./clean.py --container-only
```

### SSH Connection Issues
```bash
# Check if container is running
docker ps | grep aocl-dlp-dev

# Check SSH service
docker exec aocl-dlp-dev systemctl status ssh

# Verify port mapping
docker port aocl-dlp-dev
```

## Security Considerations

- Container runs with your user privileges (not root)
- SSH server configured for key-based authentication only
- Privileged mode enabled for development tools access
- Workspace directory isolated from host system

## Customization

### Adding Packages
Edit `docker_config.yaml`:
```yaml
packages:
  system:
    - your-package-here
```

### Changing Base Image
```yaml
base_image:
  name: "ubuntu"
  tag: "22.04"  # or "fedora", "debian", etc.
```

### Custom SSH Port
```yaml
network:
  ssh_port: 2222  # Change from default 8022
```

## Contributing

When modifying the Docker environment:
1. Update `docker_config.yaml` for configuration changes
2. Modify `Dockerfile.in` for image changes
3. Update scripts for functionality changes
4. Test with both Docker and Podman
5. Update this README for user-facing changes
