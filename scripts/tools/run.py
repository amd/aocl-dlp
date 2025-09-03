#!/usr/bin/env python3
"""
AOCL-DLP Docker Run Script

This script runs the AOCL-DLP development container with proper workspace mounting
and user configuration.
"""

import os
import sys
import yaml
import subprocess
import argparse
from pathlib import Path


def load_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file {config_path} not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        sys.exit(1)


def check_docker_available():
    """Check if Docker or Podman is available."""
    docker_cmd = None

    # Check for docker first
    try:
        subprocess.run(['docker', '--version'],
                       capture_output=True, check=True)
        docker_cmd = 'docker'
        print("Using Docker")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Check for podman if docker not found
    if not docker_cmd:
        try:
            subprocess.run(['podman', '--version'],
                           capture_output=True, check=True)
            docker_cmd = 'podman'
            print("Using Podman")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    if not docker_cmd:
        print("Error: Neither Docker nor Podman is available.")
        sys.exit(1)

    return docker_cmd


def check_image_exists(docker_cmd, image_name):
    """Check if the Docker image exists."""
    try:
        result = subprocess.run([docker_cmd, 'images', '-q', image_name],
                                capture_output=True, text=True, check=True)
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError:
        return False


def remove_existing_container(docker_cmd, container_name):
    """Remove existing container if it exists."""
    try:
        # Check if container exists (running or stopped)
        result = subprocess.run([docker_cmd, 'ps', '-a', '-q', '-f', f'name={container_name}'],
                                capture_output=True, text=True, check=True)

        if result.stdout.strip():
            print(f"Removing existing container: {container_name}")
            subprocess.run(
                [docker_cmd, 'rm', '-f', container_name], check=True)
            print("Existing container removed")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to remove existing container: {e}")


def setup_workspace_permissions(workspace_path):
    """Ensure workspace directory exists."""
    workspace_path = Path(workspace_path).resolve()

    # Create workspace directory if it doesn't exist
    workspace_path.mkdir(parents=True, exist_ok=True)
    print(f"Workspace directory ready: {workspace_path}")

    return str(workspace_path)


def setup_ssh_keys():
    """Setup SSH keys for the container."""
    ssh_dir = Path.home() / '.ssh'

    if not ssh_dir.exists():
        print("No SSH directory found. SSH key forwarding will not be available.")
        return None

    # Look for common SSH key files
    key_files = ['id_rsa', 'id_ed25519', 'id_ecdsa']
    available_keys = []

    for key_file in key_files:
        private_key = ssh_dir / key_file
        public_key = ssh_dir / f'{key_file}.pub'

        if private_key.exists() and public_key.exists():
            available_keys.append(str(ssh_dir))
            break

    return str(ssh_dir) if available_keys else None


def run_container(docker_cmd, config, workspace_path, ssh_service=False):
    """Run the Docker container with proper configuration."""
    container_name = config['container']['name']
    image_name = container_name

    # Check if image exists
    if not check_image_exists(docker_cmd, image_name):
        print(f"Error: Image '{image_name}' not found.")
        print("Please run './build.py' first to build the image.")
        sys.exit(1)

    # Remove existing container if auto-remove is enabled
    if config['runtime']['auto_remove_existing']:
        remove_existing_container(docker_cmd, container_name)

    # Setup workspace permissions
    workspace_path = setup_workspace_permissions(workspace_path)

    # Build run command
    run_cmd = [docker_cmd, 'run']

    # Container options
    run_cmd.extend([
        '--name', container_name,
        '--hostname', container_name,
        '-it',  # Interactive with TTY
        # Remove on exit unless running SSH service
        '--rm' if not ssh_service else '--detach',
    ])

    # Privileged mode if requested
    if config['container']['privileged']:
        run_cmd.append('--privileged')

    # Mount workspace
    workspace_mount = config['volumes']['workspace_mount']
    run_cmd.extend(['-v', f'{workspace_path}:{workspace_mount}'])

    # Setup SSH key forwarding if available
    ssh_dir = setup_ssh_keys()
    if ssh_dir:
        run_cmd.extend(['-v', f'{ssh_dir}:/root/.ssh:ro'])

    # Port mapping for SSH if running as service
    if ssh_service:
        ssh_port = config['network']['ssh_port']
        run_cmd.extend(['-p', f'{ssh_port}:22'])

    # Environment variables
    for env_var in config['runtime']['environment_vars']:
        run_cmd.extend(['-e', env_var])

    # Working directory
    run_cmd.extend(['-w', workspace_mount])

    # Image name
    run_cmd.append(image_name)

    # Command to run
    if ssh_service:
        run_cmd.extend(['/usr/sbin/sshd', '-D'])
    else:
        run_cmd.append('/bin/bash')

    print(f"Starting container: {container_name}")
    print(f"Workspace: {workspace_path} -> {workspace_mount}")
    if ssh_service:
        print(
            f"SSH service will be available on port {config['network']['ssh_port']}")
    print(f"Run command: {' '.join(run_cmd)}")
    print()

    try:
        if ssh_service:
            subprocess.run(run_cmd, check=True)
            print(f"Container '{container_name}' is running in the background")
            print(
                f"Connect via SSH: ssh -p {config['network']['ssh_port']} root@localhost")
        else:
            print("Entering container shell as root...")
            subprocess.run(run_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running container: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nContainer execution interrupted")


def main():
    parser = argparse.ArgumentParser(
        description='Run AOCL-DLP development container')
    parser.add_argument('--workspace', '-w', required=True,
                        help='Path to workspace directory to mount in container')
    parser.add_argument('--config', '-c',
                        default='docker_config.yaml',
                        help='Path to configuration YAML file (default: docker_config.yaml)')
    parser.add_argument('--ssh-service', '-s', action='store_true',
                        help='Run container as SSH service in background')

    args = parser.parse_args()

    # Get script directory
    script_dir = Path(__file__).parent.absolute()

    # Resolve paths
    config_path = script_dir / args.config
    workspace_path = Path(args.workspace).expanduser().resolve()

    print("AOCL-DLP Docker Run Script")
    print("=" * 40)

    # Load configuration
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    # Check Docker/Podman availability
    docker_cmd = check_docker_available()

    # Validate workspace path
    if not workspace_path.exists():
        print(f"Creating workspace directory: {workspace_path}")
        workspace_path.mkdir(parents=True, exist_ok=True)

    # Run container
    run_container(docker_cmd, config, str(workspace_path), args.ssh_service)


if __name__ == '__main__':
    main()
