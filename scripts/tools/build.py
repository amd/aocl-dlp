#!/usr/bin/env python3
"""
AOCL-DLP Docker Build Script

This script builds a Docker container for AOCL-DLP development environment.
It reads configuration from docker_config.yaml and processes Dockerfile.in template.
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
    is_real_docker = False

    # Check for docker first
    try:
        result = subprocess.run(['docker', '--version'],
                                capture_output=True, check=True, text=True)
        # Check if it's real Docker or Podman emulating Docker
        if 'podman' in result.stdout.lower():
            docker_cmd = 'podman'
            print("Using Podman (via docker command)")
        else:
            docker_cmd = 'docker'
            is_real_docker = True
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

    return docker_cmd, is_real_docker


def generate_dockerfile(config, dockerfile_template_path, dockerfile_output_path):
    """Generate Dockerfile from template using configuration."""
    try:
        with open(dockerfile_template_path, 'r') as f:
            template_content = f.read()
    except FileNotFoundError:
        print(
            f"Error: Dockerfile template {dockerfile_template_path} not found.")
        sys.exit(1)

    # Prepare substitution values
    base_image = f"{config['base_image']['name']}:{config['base_image']['tag']}"
    system_packages = ' \\\n    '.join(config['packages']['system'])

    substitutions = {
        'BASE_IMAGE': base_image,
        'SYSTEM_PACKAGES': system_packages,
        'SSH_PORT': str(config['ssh']['port']),
        'WORKSPACE_MOUNT': config['volumes']['workspace_mount']
    }

    # Perform substitutions
    dockerfile_content = template_content
    for key, value in substitutions.items():
        dockerfile_content = dockerfile_content.replace(f'{{{key}}}', value)

    # Write generated Dockerfile
    with open(dockerfile_output_path, 'w') as f:
        f.write(dockerfile_content)

    print(f"Generated Dockerfile at {dockerfile_output_path}")


def pull_base_image(docker_cmd, base_image):
    """Pull the base image to ensure we have the latest version."""
    print(f"Pulling base image: {base_image}")
    try:
        subprocess.run([docker_cmd, 'pull', base_image], check=True)
        print("Base image pulled successfully")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to pull base image: {e}")
        print("Continuing with existing image if available...")


def build_image(docker_cmd, config, dockerfile_path, context_path, is_real_docker=True):
    """Build the Docker image."""
    image_name = config['container']['name']

    build_cmd = [docker_cmd, 'build']

    # Add buildx support if requested and available (only for real Docker)
    if config['build']['use_buildx'] and is_real_docker and docker_cmd == 'docker':
        try:
            subprocess.run(['docker', 'buildx', 'version'],
                           capture_output=True, check=True)
            build_cmd = ['docker', 'buildx', 'build']

            # Add cache options
            if config['build']['cache_from']:
                build_cmd.extend(
                    ['--cache-from', config['build']['cache_from']])
            if config['build']['cache_to']:
                build_cmd.extend(['--cache-to', config['build']['cache_to']])

            # Load the image into docker images
            build_cmd.append('--load')

        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Warning: buildx not available, using regular build")
            build_cmd = [docker_cmd, 'build']
    elif config['build']['use_buildx'] and not is_real_docker:
        print("Warning: buildx not supported with Podman, using regular build")

    build_cmd.extend([
        '-t', image_name,
        '-f', str(dockerfile_path),
        str(context_path)
    ])

    print(f"Building image: {image_name}")
    print(f"Build command: {' '.join(build_cmd)}")

    try:
        subprocess.run(build_cmd, check=True)
        print(f"Successfully built image: {image_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error building image: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Build AOCL-DLP development container')
    parser.add_argument('--config', '-c',
                        default='docker_config.yaml',
                        help='Path to configuration YAML file (default: docker_config.yaml)')
    parser.add_argument('--dockerfile-template', '-t',
                        default='Dockerfile.in',
                        help='Path to Dockerfile template (default: Dockerfile.in)')
    parser.add_argument('--no-pull', action='store_true',
                        help='Skip pulling base image')

    args = parser.parse_args()

    # Get script directory
    script_dir = Path(__file__).parent.absolute()

    # Resolve paths
    config_path = script_dir / args.config
    dockerfile_template_path = script_dir / args.dockerfile_template
    dockerfile_output_path = script_dir / 'Dockerfile'

    print("AOCL-DLP Docker Build Script")
    print("=" * 40)

    # Load configuration
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    # Check Docker/Podman availability
    docker_cmd, is_real_docker = check_docker_available()

    # Generate Dockerfile
    generate_dockerfile(config, dockerfile_template_path,
                        dockerfile_output_path)

    # Pull base image if requested
    base_image = f"{config['base_image']['name']}:{config['base_image']['tag']}"
    if config['build']['pull_base_image'] and not args.no_pull:
        pull_base_image(docker_cmd, base_image)

    # Build image
    build_image(docker_cmd, config, dockerfile_output_path,
                script_dir, is_real_docker)

    print("\nBuild completed successfully!")
    print(f"Container image: {config['container']['name']}")
    print(f"Use './run.py --workspace=<path>' to start the container")


if __name__ == '__main__':
    main()
