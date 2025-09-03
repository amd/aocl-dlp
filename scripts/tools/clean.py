#!/usr/bin/env python3
"""
AOCL-DLP Docker Clean Script

This script cleans up Docker images, containers, and build cache related to AOCL-DLP
development environment. It only removes resources created by this project.
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
        subprocess.run(['docker', '--version'], capture_output=True, check=True)
        docker_cmd = 'docker'
        print("Using Docker")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Check for podman if docker not found
    if not docker_cmd:
        try:
            subprocess.run(['podman', '--version'], capture_output=True, check=True)
            docker_cmd = 'podman'
            print("Using Podman")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    if not docker_cmd:
        print("Error: Neither Docker nor Podman is available.")
        sys.exit(1)

    return docker_cmd


def stop_and_remove_container(docker_cmd, container_name):
    """Stop and remove the AOCL-DLP container."""
    print(f"Stopping and removing container: {container_name}")

    try:
        # Check if container exists
        result = subprocess.run([docker_cmd, 'ps', '-a', '-q', '-f', f'name={container_name}'],
                              capture_output=True, text=True, check=True)

        if result.stdout.strip():
            # Stop container if running
            subprocess.run([docker_cmd, 'stop', container_name],
                         capture_output=True, check=False)

            # Remove container
            subprocess.run([docker_cmd, 'rm', container_name], check=True)
            print(f"Container '{container_name}' removed successfully")
        else:
            print(f"Container '{container_name}' not found")

    except subprocess.CalledProcessError as e:
        print(f"Error removing container: {e}")


def remove_image(docker_cmd, image_name):
    """Remove the AOCL-DLP Docker image."""
    print(f"Removing image: {image_name}")

    try:
        # Check if image exists
        result = subprocess.run([docker_cmd, 'images', '-q', image_name],
                              capture_output=True, text=True, check=True)

        if result.stdout.strip():
            subprocess.run([docker_cmd, 'rmi', image_name], check=True)
            print(f"Image '{image_name}' removed successfully")
        else:
            print(f"Image '{image_name}' not found")

    except subprocess.CalledProcessError as e:
        print(f"Error removing image: {e}")


def clean_build_cache(docker_cmd):
    """Clean Docker build cache (only for Docker, not Podman)."""
    if docker_cmd != 'docker':
        print("Build cache cleaning only supported with Docker")
        return

    print("Cleaning Docker build cache...")

    try:
        # Clean buildx cache if available
        try:
            subprocess.run(['docker', 'buildx', 'version'], capture_output=True, check=True)
            subprocess.run(['docker', 'buildx', 'prune', '-f'], check=True)
            print("Buildx cache cleaned")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

        # Clean general build cache
        subprocess.run([docker_cmd, 'builder', 'prune', '-f'], check=True)
        print("Build cache cleaned successfully")

    except subprocess.CalledProcessError as e:
        print(f"Error cleaning build cache: {e}")


def clean_local_cache():
    """Clean local build cache directory."""
    cache_dir = Path('/tmp/.buildx-cache')

    if cache_dir.exists():
        print(f"Removing local cache directory: {cache_dir}")
        try:
            import shutil
            shutil.rmtree(cache_dir)
            print("Local cache directory removed successfully")
        except Exception as e:
            print(f"Error removing local cache directory: {e}")
    else:
        print("Local cache directory not found")


def remove_generated_dockerfile(script_dir):
    """Remove generated Dockerfile."""
    dockerfile_path = script_dir / 'Dockerfile'

    if dockerfile_path.exists():
        print(f"Removing generated Dockerfile: {dockerfile_path}")
        try:
            dockerfile_path.unlink()
            print("Generated Dockerfile removed successfully")
        except Exception as e:
            print(f"Error removing generated Dockerfile: {e}")
    else:
        print("Generated Dockerfile not found")


def clean_dangling_images(docker_cmd):
    """Clean dangling images (optional)."""
    print("Cleaning dangling images...")

    try:
        subprocess.run([docker_cmd, 'image', 'prune', '-f'], check=True)
        print("Dangling images cleaned successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error cleaning dangling images: {e}")


def main():
    parser = argparse.ArgumentParser(description='Clean AOCL-DLP Docker resources')
    parser.add_argument('--config', '-c',
                       default='docker_config.yaml',
                       help='Path to configuration YAML file (default: docker_config.yaml)')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Clean everything including dangling images')
    parser.add_argument('--container-only', action='store_true',
                       help='Only remove container, keep image')
    parser.add_argument('--cache-only', action='store_true',
                       help='Only clean build cache')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be cleaned without actually doing it')

    args = parser.parse_args()

    # Get script directory
    script_dir = Path(__file__).parent.absolute()

    # Resolve paths
    config_path = script_dir / args.config

    print("AOCL-DLP Docker Clean Script")
    print("=" * 40)

    if args.dry_run:
        print("DRY RUN MODE - No actual changes will be made")
        print()

    # Load configuration
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    # Check Docker/Podman availability
    docker_cmd = check_docker_available()

    container_name = config['container']['name']
    image_name = container_name

    if args.dry_run:
        print("\nWould perform the following actions:")
        if not args.cache_only:
            print(f"- Stop and remove container: {container_name}")
            if not args.container_only:
                print(f"- Remove image: {image_name}")
                print(f"- Remove generated Dockerfile")
        if not args.container_only:
            print(f"- Clean build cache")
            print(f"- Clean local cache directory")
        if args.all:
            print(f"- Clean dangling images")
        return

    # Perform cleanup operations
    if not args.cache_only:
        # Remove container
        stop_and_remove_container(docker_cmd, container_name)

        if not args.container_only:
            # Remove image
            remove_image(docker_cmd, image_name)

            # Remove generated Dockerfile
            remove_generated_dockerfile(script_dir)

    if not args.container_only:
        # Clean build cache
        clean_build_cache(docker_cmd)

        # Clean local cache
        clean_local_cache()

    # Clean dangling images if requested
    if args.all:
        clean_dangling_images(docker_cmd)

    print("\nCleanup completed!")
    print("\nNote: Workspace directories are not cleaned automatically.")
    print("Please clean your workspace directories manually if needed.")


if __name__ == '__main__':
    main()
