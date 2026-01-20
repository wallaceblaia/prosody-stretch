#!/usr/bin/env python3
"""
Build and release script for prosody-stretch.

Usage:
    python release.py patch    # 0.1.0 -> 0.1.1 (bug fixes)
    python release.py minor    # 0.1.0 -> 0.2.0 (new features)  
    python release.py major    # 0.1.0 -> 1.0.0 (breaking changes)
    python release.py build    # Build without version bump
    python release.py publish  # Publish to PyPI
"""

import re
import sys
import subprocess
from pathlib import Path
from datetime import datetime


# Project paths
PROJECT_ROOT = Path(__file__).parent
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
INIT_FILE = PROJECT_ROOT / "prosody_stretch" / "__init__.py"
CLI_FILE = PROJECT_ROOT / "prosody_stretch" / "cli.py"
CHANGELOG = PROJECT_ROOT / "CHANGELOG.md"


def get_current_version() -> str:
    """Read current version from pyproject.toml."""
    content = PYPROJECT.read_text()
    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if match:
        return match.group(1)
    raise ValueError("Version not found in pyproject.toml")


def parse_version(version: str) -> tuple:
    """Parse version string to tuple."""
    parts = version.split(".")
    return tuple(int(p) for p in parts)


def bump_version(version: str, bump_type: str) -> str:
    """Bump version according to semantic versioning."""
    major, minor, patch = parse_version(version)
    
    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        raise ValueError(f"Unknown bump type: {bump_type}")


def update_version_in_file(filepath: Path, old_version: str, new_version: str):
    """Update version string in a file."""
    if not filepath.exists():
        return False
    
    content = filepath.read_text()
    
    # Handle different version patterns
    patterns = [
        (f'version = "{old_version}"', f'version = "{new_version}"'),
        (f'version="{old_version}"', f'version="{new_version}"'),
        (f"__version__ = '{old_version}'", f"__version__ = '{new_version}'"),
        (f'__version__ = "{old_version}"', f'__version__ = "{new_version}"'),
        (f"version='{old_version}'", f"version='{new_version}'"),
    ]
    
    updated = False
    for old, new in patterns:
        if old in content:
            content = content.replace(old, new)
            updated = True
    
    if updated:
        filepath.write_text(content)
        print(f"  Updated: {filepath.name}")
    
    return updated


def get_changelog_entry() -> str:
    """Prompt for changelog entry."""
    print("\nWhat changed in this version? (Enter empty line to finish)")
    print("Use prefixes: [Added], [Fixed], [Changed], [Removed]")
    print("-" * 50)
    
    lines = []
    while True:
        try:
            line = input("  > ")
            if not line:
                break
            lines.append(f"- {line}")
        except EOFError:
            break
    
    return "\n".join(lines)


def update_changelog(version: str, changes: str):
    """Add new version entry to CHANGELOG.md."""
    date = datetime.now().strftime("%Y-%m-%d")
    
    new_entry = f"""
## [{version}] - {date}

{changes}
"""
    
    if CHANGELOG.exists():
        content = CHANGELOG.read_text()
        # Insert after header
        if "# Changelog" in content:
            parts = content.split("# Changelog", 1)
            content = parts[0] + "# Changelog\n" + new_entry + parts[1].lstrip("\n")
        else:
            content = new_entry + "\n" + content
    else:
        content = f"""# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).
{new_entry}"""
    
    CHANGELOG.write_text(content)
    print(f"  Updated: CHANGELOG.md")


def build_package():
    """Build the package using build module."""
    print("\n📦 Building package...")
    
    # Clean previous builds
    dist_dir = PROJECT_ROOT / "dist"
    if dist_dir.exists():
        for f in dist_dir.iterdir():
            f.unlink()
    
    # Build
    result = subprocess.run(
        [sys.executable, "-m", "build"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Build failed:\n{result.stderr}")
        return False
    
    # List built files
    if dist_dir.exists():
        print("✅ Built packages:")
        for f in dist_dir.iterdir():
            size = f.stat().st_size / 1024
            print(f"   {f.name} ({size:.1f} KB)")
    
    return True


def publish_package(test: bool = False):
    """Publish package to PyPI."""
    print("\n🚀 Publishing package...")
    
    if test:
        repo = ["--repository", "testpypi"]
        print("   (Publishing to TestPyPI)")
    else:
        repo = []
        print("   (Publishing to PyPI)")
    
    result = subprocess.run(
        [sys.executable, "-m", "twine", "upload", *repo, "dist/*"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Publish failed:\n{result.stderr}")
        return False
    
    print("✅ Published successfully!")
    return True


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    current_version = get_current_version()
    print(f"📋 Current version: {current_version}")
    
    if command in ("major", "minor", "patch"):
        # Bump version
        new_version = bump_version(current_version, command)
        print(f"📈 New version: {new_version}")
        
        # Update files
        print("\n🔄 Updating version in files...")
        update_version_in_file(PYPROJECT, current_version, new_version)
        update_version_in_file(INIT_FILE, current_version, new_version)
        update_version_in_file(CLI_FILE, current_version, new_version)
        
        # Get changelog entry
        changes = get_changelog_entry()
        if changes:
            update_changelog(new_version, changes)
        
        # Build
        if build_package():
            print(f"\n✅ Version {new_version} ready for distribution!")
            print(f"\nTo publish:")
            print(f"  python build.py publish      # Publish to PyPI")
            print(f"  python build.py publish-test # Publish to TestPyPI")
        
    elif command == "build":
        # Build without version bump
        build_package()
        
    elif command == "publish":
        result = input("Publish to PyPI? (y/N): ")
        if result.lower() == "y":
            publish_package(test=False)
        
    elif command == "publish-test":
        publish_package(test=True)
        
    elif command == "version":
        print(current_version)
        
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
