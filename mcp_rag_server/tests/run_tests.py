#!/usr/bin/env python3
"""
Test runner for MCP RAG server end-to-end tests.

This script helps run the test suite with proper setup and validation.
It checks dependencies and provides helpful error messages.
"""

import subprocess
import sys
from pathlib import Path


def check_dependency(package_name, import_name=None):
    """Check if a Python package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70 + "\n")


def check_dependencies():
    """Check if all required dependencies are installed."""
    print_header("Checking Dependencies")
    
    dependencies = {
        "pytest": "pytest",
        "pytest-asyncio": "pytest_asyncio",
        "mcp": "mcp",
    }
    
    optional_dependencies = {
        "langchain": "langchain",
        "langchain-core": "langchain_core",
    }
    
    missing_required = []
    missing_optional = []
    
    print("Required dependencies:")
    for package, import_name in dependencies.items():
        if check_dependency(package, import_name):
            print(f"  ✓ {package}")
        else:
            print(f"  ✗ {package} (MISSING)")
            missing_required.append(package)
    
    print("\nOptional dependencies:")
    for package, import_name in optional_dependencies.items():
        if check_dependency(package, import_name):
            print(f"  ✓ {package}")
        else:
            print(f"  ○ {package} (optional, not installed)")
            missing_optional.append(package)
    
    return missing_required, missing_optional


def install_dependencies(packages):
    """Install missing dependencies."""
    if not packages:
        return True
    
    print(f"\nAttempting to install: {', '.join(packages)}")
    print("This may take a few minutes...\n")
    
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install"] + packages,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("✓ Installation successful!")
        return True
    except subprocess.CalledProcessError:
        print("✗ Installation failed. Please install manually:")
        print(f"  pip install {' '.join(packages)}")
        return False


def run_tests(test_file=None, verbose=False):
    """Run the test suite."""
    print_header("Running Tests")
    
    # Determine which tests to run
    test_dir = Path(__file__).parent
    
    if test_file:
        test_path = test_dir / test_file
        if not test_path.exists():
            print(f"Error: Test file not found: {test_file}")
            return False
    else:
        test_path = test_dir / "test_e2e_file_operations.py"
    
    # Build pytest command
    cmd = [sys.executable, "-m", "pytest", str(test_path)]
    
    if verbose:
        cmd.append("-v")
    
    cmd.extend(["-s", "--tb=short"])
    
    print(f"Running: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd)
        return result.returncode == 0
    except FileNotFoundError:
        print("Error: pytest not found. Please install it:")
        print("  pip install pytest pytest-asyncio")
        return False


def run_standalone_demo():
    """Run standalone demo without pytest."""
    print_header("Running Standalone Demo")
    
    test_dir = Path(__file__).parent
    demo_file = test_dir / "test_e2e_file_operations.py"
    
    if not demo_file.exists():
        print(f"Error: Demo file not found: {demo_file}")
        return False
    
    print(f"Running: python {demo_file}\n")
    
    result = subprocess.run([sys.executable, str(demo_file)])
    return result.returncode == 0


def run_langchain_demo():
    """Run LangChain integration demo."""
    print_header("Running LangChain Integration Demo")
    
    test_dir = Path(__file__).parent
    demo_file = test_dir / "test_langchain_integration.py"
    
    if not demo_file.exists():
        print(f"Error: Demo file not found: {demo_file}")
        return False
    
    print(f"Running: python {demo_file}\n")
    
    result = subprocess.run([sys.executable, str(demo_file)])
    return result.returncode == 0


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run MCP RAG server end-to-end tests"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check dependencies only, don't run tests"
    )
    
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install missing dependencies automatically"
    )
    
    parser.add_argument(
        "--test-file",
        type=str,
        help="Specific test file to run"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run standalone demo (no pytest required)"
    )
    
    parser.add_argument(
        "--langchain-demo",
        action="store_true",
        help="Run LangChain integration demo"
    )
    
    args = parser.parse_args()
    
    print_header("MCP RAG Server - Test Runner")
    
    # Check dependencies
    missing_required, missing_optional = check_dependencies()
    
    if args.check_deps:
        if missing_required:
            print("\n⚠ Missing required dependencies. Install with:")
            print(f"  pip install {' '.join(missing_required)}")
            return 1
        else:
            print("\n✓ All required dependencies are installed!")
            return 0
    
    # Install dependencies if requested
    if args.install_deps and missing_required:
        if not install_dependencies(missing_required):
            return 1
    
    # Run demo mode
    if args.demo:
        success = run_standalone_demo()
        return 0 if success else 1
    
    # Run LangChain demo
    if args.langchain_demo:
        success = run_langchain_demo()
        return 0 if success else 1
    
    # Check if we can run tests
    if missing_required:
        print("\n⚠ Cannot run tests without required dependencies.")
        print("Install them with:")
        print(f"  pip install {' '.join(missing_required)}")
        print("\nOr run standalone demo:")
        print("  python run_tests.py --demo")
        return 1
    
    # Run tests
    success = run_tests(args.test_file, args.verbose)
    
    if success:
        print_header("✓ All Tests Passed!")
        return 0
    else:
        print_header("✗ Some Tests Failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
