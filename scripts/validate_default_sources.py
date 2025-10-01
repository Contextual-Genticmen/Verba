#!/usr/bin/env python3
"""
Validation script for default_sources.yaml configuration
Tests that the YAML is properly structured and can be loaded by VerbaManager
"""

import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path to import verba modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from goldenverba.server.types import FileConfig, FileStatus


def validate_yaml_structure(yaml_path: Path) -> tuple[bool, str]:
    """Validate basic YAML structure"""
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if not config:
            return False, "YAML file is empty"
        
        if 'sources' not in config:
            return False, "Missing 'sources' key in YAML"
        
        if not isinstance(config['sources'], list):
            return False, "'sources' must be a list"
        
        if len(config['sources']) == 0:
            return False, "'sources' list is empty"
        
        return True, f"Found {len(config['sources'])} source(s) in configuration"
    
    except yaml.YAMLError as e:
        return False, f"YAML parsing error: {str(e)}"
    except Exception as e:
        return False, f"Error reading file: {str(e)}"


def validate_source_schema(source: Dict[str, Any], index: int) -> List[str]:
    """Validate individual source configuration"""
    errors = []
    
    # Required fields
    required_fields = ['fileID', 'filename', 'isURL', 'extension', 'source', 'labels']
    for field in required_fields:
        if field not in source:
            errors.append(f"Source {index}: Missing required field '{field}'")
    
    # Validate types
    if 'fileID' in source and not isinstance(source['fileID'], str):
        errors.append(f"Source {index}: 'fileID' must be a string")
    
    if 'filename' in source and not isinstance(source['filename'], str):
        errors.append(f"Source {index}: 'filename' must be a string")
    
    if 'isURL' in source and not isinstance(source['isURL'], bool):
        errors.append(f"Source {index}: 'isURL' must be a boolean")
    
    if 'labels' in source and not isinstance(source['labels'], list):
        errors.append(f"Source {index}: 'labels' must be a list")
    
    # Validate RAG config structure
    if 'rag_config' in source:
        rag_config = source['rag_config']
        if 'Reader' not in rag_config:
            errors.append(f"Source {index}: 'rag_config' must contain 'Reader' configuration")
        else:
            reader = rag_config['Reader']
            if 'selected' not in reader:
                errors.append(f"Source {index}: Reader config missing 'selected' field")
            if 'components' not in reader:
                errors.append(f"Source {index}: Reader config missing 'components' field")
            else:
                # Validate that selected reader exists in components
                selected = reader.get('selected', '')
                if selected and selected not in reader['components']:
                    errors.append(f"Source {index}: Selected reader '{selected}' not found in components")
    
    return errors


def validate_git_reader_config(source: Dict[str, Any], index: int) -> List[str]:
    """Validate Git reader specific configuration"""
    errors = []
    
    if 'rag_config' not in source or 'Reader' not in source['rag_config']:
        return errors
    
    reader = source['rag_config']['Reader']
    if reader.get('selected') != 'Git':
        return errors
    
    git_config = reader.get('components', {}).get('Git', {}).get('config', {})
    
    required_git_fields = ['Platform', 'Owner', 'Name', 'Branch']
    for field in required_git_fields:
        if field not in git_config:
            errors.append(f"Source {index}: Git reader missing '{field}' configuration")
        else:
            field_config = git_config[field]
            if 'value' not in field_config:
                errors.append(f"Source {index}: Git reader '{field}' missing 'value'")
    
    # Validate platform
    if 'Platform' in git_config:
        platform = git_config['Platform'].get('value', '')
        if platform not in ['GitHub', 'GitLab']:
            errors.append(f"Source {index}: Invalid platform '{platform}'. Must be 'GitHub' or 'GitLab'")
    
    # Validate owner and name are not empty
    if 'Owner' in git_config and not git_config['Owner'].get('value'):
        errors.append(f"Source {index}: Git 'Owner' cannot be empty")
    
    if 'Name' in git_config and not git_config['Name'].get('value'):
        errors.append(f"Source {index}: Git 'Name' cannot be empty")
    
    return errors


def test_fileconfig_creation(source: Dict[str, Any], index: int) -> tuple[bool, str]:
    """Test that source can be converted to FileConfig"""
    try:
        file_config = FileConfig(
            fileID=source.get('fileID', f"test-{index}"),
            filename=source['filename'],
            isURL=source.get('isURL', True),
            overwrite=source.get('overwrite', False),
            extension=source.get('extension', 'URL'),
            source=source.get('source', ''),
            content=source.get('content', ''),
            labels=source.get('labels', ['Document']),
            rag_config=source.get('rag_config', {}),
            file_size=source.get('file_size', 0),
            status=FileStatus.READY,
            metadata=source.get('metadata', ''),
            status_report={}
        )
        return True, f"Source {index} successfully converted to FileConfig"
    except Exception as e:
        return False, f"Source {index} FileConfig creation failed: {str(e)}"


def main():
    """Main validation function"""
    print("=" * 70)
    print("Default Sources Configuration Validation")
    print("=" * 70)
    
    # Locate YAML file
    yaml_path = Path(__file__).parent.parent / "goldenverba" / "default_sources.yaml"
    print(f"\nValidating: {yaml_path}")
    
    if not yaml_path.exists():
        print(f"❌ ERROR: File not found at {yaml_path}")
        return 1
    
    # Validate YAML structure
    print("\n1. Validating YAML structure...")
    is_valid, message = validate_yaml_structure(yaml_path)
    if not is_valid:
        print(f"❌ FAILED: {message}")
        return 1
    print(f"✅ PASSED: {message}")
    
    # Load configuration
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate each source
    print("\n2. Validating source configurations...")
    all_errors = []
    
    for i, source in enumerate(config['sources']):
        print(f"\n   Source {i + 1}: {source.get('filename', 'Unknown')}")
        
        # Schema validation
        schema_errors = validate_source_schema(source, i + 1)
        if schema_errors:
            all_errors.extend(schema_errors)
            for error in schema_errors:
                print(f"   ❌ {error}")
        else:
            print(f"   ✅ Schema validation passed")
        
        # Git reader specific validation
        git_errors = validate_git_reader_config(source, i + 1)
        if git_errors:
            all_errors.extend(git_errors)
            for error in git_errors:
                print(f"   ❌ {error}")
        elif source.get('rag_config', {}).get('Reader', {}).get('selected') == 'Git':
            print(f"   ✅ Git reader configuration valid")
        
        # FileConfig creation test
        can_create, create_msg = test_fileconfig_creation(source, i + 1)
        if can_create:
            print(f"   ✅ {create_msg}")
        else:
            all_errors.append(create_msg)
            print(f"   ❌ {create_msg}")
    
    # Summary
    print("\n" + "=" * 70)
    if all_errors:
        print(f"❌ VALIDATION FAILED with {len(all_errors)} error(s):")
        for error in all_errors:
            print(f"   - {error}")
        return 1
    else:
        print("✅ ALL VALIDATIONS PASSED")
        print(f"\nConfiguration contains {len(config['sources'])} valid source(s):")
        for i, source in enumerate(config['sources']):
            print(f"   {i + 1}. {source.get('filename')} ({source.get('source', 'N/A')})")
        return 0


if __name__ == "__main__":
    sys.exit(main())
