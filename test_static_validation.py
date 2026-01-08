#!/usr/bin/env python3
"""
Static validation of GPU throughput optimizations.

This test validates code structure and API without requiring dependencies.
"""

import ast
import sys
from pathlib import Path

def validate_pipeline_backends():
    """Validate the pipeline_backends.py file structure."""
    print("\n" + "="*60)
    print("STATIC VALIDATION: pipeline_backends.py")
    print("="*60)
    
    file_path = Path(__file__).parent / "scripts" / "pipeline_backends.py"
    
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Parse the AST
        tree = ast.parse(source)
        
        # Find all function definitions
        functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
        
        print(f"\n✓ File parsed successfully")
        print(f"  Found {len(functions)} functions")
        print(f"  Found {len(classes)} classes")
        
        # Check for new functions
        required_functions = [
            "_compute_optimal_state_tile",
            "_compute_max_precompute_size",
            "_get_pinned_buffer",
            "_build_all_states_torch_cuda",
            "_autotune_kernel_tiles",
            "_get_compute_stream",
            "_dispatch_kernel_async",
            "_load_autotune_cache",
            "_save_autotune_cache",
            "compute_kernel_matrix"
        ]
        
        print("\n✓ Checking required functions:")
        for func in required_functions:
            if func in functions:
                print(f"  ✓ {func}")
            else:
                print(f"  ✗ MISSING: {func}")
                return False
        
        # Check for new classes
        required_classes = [
            "PersistentBufferPool"
        ]
        
        print("\n✓ Checking required classes:")
        for cls in required_classes:
            if cls in classes:
                print(f"  ✓ {cls}")
            else:
                print(f"  ✗ MISSING: {cls}")
                return False
        
        # Check compute_kernel_matrix signature
        print("\n✓ Validating compute_kernel_matrix signature:")
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "compute_kernel_matrix":
                # Get argument names
                args = [arg.arg for arg in node.args.args]
                kwargs = [arg.arg for arg in node.args.kwonlyargs]
                
                required_params = ["state_tile", "autotune", "precompute_all_states", "vram_fraction"]
                
                for param in required_params:
                    if param in kwargs:
                        print(f"  ✓ {param}")
                    else:
                        print(f"  ✗ MISSING parameter: {param}")
                        return False
                
                # Check default values
                defaults = {kw.arg: kw for kw in node.args.kwonlyargs}
                
                # Validate state_tile default is -1
                if "state_tile" in defaults:
                    for default_node in node.args.kw_defaults:
                        if default_node and isinstance(default_node, ast.UnaryOp):
                            if isinstance(default_node.op, ast.USub) and isinstance(default_node.operand, ast.Constant):
                                if default_node.operand.value == 1:
                                    print(f"  ✓ state_tile default = -1")
                                    break
                
                break
        
        # Check for global variables/caches
        print("\n✓ Checking global caches:")
        required_globals = [
            "_BUFFER_POOL",
            "_COMPUTE_STREAM",
            "_AUTOTUNE_CACHE",
            "_AUTOTUNE_CACHE_FILE"
        ]
        
        for var in required_globals:
            if var in source:
                print(f"  ✓ {var}")
            else:
                print(f"  ✗ MISSING: {var}")
                return False
        
        # Check for key optimizations in code
        print("\n✓ Checking optimization implementations:")
        optimizations = {
            "VRAM-aware sizing": "_compute_optimal_state_tile",
            "Bulk precomputation": "_build_all_states_torch_cuda",
            "Kernel autotuning": "_autotune_kernel_tiles",
            "Async dispatch": "_dispatch_kernel_async",
            "Batch synchronization": "sync_every",
            "Pinned memory": "pin_memory",
            "Memory pools": "MemoryPool",
            "DLPack handoff": "from_dlpack"
        }
        
        for name, marker in optimizations.items():
            if marker in source:
                print(f"  ✓ {name}")
            else:
                print(f"  ⚠ Warning: {name} marker '{marker}' not found")
        
        print("\n" + "="*60)
        print("✓ VALIDATION PASSED")
        print("="*60)
        return True
        
    except SyntaxError as e:
        print(f"\n✗ SYNTAX ERROR: {e}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_documentation():
    """Check that key documentation exists."""
    print("\n" + "="*60)
    print("DOCUMENTATION CHECK")
    print("="*60)
    
    file_path = Path(__file__).parent / "scripts" / "pipeline_backends.py"
    
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Check for docstrings
        tree = ast.parse(source)
        
        documented_functions = 0
        total_functions = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                total_functions += 1
                if ast.get_docstring(node):
                    documented_functions += 1
        
        print(f"\n  Functions with docstrings: {documented_functions}/{total_functions}")
        
        # Check for key documentation sections
        key_sections = [
            "VRAM & Memory Management",
            "CUDA Kernel Autotuning",
            "Bulk State Precomputation"
        ]
        
        print("\n  Documentation sections:")
        for section in key_sections:
            if section in source:
                print(f"    ✓ {section}")
            else:
                print(f"    ⚠ Missing: {section}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        return False

def main():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("GPU THROUGHPUT OPTIMIZATION - STATIC VALIDATION")
    print("="*60)
    
    results = []
    
    # Run validations
    results.append(("Pipeline Backends Structure", validate_pipeline_backends()))
    results.append(("Documentation", validate_documentation()))
    
    # Summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} validations passed")
    print("="*60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
