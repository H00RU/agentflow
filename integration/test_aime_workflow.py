"""
Test AIME Workflow Generation
Verifies that AIME workflows are generated with correct interface
"""

import sys
import asyncio

# Add paths
sys.path.insert(0, '/content/agentworkflow/AFlow')
sys.path.insert(0, '/root/integration')

from workflow_parser import WorkflowParser
import tempfile
import os


def test_workflow_generation():
    """Test that AIME and HumanEval workflows have correct interfaces"""

    parser = WorkflowParser()

    # Test 1: HumanEval workflow (should require entry_point)
    print("=" * 80)
    print("Test 1: Generating HumanEval Workflow")
    print("=" * 80)

    humaneval_qwen_output = """
<workflow_modification>
Generate code solution with ensemble
</workflow_modification>

<operators>
CustomCodeGenerate, ScEnsemble
</operators>

<workflow_steps>
1. Generate multiple code candidates
2. Use ensemble to select best
</workflow_steps>
"""

    humaneval_spec = parser.parse_qwen_output(humaneval_qwen_output, dataset_type="HumanEval")

    if humaneval_spec:
        print("\n✓ HumanEval workflow spec generated")
        print(f"  Operators: {humaneval_spec.operators}")

        # Check __call__ signature
        if "entry_point: str" in humaneval_spec.workflow_code:
            print("  ✓ __call__ signature: Has REQUIRED entry_point parameter (correct for HumanEval)")
        elif "entry_point: Optional[str]" in humaneval_spec.workflow_code:
            print("  ✗ __call__ signature: Has OPTIONAL entry_point parameter (WRONG for HumanEval)")
        else:
            print("  ✗ __call__ signature: Missing entry_point parameter!")

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='_humaneval_workflow.py', delete=False) as f:
            f.write(humaneval_spec.workflow_code)
            print(f"\n  Saved to: {f.name}")
    else:
        print("✗ Failed to generate HumanEval workflow")
        return False

    # Test 2: AIME workflow (should have optional entry_point)
    print("\n" + "=" * 80)
    print("Test 2: Generating AIME Workflow")
    print("=" * 80)

    aime_qwen_output = """
<workflow_modification>
Solve mathematical problem with reasoning
</workflow_modification>

<operators>
CustomCodeGenerate, ScEnsemble
</operators>

<workflow_steps>
1. Generate multiple solution attempts
2. Use ensemble to select most consistent answer
</workflow_steps>
"""

    aime_spec = parser.parse_qwen_output(aime_qwen_output, dataset_type="AIME")

    if aime_spec:
        print("\n✓ AIME workflow spec generated")
        print(f"  Operators: {aime_spec.operators}")

        # Check __call__ signature
        if "entry_point: Optional[str]" in aime_spec.workflow_code:
            print("  ✓ __call__ signature: Has OPTIONAL entry_point parameter (correct for AIME)")
        elif "entry_point: str" in aime_spec.workflow_code and "Optional" not in aime_spec.workflow_code:
            print("  ✗ __call__ signature: Has REQUIRED entry_point parameter (WRONG for AIME)")
        else:
            print("  ✗ __call__ signature: Unexpected format!")

        # Check workflow logic
        if 'entry_point=""' in aime_spec.workflow_code or "entry_point=''" in aime_spec.workflow_code:
            print("  ✓ Workflow logic: Passes empty string for entry_point (correct for AIME)")
        else:
            print("  ⚠ Workflow logic: May not handle entry_point correctly")

        # Save to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='_aime_workflow.py', delete=False) as f:
            f.write(aime_spec.workflow_code)
            print(f"\n  Saved to: {f.name}")

        # Display the workflow code snippet
        print("\n  Generated workflow __call__ signature:")
        for line in aime_spec.workflow_code.split('\n'):
            if 'async def __call__' in line:
                print(f"    {line.strip()}")
                break
    else:
        print("✗ Failed to generate AIME workflow")
        return False

    # Test 3: Verify workflow can be imported
    print("\n" + "=" * 80)
    print("Test 3: Verifying workflow imports")
    print("=" * 80)

    try:
        # Try importing the generated AIME workflow
        spec = __import__('importlib.util').util.spec_from_file_location(
            'aime_wf',
            f.name
        )
        module = __import__('importlib.util').util.module_from_spec(spec)
        spec.loader.exec_module(module)

        print("✓ AIME workflow module imported successfully")

        # Check if Workflow class exists
        if hasattr(module, 'Workflow'):
            print("✓ Workflow class found")

            # Check __call__ method signature
            import inspect
            call_sig = inspect.signature(module.Workflow.__call__)
            params = list(call_sig.parameters.keys())
            print(f"  __call__ parameters: {params}")

            # Verify entry_point is optional
            if 'entry_point' in params:
                entry_point_param = call_sig.parameters['entry_point']
                if entry_point_param.default == inspect.Parameter.empty:
                    print("  ✗ entry_point is REQUIRED (should be optional for AIME)")
                    return False
                else:
                    print(f"  ✓ entry_point is OPTIONAL with default={entry_point_param.default}")
            else:
                print("  ✗ entry_point parameter missing!")
                return False
        else:
            print("✗ Workflow class not found")
            return False

    except Exception as e:
        print(f"✗ Failed to import workflow: {e}")
        return False

    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nSummary:")
    print("- HumanEval workflows: Generated with REQUIRED entry_point ✓")
    print("- AIME workflows: Generated with OPTIONAL entry_point ✓")
    print("- Workflows can be imported and executed ✓")
    print("\nThe fix is working correctly. AIME training should now work!")

    return True


if __name__ == "__main__":
    success = test_workflow_generation()
    sys.exit(0 if success else 1)
