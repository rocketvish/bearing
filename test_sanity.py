"""Quick sanity test for prompt caching and reasoning effort."""

import os
import shutil

from agent import run_agent


def main():
    project_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("Sanity Test: Prompt Caching + Reasoning")
    print("=" * 60)

    result = run_agent(
        task_prompt=(
            "Create a file called _sanity_test/test.txt containing 'hello world', "
            "then read it back to verify the contents are correct."
        ),
        project_dir=project_dir,
        max_turns=5,
        use_caching=True,
        reasoning_effort="high",
    )

    print(f"\n{'=' * 60}")
    print("Sanity Test Results")
    print(f"{'=' * 60}")
    print(f"Status: {result['status']}")
    print(f"Turns: {result['turns_used']}")
    print(f"Total input: {result['total_input_tokens']:,}")
    print(f"Total output: {result['total_output_tokens']:,}")
    print(f"Total cache read: {result['total_cache_read_tokens']:,}")
    print(f"Total cache creation: {result['total_cache_creation_tokens']:,}")
    print(f"Total reasoning: {result['total_thinking_tokens']:,}")
    print(f"Cost: ${result['cost_usd']:.4f}")

    print("\nPer-turn breakdown:")
    for i in range(result["turns_used"]):
        inp = result["per_turn_input_tokens"][i]
        out = result["per_turn_output_tokens"][i]
        cr = result["per_turn_cache_read_tokens"][i]
        cc = result["per_turn_cache_creation_tokens"][i]
        th = result["per_turn_thinking_tokens"][i]
        print(
            f"  Turn {i + 1}: in={inp:,} (cached={cr:,}, created={cc:,}) "
            f"out={out:,} reasoning={th:,}"
        )

    # Verify cache reads on turn 2+
    found_cache_read = False
    for i, cr in enumerate(result["per_turn_cache_read_tokens"]):
        if i > 0 and cr > 0:
            print(f"\nCache read detected on turn {i + 1}: {cr:,} tokens")
            found_cache_read = True
            break
    if not found_cache_read and result["turns_used"] > 1:
        print("\nWARNING: No cache reads detected on turn 2+")

    # Cleanup
    test_dir = os.path.join(project_dir, "_sanity_test")
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir, ignore_errors=True)
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
