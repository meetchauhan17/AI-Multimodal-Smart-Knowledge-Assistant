"""
verify_llm.py — Script to verify the LLMProvider fallback chain in action.
"""

import sys
from core.logger import logger
from core.llm_provider import LLMProvider


def main():
    print("Initializing LLMProvider...")
    try:
        provider = LLMProvider()
    except Exception as exc:
        print(f"Failed to initialize LLMProvider: {exc}", file=sys.stderr)
        sys.exit(1)

    prompt = "State in exactly one sentence: What is the capital of France?"
    print(f"\nSending query: {prompt!r}")

    try:
        response = provider.generate(prompt)
        print("\n=== LLM Response ===")
        print(f"Answer:            {response.text.strip()}")
        print(f"Provider Used:     {response.provider_used}")
        print(f"Latency:           {response.latency_ms:.2f}ms")
        print(f"Fallback Triggered: {response.fallback_triggered}")
        print("====================")
    except Exception as exc:
        print(f"\nError: Query failed. All providers in the chain failed.", file=sys.stderr)
        print(f"Details: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
