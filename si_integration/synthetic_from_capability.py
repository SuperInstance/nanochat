#!/usr/bin/env python3
"""Generate synthetic Q&A conversations from SuperInstance CAPABILITY.toml files.

Reads CAPABILITY.toml from each crate directory, produces JSONL conversations
suitable for nanochat's CustomJSON fine-tuning task.

Usage:
    python si_integration/synthetic_from_capability.py \
        --crates-dir ../crates \
        --output data/si_synthetic.jsonl \
        --conversations-per-crate 50
"""

import json
import os
import sys
from pathlib import Path

# Minimal TOML parser (stdlib only, Python 3.11+ has tomllib but this works on 3.10)
def parse_toml(text):
    """Parse a subset of TOML sufficient for CAPABILITY.toml files."""
    result = {}
    current_section = result
    current_key = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            # Section header
            section_name = line.strip("[]").strip()
            parts = section_name.split(".")
            current_section = result
            for part in parts:
                if part not in current_section:
                    current_section[part] = {}
                current_section = current_section[part]
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Strip inline comments
            if "#" in value and not value.startswith('"'):
                value = value.split("#")[0].strip()
            # Parse value types
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("[") and value.endswith("]"):
                # Simple array of strings
                inner = value[1:-1].strip()
                if inner:
                    items = [s.strip().strip('"').strip("'") for s in inner.split(",")]
                    value = [item for item in items if item]
                else:
                    value = []
            elif value in ("true", "false"):
                value = value == "true"
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # keep as string
            current_section[key] = value
    return result


def load_capability(crate_dir):
    """Load CAPABILITY.toml from a crate directory."""
    cap_path = Path(crate_dir) / "CAPABILITY.toml"
    if not cap_path.exists():
        return None
    return parse_toml(cap_path.read_text(encoding="utf-8"))


def generate_conversations(cap, crate_name, n=50):
    """Generate synthetic Q&A conversations about a crate from its CAPABILITY.toml."""
    crate_info = cap.get("crate", {})
    description = crate_info.get("description", f"the {crate_name} crate")
    layer = crate_info.get("layer", "unknown")
    capabilities = cap.get("provides", {}).get("capabilities", [])
    integrates = cap.get("integrates", {})

    conversations = []
    templates = [
        _qa_what_does_it_do,
        _qa_how_to_use,
        _qa_capabilities,
        _qa_integration,
        _qa_layer,
        _qa_deep_dive,
    ]

    for i in range(n):
        template = templates[i % len(templates)]
        conv = template(crate_name, description, layer, capabilities, integrates)
        if conv:
            conversations.append(conv)

    return conversations


def _make_conv(user_msg, assistant_msg):
    """Format a conversation in nanochat's CustomJSON format."""
    return {
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def _qa_what_does_it_do(name, desc, layer, caps, integrates):
    return _make_conv(
        f"What does the {name} crate do in the SuperInstance ecosystem?",
        f"The {name} crate {desc}. It operates at the {layer} layer "
        f"of the SuperInstance architecture, providing core functionality "
        f"for the broader system."
    )


def _qa_how_to_use(name, desc, layer, caps, integrates):
    return _make_conv(
        f"How do I use {name}?",
        f"To use {name}, import it in your Rust project by adding it as a dependency. "
        f"{name} {desc}. Check the crate's documentation for specific API entry points "
        f"and configuration options."
    )


def _qa_capabilities(name, desc, layer, caps, integrates):
    cap_list = ", ".join(caps) if caps else "core functionality"
    return _make_conv(
        f"What capabilities does {name} provide?",
        f"{name} provides the following capabilities: {cap_list}. "
        f"These capabilities are exposed through the crate's public API "
        f"and can be integrated with other SuperInstance components."
    )


def _qa_integration(name, desc, layer, caps, integrates):
    if not integrates:
        partners = "other SuperInstance crates through standard interfaces"
    else:
        partners = ", ".join(integrates.keys())
    return _make_conv(
        f"What does {name} integrate with?",
        f"{name} integrates with {partners}. "
        f"The integration follows SuperInstance's modular architecture, "
        f"allowing each component to operate independently while composing "
        f"effectively with others."
    )


def _qa_layer(name, desc, layer, caps, integrates):
    return _make_conv(
        f"At which layer does {name} operate?",
        f"{name} operates at the {layer} layer. "
        f"In the SuperInstance architecture, the {layer} layer "
        f"{'provides foundational abstractions used by higher layers' if layer == 'core' else 'builds on core primitives to deliver user-facing functionality' if layer == 'application' else 'connects core and application layers'}."
    )


def _qa_deep_dive(name, desc, layer, caps, integrates):
    return _make_conv(
        f"Explain the design philosophy behind {name}.",
        f"{name} follows the SuperInstance design principle of minimal, "
        f"composable components. {desc}. "
        f"The crate is designed to do one thing well — {caps[0] if caps else 'its core function'} — "
        f"and compose cleanly with the rest of the ecosystem "
        f"rather than attempting to be a monolithic solution."
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic Q&A from SuperInstance CAPABILITY.toml files"
    )
    parser.add_argument(
        "--crates-dir", required=True, help="Path to the crates/ directory"
    )
    parser.add_argument(
        "--output", required=True, help="Output JSONL file path"
    )
    parser.add_argument(
        "--conversations-per-crate",
        type=int,
        default=50,
        help="Number of conversations to generate per crate (default: 50)",
    )
    args = parser.parse_args()

    crates_dir = Path(args.crates_dir)
    if not crates_dir.is_dir():
        print(f"Error: {args.crates_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(output_path, "w") as f:
        for crate_dir in sorted(crates_dir.iterdir()):
            if not crate_dir.is_dir():
                continue
            cap = load_capability(crate_dir)
            if cap is None:
                print(f"  skipping {crate_dir.name} (no CAPABILITY.toml)")
                continue

            conversations = generate_conversations(
                cap, crate_dir.name, args.conversations_per_crate
            )
            for conv in conversations:
                f.write(json.dumps(conv) + "\n")
                total += 1
            print(f"  {crate_dir.name}: {len(conversations)} conversations")

    print(f"\nGenerated {total} conversations → {output_path}")


if __name__ == "__main__":
    main()
