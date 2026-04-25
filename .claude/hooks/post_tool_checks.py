#!/usr/bin/env python3
"""PostToolUse hook: Silent floor detection + gold-plating + route-around detection."""
import json, os, sys, re, subprocess

input_data = json.load(sys.stdin)
file_path = input_data.get("tool_input", {}).get("file_path", "")
if not file_path or not os.path.isfile(file_path):
    sys.exit(0)

warnings = []
blocker_path = os.path.join(os.environ.get("CLAUDE_PROJECT_DIR", "."), "CRITICAL_BLOCKER.md")

# --- Silent floor detection (only src/dpf/ Python files) ---
if file_path.endswith(".py") and "src/dpf/" in file_path:
    try:
        with open(file_path) as f:
            for i, line in enumerate(f, 1):
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
                    continue
                if "telemetry" in stripped.lower() or "# no-floor-check" in stripped:
                    continue
                if re.search(r'np\.(maximum|clip|fmax)', stripped):
                    if re.search(r'(rho|pressure|density|_p\b|_P\b|emf|velocity|_v\b|_B\b|floor)', stripped, re.IGNORECASE):
                        warnings.append(f"  Line {i}: Possible silent floor: {stripped[:80]}")
    except Exception:
        pass
    if warnings:
        warnings.insert(0, f"WARNING: Possible silent numerical floors in {file_path}:")
        warnings.append("Use telemetry.apply_floor() or add '# no-floor-check' if intentional.")

# --- Anti-gold-plating ---
DIAG_PATTERN = re.compile(r'(telemetry|extract_|pirt|diagnostic|postprocess|scaffold|helper|util)', re.IGNORECASE)
if file_path.endswith(".py") and DIAG_PATTERN.search(file_path):
    try:
        with open(file_path) as f:
            line_count = sum(1 for _ in f)
        if line_count > 50:
            warnings.append(f"WARNING: {os.path.basename(file_path)} is {line_count} lines. Keep under 50 LOC.")
    except Exception:
        pass

# --- Route-around detection (Edit expansion during blocker) ---
if os.path.isfile(blocker_path) and os.path.getsize(blocker_path) > 0:
    tool_name = input_data.get("tool_name", "")
    if tool_name == "Edit" and file_path.endswith(".py"):
        try:
            result = subprocess.run(["git", "diff", "--stat", file_path], capture_output=True, text=True, timeout=5)
            if "insertion" in result.stdout:
                match = re.search(r'(\d+) insertion', result.stdout)
                if match and int(match.group(1)) > 50:
                    warnings.append(
                        f"WARNING: {file_path} grew by {match.group(1)} lines during active CRITICAL_BLOCKER. "
                        f"Possible route-around behavior."
                    )
        except Exception:
            pass

if warnings:
    output = {"additionalContext": "\n".join(warnings)}
    json.dump(output, sys.stdout)

sys.exit(0)
