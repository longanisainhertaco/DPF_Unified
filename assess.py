import os
import ast
import json

base_dir = "/Users/anthonyzamora/dpf-unified"
kr_dir = os.path.join(base_dir, "KnowledgeReference")
src_dir = os.path.join(base_dir, "src", "dpf")
root_dir = base_dir

kr_files = []
if os.path.exists(kr_dir):
    for root, dirs, files in os.walk(kr_dir):
        for f in files:
            if not f.startswith('.'):
                kr_files.append(f)

physics_keywords = ["plasma", "physics", "mhd", "voltage", "current", "inductance", "capacitance", "runge_kutta", "simulate", "solver", "equation", "circuit", "lee_model", "snowplow", "anode", "cathode"]

stats = {
    "kr_count": len(kr_files),
    "modules": {},
    "wired": [],
    "unwired": [],
    "todos": []
}

def analyze_file(filepath):
    rel_path = os.path.relpath(filepath, base_dir)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return

    # Find TODOs
    for i, line in enumerate(content.split('\n')):
        if 'TODO' in line or 'FIXME' in line:
            stats['todos'].append(f"- **{rel_path}:{i+1}** `{line.strip()}`")

    try:
        tree = ast.parse(content)
    except Exception:
        return

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            name = node.name.lower()
            docstring = ast.get_docstring(node) or ""
            
            is_physics = any(k in name for k in physics_keywords) or any(k in docstring.lower() for k in physics_keywords)
            
            if is_physics:
                # heuristic for wiring: mentions KnowledgeReference or references a file from it
                has_citation = "KnowledgeReference" in docstring or "citation" in docstring.lower()
                if not has_citation:
                    # check if any specific KR file is mentioned
                    has_citation = any(kr_f in docstring for kr_f in kr_files if len(kr_f) > 8)
                
                comp_str = f"- `{node.name}` in `{rel_path}`"
                if has_citation:
                    stats['wired'].append(comp_str)
                else:
                    stats['unwired'].append(comp_str)

for root, dirs, files in os.walk(src_dir):
    for f in files:
        if f.endswith(".py"):
            analyze_file(os.path.join(root, f))
            rel_dir = os.path.relpath(root, base_dir)
            if rel_dir not in stats['modules']:
                stats['modules'][rel_dir] = []
            stats['modules'][rel_dir].append(f)

for f in os.listdir(root_dir):
    if f.startswith("app") and f.endswith(".py"):
        analyze_file(os.path.join(root_dir, f))
        if "root" not in stats['modules']:
            stats['modules']["root"] = []
        stats['modules']["root"].append(f)

md = []
md.append("# Gemini Assessment: DPF-Unified Codebase Architecture & Wiring Report")
md.append("\n## 1. Executive Summary")
md.append(f"A comprehensive analysis was conducted on `/Users/anthonyzamora/dpf-unified`. The project contains {stats['kr_count']} files in `KnowledgeReference` acting as the Single Source of Truth.")
md.append(f"We analyzed `{len(stats['modules'])}` directories containing Python logic. Found {len(stats['wired'])} correctly wired physics components and {len(stats['unwired'])} physics components that require citation or formal wiring to the source of truth.")

md.append("\n## 2. Architecture Map")
for mod, files in stats['modules'].items():
    md.append(f"### `{mod}`")
    md.append(f"- **Files:** {', '.join(files[:10])}{'...' if len(files) > 10 else ''}")

md.append("\n## 3. Wired Physics Components (Compliant)")
if stats['wired']:
    md.extend(stats['wired'][:30])
    if len(stats['wired']) > 30: md.append(f"- *...and {len(stats['wired']) - 30} more.*")
else:
    md.append("- *No fully wired components citing `KnowledgeReference` found.*")

md.append("\n## 4. Unwired Physics Components (Action Required)")
md.append("The following components handle physics or simulations but lack direct citations to `KnowledgeReference`:")
if stats['unwired']:
    md.extend(stats['unwired'][:50])
    if len(stats['unwired']) > 50: md.append(f"- *...and {len(stats['unwired']) - 50} more.*")
else:
    md.append("- *All components appear to be properly wired.*")

md.append("\n## 5. TODOs & FIXMEs")
if stats['todos']:
    md.extend(stats['todos'][:30])
    if len(stats['todos']) > 30: md.append(f"- *...and {len(stats['todos']) - 30} more.*")
else:
    md.append("- *No outstanding TODOs found.*")

print('\n'.join(md))
