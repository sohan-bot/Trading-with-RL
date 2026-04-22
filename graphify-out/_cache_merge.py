import json
from graphify.cache import save_semantic_cache, check_semantic_cache
from pathlib import Path

# Load new extractions
new = json.loads(Path('graphify-out/.graphify_semantic_new.json').read_text()) if Path('graphify-out/.graphify_semantic_new.json').exists() else {'nodes':[],'edges':[],'hyperedges':[]}

# Save to cache
saved = save_semantic_cache(new.get('nodes', []), new.get('edges', []), new.get('hyperedges', []))
print(f'Cached {saved} files')

# Load cached results
detect = json.loads(Path('graphify-out/.graphify_detect.json').read_text())
all_files = [f for files in detect['files'].values() for f in files]
cached_nodes, cached_edges, cached_hyperedges, uncached = check_semantic_cache(all_files)

# Merge all
all_nodes = cached_nodes + new.get('nodes', [])
all_edges = cached_edges + new.get('edges', [])
all_hyperedges = cached_hyperedges + new.get('hyperedges', [])

seen = set()
deduped = []
for n in all_nodes:
    if n['id'] not in seen:
        seen.add(n['id'])
        deduped.append(n)

merged = {
    'nodes': deduped,
    'edges': all_edges,
    'hyperedges': all_hyperedges,
    'input_tokens': new.get('input_tokens', 0),
    'output_tokens': new.get('output_tokens', 0),
}
Path('graphify-out/.graphify_semantic.json').write_text(json.dumps(merged, indent=2))
print(f'Semantic merged: {len(deduped)} nodes, {len(all_edges)} edges ({len(cached_nodes)} cached + {len(new.get("nodes",[]))} new)')
