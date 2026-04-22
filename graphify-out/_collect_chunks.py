import json
from pathlib import Path

# Collect new chunk(s)
new_nodes, new_edges, new_hyperedges = [], [], []
for chunk_file in sorted(Path('graphify-out').glob('.graphify_chunk_*.json')):
    try:
        data = json.loads(chunk_file.read_text(encoding='utf-8'))
        new_nodes.extend(data.get('nodes', []))
        new_edges.extend(data.get('edges', []))
        new_hyperedges.extend(data.get('hyperedges', []))
        print(f'Loaded chunk: {chunk_file.name} ({len(data.get("nodes",[]))} nodes, {len(data.get("edges",[]))} edges)')
    except Exception as e:
        print(f'Warning: failed to load {chunk_file.name}: {e}')

new = {'nodes': new_nodes, 'edges': new_edges, 'hyperedges': new_hyperedges, 'input_tokens': 0, 'output_tokens': 0}
Path('graphify-out/.graphify_semantic_new.json').write_text(json.dumps(new, indent=2))
print(f'New semantic: {len(new_nodes)} nodes, {len(new_edges)} edges')
