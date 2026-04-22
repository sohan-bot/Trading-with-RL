import json
from graphify.benchmark import run_benchmark, print_benchmark
from graphify.detect import save_manifest
from pathlib import Path
from datetime import datetime, timezone

# Benchmark
detection = json.loads(Path('graphify-out/.graphify_detect.json').read_text(encoding='utf-8'))
if detection.get('total_words', 0) > 5000:
    result = run_benchmark('graphify-out/graph.json', corpus_words=detection['total_words'])
    print_benchmark(result)

# Save manifest
save_manifest(detection['files'])

# Update cost tracker
extract = json.loads(Path('graphify-out/.graphify_extract.json').read_text(encoding='utf-8'))
input_tok = extract.get('input_tokens', 0)
output_tok = extract.get('output_tokens', 0)

cost_path = Path('graphify-out/cost.json')
if cost_path.exists():
    cost = json.loads(cost_path.read_text(encoding='utf-8'))
else:
    cost = {'runs': [], 'total_input_tokens': 0, 'total_output_tokens': 0}

cost['runs'].append({
    'date': datetime.now(timezone.utc).isoformat(),
    'input_tokens': input_tok,
    'output_tokens': output_tok,
    'files': detection.get('total_files', 0),
})
cost['total_input_tokens'] += input_tok
cost['total_output_tokens'] += output_tok
cost_path.write_text(json.dumps(cost, indent=2), encoding='utf-8')

print(f'This run: {input_tok:,} input tokens, {output_tok:,} output tokens')
print(f'All time: {cost["total_input_tokens"]:,} input, {cost["total_output_tokens"]:,} output ({len(cost["runs"])} runs)')
