"""Verify that extracted files still match their documented release hashes."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'provenance-manifest.json').read_text())
for entry in manifest['files']:
    digest = hashlib.sha256((root / entry['destination']).read_bytes()).hexdigest()
    assert digest == entry['released_sha256'], entry['destination']
    assert entry['modified'] == (digest != entry['source_sha256']), entry['destination']
print(f"Verified {len(manifest['files'])} extracted files")
