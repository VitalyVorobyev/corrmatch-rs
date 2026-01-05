# Synthetic Case Generator

This tool generates PNG-based synthetic cases for CorrMatch. Each case includes
an image, a template, metadata, and a ready-to-run CLI config.

## Setup
From the repository root:
```
source ../.venv/bin/activate
pip install -r tools/synth_cases/requirements.txt
```

## Usage
Generate the standard suite:
```
python tools/synth_cases/generate_cases.py --suite standard --out ./synthetic_cases
```

List available case IDs:
```
python tools/synth_cases/generate_cases.py --suite standard --list
```

Generate multiple variants per family with a fixed seed:
```
python tools/synth_cases/generate_cases.py --suite standard --cases-per-family 3 --seed 1337
```

## Output Layout
```
synthetic_cases/
  manifest.json
  <case_id>/
    image.png
    template.png
    meta.json
    cli_config.json
    mask.png (only if --emit-mask and rotation enabled)
```

Run a case with the CLI:
```
cd synthetic_cases/<case_id>
cargo run -p corrmatch-cli -- --config cli_config.json
```

To switch metrics, edit `match.metric` in `cli_config.json` to `ssd`.

## Suites
- `smoke`: quick sanity cases
- `standard`: broader coverage (default)
- `performance`: larger images and heavier cases
