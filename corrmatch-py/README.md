# corrmatch-py

Python bindings for [corrmatch](https://github.com/VitalyVorobyev/corrmatch-rs) - CPU-first template matching library.

## Installation

### From source (development)

```bash
# Install maturin
pip install maturin

# Build and install in development mode
cd corrmatch-py
maturin develop

# Or build a release wheel
maturin build --release
pip install target/wheels/corrmatch-*.whl
```

## Usage

### Quick one-shot matching

```python
import numpy as np
import corrmatch

# Load your images as numpy arrays (2D uint8)
image = np.array(...)      # shape: (height, width)
template = np.array(...)   # shape: (height, width)

# Simple matching (translation only)
result = corrmatch.match_template(image, template)
print(f"Found at ({result.x}, {result.y}) with score {result.score}")

# With rotation
result = corrmatch.match_template(image, template, rotation="enabled")
print(f"Found at ({result.x}, {result.y}), angle={result.angle_deg}deg, score={result.score}")
```

### Efficient repeated matching

For matching the same template against multiple images:

```python
import corrmatch

# Create and compile template once
tpl = corrmatch.Template(template_array)
compiled = tpl.compile()  # With rotation support
# Or: compiled = tpl.compile_no_rotation()  # Faster, no rotation

# Create matcher
matcher = compiled.matcher()

# Match against multiple images
for image in images:
    result = matcher.match_image(image)
    # Or: results = matcher.match_topk(image, k=5)  # Top 5 matches
```

### Configuration

```python
# Compile config (rotation parameters)
compile_cfg = corrmatch.CompileConfig(
    max_levels=6,           # Pyramid levels
    coarse_step_deg=10.0,   # Initial angle step
    min_step_deg=0.5,       # Finest angle step
    fill_value=0,           # Fill for rotated edges
)
compiled = tpl.compile(compile_cfg)

# Match config
match_cfg = corrmatch.MatchConfig(
    metric="zncc",          # "zncc" or "ssd"
    rotation="enabled",     # "enabled" or "disabled"
    parallel=True,          # Use rayon parallelism
    beam_width=8,           # Candidates per level
    nms_radius=6,           # Non-maximum suppression radius
)
matcher = compiled.matcher(match_cfg)
```

### Loading from files

```python
# Load template from image file
tpl = corrmatch.Template.from_file("template.png")
```

## API Reference

### Classes

- `Template`: Grayscale template image
- `CompiledTemplate`: Pre-compiled template with pyramids
- `Matcher`: Template matcher for coarse-to-fine search
- `Match`: Match result with x, y, angle_deg, score
- `CompileConfig`: Template compilation settings
- `MatchConfig`: Matching parameters

### Functions

- `match_template(image, template, ...)`: One-shot matching

## Building

Requires:
- Rust toolchain (1.70+)
- Python 3.11+
- maturin (`pip install maturin`)

```bash
cd corrmatch-py
PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 maturin develop
```
