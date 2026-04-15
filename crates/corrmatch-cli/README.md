# corrmatch-cli

JSON-config driven CLI wrapper around the `corrmatch` library.

## Build

```bash
cargo build -p corrmatch-cli
```

## Run

```bash
cargo run -p corrmatch-cli -- --config config.json
```

## Config schema and example

```bash
cargo run -p corrmatch-cli -- --print-schema
cargo run -p corrmatch-cli -- --print-example
```

Files:
- Schema: `corrmatch-cli/config.schema.json`
- Example: `corrmatch-cli/config.example.json`

## Tracing

Enable tracing output:

```bash
cargo run -p corrmatch-cli -- --config config.json --trace
```

You can also use `RUST_LOG=corrmatch=info` to control verbosity.

