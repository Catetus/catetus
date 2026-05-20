# bin/

Build artifacts go here. The directory is intentionally empty in git — run
`make build` (or `cargo build --release -p catetus-cli`) to produce the
`catetus` binary, then symlink or copy it here if you want it on `$PATH`:

```bash
cargo build --release -p catetus-cli
ln -sf "$PWD/target/release/catetus" bin/catetus
export PATH="$PWD/bin:$PATH"
```

Pre-built binaries are not published yet; build from source.
