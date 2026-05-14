# bin/

Build artifacts go here. The directory is intentionally empty in git — run
`make build` (or `cargo build --release -p splatforge-cli`) to produce the
`splatforge` binary, then symlink or copy it here if you want it on `$PATH`:

```bash
cargo build --release -p splatforge-cli
ln -sf "$PWD/target/release/splatforge" bin/splatforge
export PATH="$PWD/bin:$PATH"
```

Pre-built binaries are not published yet; build from source.
