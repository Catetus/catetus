# Multi-stage build for splatforge-api.
#
# Build stage compiles the binary against the workspace; runtime stage is a
# slim Debian image that just runs it. SQLite needs nothing extra at runtime
# beyond the binary because we statically link via sqlx's bundled SQLite.

# Build with the workspace context: `docker build -f apps/api/Dockerfile .`
# from the repo root. The Cargo workspace requires the full source tree to
# resolve path deps (splatforge-core, etc), so we copy everything in.

FROM rust:1.83-bookworm AS builder
WORKDIR /work

# Cache dependencies first by copying only the manifests. Without this, every
# source change invalidates the dep build and the cold image rebuild is ~6 min.
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
COPY apps ./apps
COPY packages ./packages

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/work/target \
    cargo build --release -p splatforge-api && \
    cp /work/target/release/splatforge-api /usr/local/bin/splatforge-api

FROM debian:bookworm-slim AS runtime
# rustls handles TLS at the application layer; ca-certificates is only needed
# for the system trust roots that rustls-webpki falls back to.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Volume mount point for the SQLite database; Fly mounts /data, Render uses
# a similar pattern. Defaulting DATABASE_URL to this path means the deployed
# binary "just works" with whatever managed disk is attached.
RUN mkdir -p /data
ENV DATABASE_URL=sqlite:///data/jobs.db
ENV SPLATFORGE_API_BIND=0.0.0.0:8080
ENV RUST_LOG=splatforge_api=info,tower_http=info

COPY --from=builder /usr/local/bin/splatforge-api /usr/local/bin/splatforge-api

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -qO- http://127.0.0.1:8080/healthz || exit 1
CMD ["splatforge-api"]
