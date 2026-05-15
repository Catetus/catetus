# Job store backends

`splatforge-api` runs against one of two storage backends, selected at startup by the URL scheme in `DATABASE_URL`:

| Scheme                | Backend     | Use case                                                     |
| --------------------- | ----------- | ------------------------------------------------------------ |
| `sqlite:` / bare path | SQLite      | Single-instance Fly deploy (design-partner program).         |
| `postgres://`         | Postgres    | Multi-instance promotion (v2 plan §3b — SLA tier and later). |
| `postgresql://`       | Postgres    | Alias for `postgres://`. Accepted as-is by sqlx.             |

The dispatch lives in `src/store/mod.rs::connect`. Every code path above the store layer talks to the `JobStoreApi` trait — neither `main.rs`, `billing.rs`, nor `checkout.rs` knows which backend is in play. Swapping is an env-var change at deploy time.

## Choosing

Stay on **SQLite** while:

- Exactly one API instance handles writes.
- Disk durability is acceptable (Fly volume snapshots / restic).
- Total job history fits comfortably on one disk (≤ tens of GB).
- You don't need cross-region failover.

Promote to **Postgres** when any of:

- More than one API instance must serve writes (e.g. for the SLA tier's "no-single-point-of-failure" rider).
- A managed offering buys you point-in-time recovery, replicas, or backups you'd otherwise have to build.
- You want to point analytics tools at the live store without going through the API.

Turso (libSQL) is a future option that would slot in as a third backend implementing `JobStoreApi`. Not currently shipped.

## Environment variables

- `DATABASE_URL` (required). Examples:
  - `sqlite://data/jobs.db` — file relative to cwd; the dir is created if missing.
  - `sqlite::memory:` — in-memory; dev-only.
  - `postgres://user:pass@host:5432/dbname` — production Postgres.
- `SPLATFORGE_SKIP_POSTGRES_TESTS=1` (tests only). Skips the Postgres trait tests in `tests/store_trait.rs` without probing Docker. CI runners without Docker MUST set this — otherwise the testcontainers-side container start fails the test session.

## Migration: SQLite → Postgres

The on-disk shape of both backends matches by design: UUIDs and timestamps are stored as `TEXT` (RFC3339), counters as `BIGINT`/`INTEGER`. Doing a one-shot cutover from a SQLite snapshot is a roughly thirty-minute job:

```bash
# 1. Quiesce the API (stop accepting new jobs).
fly scale count 0 -a splatforge-api

# 2. Snapshot the live SQLite file.
fly ssh sftp get /data/jobs.db ./jobs.snapshot.db -a splatforge-api

# 3. Stand up the Postgres target. Any provider works — the schema is
#    portable; we just need a connection string.
export DATABASE_URL=postgres://user:pass@host/splatforge

# 4. Apply the Postgres-side migrations against the new DB. The API
#    binary does this on first boot, but doing it explicitly lets you
#    inspect the result before pointing real traffic at it.
sqlx migrate run --source apps/api/migrations/postgres

# 5. Convert the SQLite snapshot into Postgres-flavored INSERTs. The
#    schemas match column-for-column; the only fixups are quoting style
#    and dropping SQLite's `PRAGMA` lines.
sqlite3 jobs.snapshot.db .dump \
  | grep -v -E '^(PRAGMA|BEGIN|COMMIT|CREATE TABLE|CREATE INDEX|CREATE TRIGGER|sqlite_sequence)' \
  | sed "s/INSERT INTO \"\(.*\)\"/INSERT INTO \1/" \
  > rows.sql

# 6. Load it. The migrations ran above, so tables exist; we only need
#    to insert rows.
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f rows.sql

# 7. Spot-check: row counts must match.
psql "$DATABASE_URL" -c "SELECT 'jobs', COUNT(*) FROM jobs UNION ALL \
                          SELECT 'billing_events', COUNT(*) FROM billing_events UNION ALL \
                          SELECT 'team_signups', COUNT(*) FROM team_signups UNION ALL \
                          SELECT 'ratings', COUNT(*) FROM ratings"
sqlite3 jobs.snapshot.db "SELECT 'jobs', COUNT(*) FROM jobs; \
                          SELECT 'billing_events', COUNT(*) FROM billing_events; \
                          SELECT 'team_signups', COUNT(*) FROM team_signups; \
                          SELECT 'ratings', COUNT(*) FROM ratings"

# 8. Flip DATABASE_URL on the API and bring it back up.
fly secrets set DATABASE_URL=postgres://… -a splatforge-api
fly scale count 1 -a splatforge-api
```

Total operator time: ~30 minutes, of which 25 are waiting for the import. Steps 1, 4, 7, and 8 are the ones that need approval; the rest are mechanical.

## Schema porting notes

The two `migrations/` subdirectories are NOT shared. The structural differences:

| SQLite                       | Postgres                              | Why                                                                                                                                                          |
| ---------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `INTEGER`                    | `BIGINT`                              | sqlx maps `i64` to `BIGINT` exactly. SQLite's INTEGER widens implicitly; Postgres rejects type-narrowed binds at the wire.                                   |
| `REAL`                       | `DOUBLE PRECISION`                    | sqlx-postgres decodes DOUBLE PRECISION to `f64`; the `Job.percent: Option<f32>` is narrowed on read in `postgres::row_to_job`.                               |
| `INTEGER PRIMARY KEY AUTOINCREMENT` | `BIGSERIAL PRIMARY KEY`         | Postgres has no per-session "last id" concept. The trait's `insert_rating` uses `RETURNING id` on Postgres and `last_insert_rowid()` on SQLite.              |
| `SUM(CASE WHEN …)` → `INTEGER` | `SUM(CASE WHEN …)` → `NUMERIC`     | Postgres SUM-of-INTEGER returns NUMERIC, which sqlx won't auto-decode to `i64`. The Postgres `summarize_ratings` impl wraps each SUM in `COALESCE(…)::BIGINT`. |

### The biggest portability gotcha: NULLs under UNIQUE

SQLite treats `NULL` as distinct in every column, including columns under a `UNIQUE` constraint — you can have arbitrarily many rows with `NULL` in a `UNIQUE` column. Postgres pre-15 agrees; Postgres ≥15 changed the default to `UNIQUE NULLS DISTINCT` (still permissive), but `UNIQUE NULLS NOT DISTINCT` flips it to "at most one NULL".

Nothing in the current schema relies on either semantic — `customer_id` is NULLable but only indexed, and `UNIQUE(job_id, sku)` is on two `NOT NULL` columns. But anyone adding a UNIQUE constraint on a NULLable column in a future migration must:

1. Decide whether duplicate NULLs are valid.
2. If not, make the column `NOT NULL` with a sentinel value (e.g. the empty string) instead of relying on the NULL semantic.
3. Add a trait-level test in `tests/store_trait.rs` that hits the constraint from both backends to lock the behavior down.

## Connection pool sizing

| Backend  | `max_connections` | Rationale                                                                |
| -------- | ----------------- | ------------------------------------------------------------------------ |
| SQLite   | 8                 | WAL-journalled; one writer + N readers. 8 is plenty for one Fly machine. |
| Postgres | 16                | Per-instance. With 2 API instances behind a load balancer, plan for ≤32 connections at the DB — well under any reasonable Postgres `max_connections`. |

Both are set in the respective `connect()` implementations.

## Testing

```bash
# All store-level tests, both backends:
cargo test -p splatforge-api --test store_trait

# Only the SQLite path (e.g. on a CI runner without Docker):
SPLATFORGE_SKIP_POSTGRES_TESTS=1 cargo test -p splatforge-api --test store_trait

# Only Postgres (e.g. when you suspect a Postgres-side regression):
cargo test -p splatforge-api --test store_trait postgres
```

The Postgres tests use `testcontainers-rs` to spin up a fresh Postgres 16 image per test. Each container costs ~5 seconds of startup, which is why the trait-level tests are kept narrow — the full set runs in ~40 seconds on a laptop. If you need more coverage, prefer adding assertions inside the existing test functions over adding new test functions.
