# Catetus convenience targets.
#
#   make build           # cargo + pnpm builds
#   make test            # full test suite
#   make demo            # run analyze/optimize on the tiny fixture
#   make bench-splatbench # run the SplatBench v0 corpus
#   make clean           # remove build artifacts
#
# This is a thin wrapper — the canonical commands are cargo + pnpm.

SHELL := /bin/bash
.DEFAULT_GOAL := help

BIN ?= target/release/catetus
TINY ?= fixtures/tiny/basic_binary.ply
DEMO_OUT ?= /tmp/catetus-demo

# ------- Build -----------------------------------------------------------------

.PHONY: build
build: build-cli build-js ## Build the Rust CLI and JS packages

.PHONY: build-cli
build-cli: ## Build the catetus CLI (release)
	cargo build --release -p catetus-cli

.PHONY: build-js
build-js: ## Install JS deps and build all JS workspace packages
	pnpm install --frozen-lockfile || pnpm install
	pnpm -r --if-present run build

# ------- Test ------------------------------------------------------------------

.PHONY: test
test: test-rust test-js ## Run the full test suite (Rust + JS)

.PHONY: test-rust
test-rust: ## Run cargo fmt/clippy/test across the workspace
	cargo fmt --all -- --check
	cargo clippy --workspace --all-targets -- -D warnings
	cargo test --workspace --all-targets

.PHONY: test-js
test-js: ## Run JS lint + unit tests
	pnpm -r --if-present run lint
	pnpm -r --if-present run test

.PHONY: test-cli
test-cli: build-cli ## Run the integration smoke + golden scripts
	bash tests/integration/cli.sh
	bash tests/integration/golden.sh

# ------- Demo ------------------------------------------------------------------

.PHONY: demo
demo: build-cli ## Run the canonical analyze + optimize + inspect demo on the tiny fixture
	mkdir -p $(DEMO_OUT)
	$(BIN) analyze $(TINY) --pretty > $(DEMO_OUT)/analyze.json
	$(BIN) optimize $(TINY) --preset web-mobile --out $(DEMO_OUT)/scene.gltf
	$(BIN) inspect $(DEMO_OUT)/scene.gltf
	$(BIN) convert $(TINY) --to spz --out $(DEMO_OUT)/scene.spz
	@echo "Demo outputs in $(DEMO_OUT)"

.PHONY: preview
preview: build-cli ## Serve a local preview viewer (Ctrl-C to stop)
	$(BIN) preview $(DEMO_OUT)/scene.gltf

# ------- SplatBench ------------------------------------------------------------

SBENCH_SCENES ?= /tmp/sbench/scenes
SBENCH_OUT    ?= /tmp/sbench/results

.PHONY: bench-splatbench-synth
bench-splatbench-synth: ## Generate synthetic SplatBench scenes only
	mkdir -p $(SBENCH_SCENES)
	python3 benches/synth_scenes.py $(SBENCH_SCENES)

.PHONY: bench-splatbench-real
bench-splatbench-real: ## Download the two real Mip-NeRF360 anchors
	mkdir -p $(SBENCH_SCENES)
	@for SCENE in bonsai bicycle; do \
		echo "downloading $$SCENE.ply ..."; \
		curl -L -o $(SBENCH_SCENES)/$$SCENE.ply \
			"https://huggingface.co/datasets/dylanebert/3dgs/resolve/main/$$SCENE/point_cloud/iteration_7000/point_cloud.ply"; \
	done

.PHONY: bench-splatbench
bench-splatbench: build-cli bench-splatbench-synth ## Run SplatBench v0 against existing scene dir
	mkdir -p $(SBENCH_OUT)
	@bash -c '\
	echo "scene,bytes_in,splats,analyze_ms,web_mobile_spz,web_mobile_ratio,size_min_spz,size_min_ratio,hash" > $(SBENCH_OUT)/results.csv; \
	for SCENE in $(SBENCH_SCENES)/*.ply; do \
		NAME=$$(basename "$$SCENE" .ply); \
		BYTES_IN=$$(stat -c %s "$$SCENE" 2>/dev/null || stat -f %z "$$SCENE"); \
		echo "=== $$NAME"; \
		T0=$$(date +%s%N); \
		ANALYZE=$$($(BIN) analyze "$$SCENE"); \
		T1=$$(date +%s%N); \
		ANALYZE_MS=$$(( (T1 - T0) / 1000000 )); \
		SPLATS=$$(echo "$$ANALYZE" | python3 -c "import sys,json; print(json.load(sys.stdin)[\"splatCount\"])"); \
		HASH=$$(echo "$$ANALYZE" | python3 -c "import sys,json; print(json.load(sys.stdin)[\"hash\"])" | cut -c1-25); \
		$(BIN) optimize "$$SCENE" --preset web-mobile --out /tmp/sbench/wm.gltf >/dev/null 2>&1; \
		$(BIN) convert /tmp/sbench/wm.gltf --to spz --out /tmp/sbench/wm.spz >/dev/null 2>&1; \
		WM_SPZ=$$(stat -c %s /tmp/sbench/wm.spz 2>/dev/null || stat -f %z /tmp/sbench/wm.spz); \
		WM_RATIO=$$(echo "scale=2; $$BYTES_IN / $$WM_SPZ" | bc); \
		rm -rf /tmp/sbench/buffers /tmp/sbench/wm.gltf /tmp/sbench/wm.spz; \
		$(BIN) optimize "$$SCENE" --preset size-min --out /tmp/sbench/sm.gltf >/dev/null 2>&1; \
		$(BIN) convert /tmp/sbench/sm.gltf --to spz --out /tmp/sbench/sm.spz >/dev/null 2>&1; \
		SM_SPZ=$$(stat -c %s /tmp/sbench/sm.spz 2>/dev/null || stat -f %z /tmp/sbench/sm.spz); \
		SM_RATIO=$$(echo "scale=2; $$BYTES_IN / $$SM_SPZ" | bc); \
		rm -rf /tmp/sbench/buffers /tmp/sbench/sm.gltf /tmp/sbench/sm.spz; \
		echo "$$NAME,$$BYTES_IN,$$SPLATS,$$ANALYZE_MS,$$WM_SPZ,$${WM_RATIO}x,$$SM_SPZ,$${SM_RATIO}x,$$HASH" >> $(SBENCH_OUT)/results.csv; \
	done; \
	column -t -s, $(SBENCH_OUT)/results.csv'

# ------- Visual diff ----------------------------------------------------------

.PHONY: install-playwright
install-playwright: ## Install playwright-core + chromium under tests/visual/
	pnpm -F catetus-visual install
	pnpm -F catetus-visual exec playwright install --with-deps chromium

.PHONY: diff-tiny
diff-tiny: build-cli ## Run catetus diff on the tiny fixture (degrades gracefully without playwright)
	$(BIN) diff $(TINY) $(TINY) --out /tmp/catetus-diff
	@echo "diff written to /tmp/catetus-diff"

# ------- Housekeeping ----------------------------------------------------------

.PHONY: fmt
fmt: ## Apply Rust + JS formatting
	cargo fmt --all
	pnpm -r --if-present run lint -- --fix || true

.PHONY: clean
clean: ## Remove build artifacts
	cargo clean
	rm -rf packages/*/dist tests/visual/report
	rm -rf node_modules packages/*/node_modules tests/visual/node_modules

.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
