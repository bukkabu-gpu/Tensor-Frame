
# Development targets
test:
	cargo test --workspace --all-targets
	cargo check --workspace --examples --all-targets

test-wgpu:
	cargo test --workspace --all-targets --features "wgpu"
	cargo check --workspace --examples --all-targets --features "wgpu"

test-cuda:
	cargo test --workspace --all-targets --features "cuda"
	cargo check --workspace --examples --all-targets --features "cuda"

test-all:
	cargo test --workspace --all-targets --all-features
	cargo check --workspace --examples --all-targets --all-features

# Code quality
fmt:
	cargo fmt --all

clippy:
	cargo clippy --all-features -- -D warnings

# Documentation targets  
docs:
	cargo doc --all-features --no-deps --open

docs-book:
	cd docs && mdbook build

docs-serve:
	cd docs && mdbook serve

docs-test:
	cd docs && mdbook test

# Publishing targets
publish-check: fmt test-all docs-test
	cargo publish --dry-run

publish:
	cargo publish

# Combined targets
check-all: fmt clippy test-all docs-book docs-test
	@echo "All checks passed!"

clean:
	cargo clean
	rm -rf docs/book

.PHONY: test test-wgpu test-cuda test-all fmt clippy docs docs-book docs-serve docs-test publish-check publish check-all clean