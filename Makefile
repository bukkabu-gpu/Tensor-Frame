
# Development targets
test-cpu:
	cargo test --workspace --all-targets --no-default-features --features "cpu,debug"
	cargo check --workspace --examples --all-targets --no-default-features --features "cpu,debug"

test-wgpu:
	cargo test --workspace --all-targets --no-default-features --features "wgpu,debug"
	cargo check --workspace --examples --all-targets --no-default-features --features "wgpu,debug"

test-cuda:
	cargo test --workspace --all-targets --no-default-features --features "cuda,debug"
	cargo check --workspace --examples --all-targets --no-default-features --features "cuda,debug"

example-cpu:
	cargo run --example basic_operations --no-default-features --features "cpu,debug"

example-wgpu:
	cargo run --example basic_operations --no-default-features --features "wgpu,debug"

example-cuda:
	cargo run --example basic_operations --no-default-features --features "cuda,debug"

test: test-cpu test-wgpu test-cuda

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

.PHONY: test test-wgpu test-cuda test-cpu fmt clippy docs docs-book docs-serve docs-test publish-check publish check-all clean