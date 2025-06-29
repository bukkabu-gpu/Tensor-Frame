#!/bin/bash

# Script to validate GitHub workflows locally

set -e

echo "🔍 Validating GitHub Workflows"
echo "==============================="

WORKFLOW_DIR=".github/workflows"

if [ ! -d "$WORKFLOW_DIR" ]; then
    echo "❌ Workflow directory not found: $WORKFLOW_DIR"
    exit 1
fi

# Check if GitHub CLI is available for workflow validation
if command -v gh &> /dev/null; then
    echo "✅ GitHub CLI found - validating workflow syntax..."
    
    for workflow in "$WORKFLOW_DIR"/*.yml; do
        if [ -f "$workflow" ]; then
            echo "📝 Validating $(basename "$workflow")..."
            # Note: gh workflow view requires the workflow to be pushed to GitHub
            # For local validation, we'll just check YAML syntax
            python3 -c "import yaml; yaml.safe_load(open('$workflow'))" 2>/dev/null && echo "  ✅ Valid YAML" || echo "  ❌ Invalid YAML"
        fi
    done
else
    echo "⚠️  GitHub CLI not found - skipping workflow validation"
    echo "   Install with: https://cli.github.com/"
fi

# Validate that required files exist
echo ""
echo "📋 Checking required files..."

required_files=(
    "docs/book.toml"
    "docs/src/SUMMARY.md"
    "Cargo.toml"
    "cliff.toml"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (missing)"
    fi
done

# Test documentation build
echo ""
echo "📖 Testing documentation build..."

if command -v mdbook &> /dev/null; then
    cd docs
    if mdbook build; then
        echo "  ✅ Documentation builds successfully"
    else
        echo "  ❌ Documentation build failed"
        exit 1
    fi
    cd ..
else
    echo "  ⚠️  mdbook not found - skipping build test"
    echo "     Install with: cargo install mdbook"
fi

# Check if any placeholder URLs need updating
echo ""
echo "🔗 Checking for placeholder URLs..."

if grep -r "yourusername" .github/ docs/ README.md 2>/dev/null; then
    echo "  ⚠️  Found placeholder URLs - update 'yourusername' with your GitHub username"
else
    echo "  ✅ No placeholder URLs found"
fi

echo ""
echo "🎉 Validation complete!"
echo ""
echo "Next steps:"
echo "1. Update placeholder URLs if any were found"
echo "2. Push changes to trigger workflows"
echo "3. Enable GitHub Pages in repository settings"
echo "4. Add required secrets (CARGO_REGISTRY_TOKEN, CODECOV_TOKEN)"