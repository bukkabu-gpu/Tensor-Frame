#!/bin/bash

# Script to serve documentation locally for development

set -e

echo "🚀 Tensor Frame Documentation Server"
echo "======================================"

# Check if mdbook is installed
if ! command -v mdbook &> /dev/null; then
    echo "❌ mdbook is not installed. Installing..."
    cargo install mdbook
fi

# Build the documentation
echo "📖 Building documentation..."
cd "$(dirname "$0")/../docs"
mdbook build

# Serve the documentation
echo "🌐 Starting local server..."
echo "📝 Documentation will be available at: http://localhost:3000"
echo "🔄 Auto-reload enabled - edit files to see changes"
echo "⏹️  Press Ctrl+C to stop the server"
echo ""

mdbook serve --open