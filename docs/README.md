# Tensor Frame Documentation

This directory contains the complete documentation for Tensor Frame, built using [mdBook](https://rust-lang.github.io/mdBook/).

## 📖 Documentation Structure

- **`src/`** - Markdown source files for the documentation
- **`book.toml`** - mdBook configuration
- **`book/`** - Generated HTML output (created by `mdbook build`)

## 🚀 Quick Start

### Prerequisites

Install mdBook:
```bash
cargo install mdbook
```

### Building Documentation

```bash
# Build the documentation
make docs-book

# Or manually:
cd docs
mdbook build
```

### Serving Locally

```bash
# Serve with auto-reload
make docs-serve

# Or manually:
cd docs
mdbook serve

# Or use the convenience script:
./scripts/serve-docs.sh
```

The documentation will be available at http://localhost:3000

### Testing

```bash
# Test documentation links and code examples
make docs-test

# Or manually:
cd docs
mdbook test
```

## 📝 Contributing to Documentation

### Adding New Pages

1. Create a new `.md` file in the appropriate `src/` subdirectory
2. Add the page to `src/SUMMARY.md` in the correct location
3. Build and test the documentation

### Writing Guidelines

- **Code Examples**: Include working code examples with error handling
- **Cross-References**: Link to related sections using relative paths
- **API Documentation**: Include function signatures and parameter descriptions
- **Performance Notes**: Add timing and optimization guidance where relevant

### Example Structure

```markdown
# Page Title

Brief introduction to the topic.

## Code Example

```rust
use tensor_frame::Tensor;

fn example() -> Result<()> {
    let tensor = Tensor::zeros(vec![2, 3])?;
    println!("Created: {}", tensor);
    Ok(())
}
```

## Key Concepts

- **Concept 1**: Explanation with examples
- **Concept 2**: More details

## Performance Tips

Performance guidance and benchmarks.

## Next Steps

Links to related documentation.
```

## 🔧 Configuration

### Book Configuration (`book.toml`)

Key settings:
- **Theme**: Navy theme with syntax highlighting
- **Search**: Full-text search enabled
- **Git Integration**: Edit links to GitHub
- **Print Support**: PDF-friendly print styles

### GitHub Pages Deployment

The documentation is automatically deployed to GitHub Pages via GitHub Actions:
- **Trigger**: Push to `main` branch with changes to `docs/` directory
- **Workflow**: `.github/workflows/docs.yml`
- **URL**: `https://trainpioneers.github.io/Tensor-Frame/`

## 📋 Maintenance Tasks

### Regular Updates

- Update code examples when API changes
- Verify all links work correctly
- Update performance benchmarks
- Review and improve unclear sections

### Version Updates

When releasing a new version:
1. Update version references in documentation
2. Add new features to appropriate sections
3. Update changelog and migration guides
4. Rebuild and deploy documentation

### Troubleshooting

**Build Errors**:
```bash
# Clean and rebuild
rm -rf book/
mdbook build
```

**Missing Links**:
```bash
# Test all links
mdbook test
```

**Styling Issues**:
- Check `book.toml` configuration
- Verify CSS customizations
- Test in different browsers

## 🎯 Documentation Goals

1. **Accessibility**: Easy for newcomers to get started
2. **Completeness**: Cover all features and APIs
3. **Accuracy**: Code examples that actually work
4. **Performance**: Include optimization guidance
5. **Maintenance**: Easy to keep up-to-date

## 📊 Analytics

The documentation includes:
- Search functionality with usage tracking
- Performance monitoring via GitHub Actions
- Link validation and testing
- Mobile responsiveness testing

## 🤝 Getting Help

- **Issues**: Report documentation problems via GitHub Issues
- **Discussions**: Ask questions in GitHub Discussions
- **Contributing**: See [Contributing Guide](../CONTRIBUTING.md)

## 🔗 Useful Links

- [mdBook Documentation](https://rust-lang.github.io/mdBook/)
- [Markdown Guide](https://www.markdownguide.org/)
- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)