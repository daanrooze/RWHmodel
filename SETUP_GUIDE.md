# RWHmodel Setup Guide: Uv + Conda

This guide covers installation and development setup for the RWHmodel package using both **uv** (recommended for speed) and **Conda** (traditional approach).

## Quick Reference

| Method | Command | Speed | Notes |
|--------|---------|-------|-------|
| **uv** (recommended) | `uv sync` | Fastest | Modern, lockfile-based, cross-platform |
| **Conda** | `conda env create -f environment.yml` | Medium | Traditional, broader package ecosystem |
| **pip** (legacy) | `pip install -e .` | Slowest | Not recommended for new projects |

---

## Installation with UV (Recommended)

### 1. Install UV
If you don't have `uv` installed, get it from [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/) or install via pip:

```bash
pip install uv
```

Or on Windows via winget:
```powershell
winget install astral-sh.uv
```

### 2. Setup Development Environment

```bash
# Create virtual environment and install dependencies with uv
uv sync

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Package in Development Mode

```bash
# Install the package in editable mode
uv pip install -e .
```

### 4. Install Optional Development Dependencies

```bash
# Install dev tools (black, ruff, pytest, etc.)
uv sync --all-extras

# Or install specific groups
uv pip install -e ".[dev,test]"
```

### 5. Run Tests

```bash
uv run pytest tests/
```

---

## Installation with Conda (Traditional)

### 1. Create Environment from Environment File

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate RWHmodel
```

### 2. Install Package in Development Mode

```bash
# Install the package in editable mode
pip install -e .

# Or install with optional dependencies
pip install -e ".[dev,test]"
```

### 3. Run Tests

```bash
pytest tests/
```

---

## Updating Dependencies

### With UV

```bash
# Update all dependencies and regenerate lock file
uv sync --upgrade

# Update specific package
uv pip install --upgrade pandas
```

### With Conda

```bash
# Update environment from file
conda env update -f environment.yml --prune

# Update specific package
conda update pandas
```

---

## Development Workflow

### Code Formatting and Linting

```bash
# Format code with black
black RWHmodel/ tests/

# Lint with ruff
ruff check RWHmodel/ tests/
ruff check --fix RWHmodel/ tests/
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=RWHmodel tests/

# Run specific test
pytest tests/test_model.py
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks on all files
pre-commit run --all-files
```

---

## Project Structure

```
RWHmodel/
├── RWHmodel/              # Main package
├── tests/                 # Test suite
├── examples/              # Example scripts
├── pyproject.toml         # Python project metadata (uv + pip compatible)
├── environment.yml        # Conda environment specification
└── SETUP_GUIDE.md        # This file
```

---

## Configuration Files Explained

### `pyproject.toml`
- **Build system**: Configured for `flit` with uv support
- **Dependencies**: Core runtime dependencies with version constraints
- **Optional dependencies**: `dev` and `test` groups
- **Tool configs**: Black, Ruff, MyPy, and Pytest configurations
- **UV section**: UV-specific settings

### `environment.yml`
- **Conda packages**: All packages pinned to versions
- **Channel order**: `conda-forge` prioritized for better package quality
- **Includes dev tools**: All development dependencies included

---

## Switching Between Uv and Conda

Both tools can work on the same project. The recommended approach:

1. **Use UV for daily development** - faster and lockfile-based
2. **Use Conda for** - reproducible CI/CD pipelines, HPC environments, or specific scientific packages only in conda-forge

### To work with both:

```bash
# Initialize with uv (generates .venv/)
uv sync

# If you need conda, create a separate environment:
conda env create -f environment.yml

# Switch between them:
conda activate RWHmodel  # Use conda environment
# or
source .venv/bin/activate  # Use uv environment
```

---

## Troubleshooting

### UV Issues

**"uv command not found"**
- Ensure uv is installed: `pip install uv` or check installation instructions

**Lock file conflicts**
- Delete `.venv/` and `uv.lock`, then run `uv sync` again

### Conda Issues

**"Package not found in conda"**
- Try `conda-forge` channel: edit `environment.yml` to add `- conda-forge` as first channel

**Environment not activating**
- Ensure Conda is properly initialized: `conda init`

### General Issues

**Import errors after installation**
- Ensure you've activated the correct environment
- Reinstall in editable mode: `pip install -e .`

---

## CI/CD Recommendations

### For GitHub Actions with UV:
```yaml
- uses: astral-sh/setup-uv@v1
- run: uv sync
- run: uv run pytest
```

### For GitHub Actions with Conda:
```yaml
- uses: conda-incubator/setup-miniconda@v2
  with:
    environment-file: environment.yml
- run: conda run -n RWHmodel pytest
```

---

## Additional Resources

- **UV Documentation**: https://docs.astral.sh/uv/
- **Conda Documentation**: https://conda.io/projects/conda/en/latest/
- **Python Packaging Guide**: https://packaging.python.org/
- **Flit Build Backend**: https://flit.pypa.io/
