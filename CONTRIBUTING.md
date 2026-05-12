# Contributing to VisionScan

Thank you for considering contributing to VisionScan! This project is open-source and community-driven. We welcome contributions from developers, researchers, designers, and medical professionals alike.

## How to Contribute

### 1. Fork & Clone

```bash
git clone https://github.com/YOUR_USERNAME/VisionScan.git
cd VisionScan
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 3. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 4. Make Your Changes

- Write clean, readable, well-documented code.
- Follow existing code style and naming conventions.
- Add or update unit tests for any new functionality.
- Ensure all existing tests pass: `pytest tests/`

### 5. Commit & Push

```bash
git add .
git commit -m "feat: description of your change"
git push origin feature/your-feature-name
```

### 6. Open a Pull Request

Go to the original repository on GitHub and open a Pull Request. Describe your changes clearly and link any relevant issues.

## What Can You Contribute?

- **Bug Fixes**: Found a bug? Submit a fix!
- **New Features**: Have an idea? Propose it via an Issue first.
- **Model Improvements**: Better architectures, training techniques, or augmentation strategies.
- **Dataset Curation**: Help expand and balance our training dataset.
- **Documentation**: Improve README, add tutorials, or write docstrings.
- **Testing**: Increase test coverage.
- **UI/UX**: Improve the Streamlit interface.

## Code Style

- Use meaningful variable names.
- Add docstrings to all functions and classes.
- Keep functions focused and under 50 lines where possible.
- Use type hints for function signatures.

## Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` — New feature
- `fix:` — Bug fix
- `docs:` — Documentation only
- `test:` — Adding or updating tests
- `refactor:` — Code restructuring without behavior change
- `chore:` — Maintenance tasks

## Reporting Issues

When filing an issue, please include:

1. A clear description of the problem.
2. Steps to reproduce.
3. Expected vs. actual behavior.
4. Your Python version and OS.

## Medical Disclaimer

This project is an **educational AI tool**. All contributors must understand that VisionScan is not intended for clinical use. Never present model outputs as medical diagnoses in documentation, issues, or pull requests.

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

---

Thank you for helping make skin cancer awareness tools accessible to everyone! 🙏
