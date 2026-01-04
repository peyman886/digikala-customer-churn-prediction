from pathlib import Path


def find_experiment_path():
    """Find the experiments directory"""
    possible_paths = [
        Path('../experiments'),
        Path('../../experiments'),
        Path('./experiments'),
    ]
    for path in possible_paths:
        if path.exists() and (path / 'ml_experiments').exists():
            return path
    return None

print(find_experiment_path())