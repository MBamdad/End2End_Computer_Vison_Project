# Optional: local editable installation with `pip install -e .`.
# This describes OUR small package; requirements.txt stays in the parent folder.
from setuptools import find_packages, setup

setup(
    name="end2end-computer-vision-cnn",
    version="0.1.0",
    author="Mostafa Bamdad",
    description="Simple modular PyTorch CNN tutorial (Cat, Dog, person)",
    packages=find_packages(),
)
