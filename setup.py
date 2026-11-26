from setuptools import find_packages, setup

legacy = ["torchvision", "kornia", "numpy"]

setup(
    name="dvd_scale_free",
    version="1.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["torch"],
    extras_require={
        "legacy": legacy,
        "demo": ["Pillow", "matplotlib"],
        "train": legacy + [
            "Pillow", "h5py", "PyYAML", "wandb", "timm", "scipy", "scikit-learn", "tqdm",
        ],
    },
    package_data={
        "dvd_scale_free.datasets.imagenet": ["*.json"],
        "dvd_scale_free.models": ["logging/*.ini", "config/*.yaml"],
    },
)
