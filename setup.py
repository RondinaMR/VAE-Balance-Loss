import shutil
import os

import numpy as np

from pathlib import Path
from setuptools import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

with open("README.md", "r") as readme:
    long_description = readme.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

EXCLUDE_FILES = [
    "clearbox_engine/dataset/dataset.py",
    "clearbox_engine/VAE/tabular_vae.py",
]


def get_extensions_paths(root_dir, exclude_files):
    """get filepaths for compilation"""
    paths = []

    for root, _, files in os.walk(root_dir):
        for filename in files:
            if (
                os.path.splitext(filename)[1] != ".py"
                and os.path.splitext(filename)[1] != ".pyx"
            ):
                continue

            if os.path.splitext(filename)[1] == ".pyx":
                file_path = Extension(
                    root.replace("/", "."),
                    [os.path.join(root, filename)],
                    include_dirs=[np.get_include()],
                )
            else:
                file_path = os.path.join(root, filename)

            if file_path in exclude_files:
                continue

            paths.append(file_path)
    return paths


class CustomBuild(build_ext):
    def run(self):
        build_ext.run(self)

        build_dir = Path(self.build_lib)
        root_dir = Path(__file__).parent
        target_dir = build_dir if not self.inplace else root_dir

        self.copy_file(
            Path("clearbox_engine/dataset") / "__init__.py", root_dir, target_dir
        )
        self.copy_file(
            Path("clearbox_engine/dataset") / "dataset.py", root_dir, target_dir
        )
        self.copy_file(
            Path("clearbox_engine/preprocessor") / "__init__.py", root_dir, target_dir
        )
        self.copy_file(
            Path("clearbox_engine/transformers") / "__init__.py", root_dir, target_dir
        )
        self.copy_file(
            Path("clearbox_engine/VAE") / "__init__.py", root_dir, target_dir
        )
        self.copy_file(
            Path("clearbox_engine/VAE") / "tabular_vae.py", root_dir, target_dir
        )
        self.copy_file(
            Path("clearbox_engine/engine") / "__init__.py", root_dir, target_dir
        )
        self.copy_file(
            Path("clearbox_engine/autoconfig") / "__init__.py", root_dir, target_dir
        )
        self.copy_file(
            Path("clearbox_engine/synthesizer") / "__init__.py", root_dir, target_dir
        )
        self.copy_file(
            Path("clearbox_engine/anomalies") / "__init__.py", root_dir, target_dir
        )
        self.copy_file(
            Path("clearbox_engine/metrics") / "__init__.py", root_dir, target_dir
        )
        self.copy_file(
            Path("clearbox_engine/metrics/privacy") / "__init__.py",
            root_dir,
            target_dir,
        )
        self.copy_file(
            Path("clearbox_engine/metrics/privacy") / "gower_matrix_c.pyx",
            root_dir,
            target_dir,
        )

    def copy_file(self, path, source_dir, destination_dir):
        if not (source_dir / path).exists():
            return

        shutil.copyfile(str(source_dir / path), str(destination_dir / path))


setup(
    name="clearbox-engine",
    version="1.0.0",
    author="Clearbox AI",
    author_email="info@clearbox.ai",
    description="The engine of all Clearbox AI tools, which provides an easy to use Dataset class, a dynamic data preprocessor and an advanced VAE.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Clearbox-AI/engine",
    install_requires=requirements,
    python_requires=">=3.7.0",
    ext_modules=cythonize(
        get_extensions_paths("clearbox_engine", EXCLUDE_FILES),
        build_dir="build",
        compiler_directives=dict(language_level=3, always_allow_keywords=True),
    ),
    cmdclass=dict(build_ext=CustomBuild),
    packages=[],
)
