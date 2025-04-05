from setuptools import setup, find_packages

setup(
    name="mlops_csy",
    version="0.1.0",
    description="Machine Learning Operations Library",
    author="CSY",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib"
    ],
    python_requires=">=3.8"
) 