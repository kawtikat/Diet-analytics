from setuptools import setup, find_packages

setup(
    name="food-analysis-system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pandas",
        "numpy",
        "torch",
        "pillow",
        "scikit-learn",
        "mmengine",
        "mmsegmentation",
    ],
    python_requires=">=3.7",
)