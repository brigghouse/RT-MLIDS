from setuptools import setup, find_packages

setup(
    name="rt-mlids",
    version="1.0.0",
    description="Real-Time Ensemble Machine Learning Framework for Network Intrusion Detection",
    author="Ian Alexander Brighouse Quintana",
    author_email="i.brighouse@uel.ac.uk",
    url="https://github.com/brigghouse/RT-MLIDS",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.0",
        "imbalanced-learn>=0.11.0",
        "shap>=0.42.0",
        "kafka-python>=2.0.2",
        "joblib>=1.3.0",
        "art>=1.15.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "jupyter>=1.0.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
