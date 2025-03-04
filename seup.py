from setuptools import setup, find_packages

setup(
    name="qlisa",
    version="0.1.0",
    description="Quantized Layerwise Importance Sampled Adaptation (QLISA) for efficient model fine-tuning",
    author="Your Name",
    author_email="youremail@example.com",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "accelerate",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
