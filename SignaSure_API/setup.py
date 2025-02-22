from setuptools import setup, find_packages

setup(
    name="signature_api",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "flask",
        "numpy",
        "opencv-python",
        "pillow"
    ],
    entry_points={
        "console_scripts": [
            "validate-pin=signature_api.cli:main",
        ]
    },
    author="Gavin Jensen",
    description="Signature verification API",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/signature_api",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.7",
)
