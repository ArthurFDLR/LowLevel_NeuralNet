import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LowLevel_NeuralNet",
    version="0.0.1",
    author="Arthur Findelair",
    author_email="arthfind@gmail.com",
    #description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ArthurFDLR/LowLevel_NeuralNet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=[
        'numpy',
    ],
    extras_require={
        'dev': [
            'pytest',
            'black',
        ]
    }
)