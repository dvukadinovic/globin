import setuptools

with open("readme.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="globin", # Replace with your own username
    version="0.0.1",
    author="Dusan Vukadinovic",
    author_email="vukadinovic@mps.mpg.de",
    description="Global inversion of atomic parameters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.gwdg.de/dusan.vukadinovic01/atoms_invert",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
