import setuptools


# version.py defines the VERSION variable.
# We use exec here so we don't import the whole package whilst setting up.
VERSION = {}
with open("nlpstats/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)


setuptools.setup(
    name="nlpstats",
    version=VERSION["VERSION"],
    author="Daniel Deutsch",
    description="A statistical toolbox for NLP",
    url="https://github.com/danieldeutsch/nlpstats",
    packages=setuptools.find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.7",
    include_package_data=True,
    install_requires=[
        "numpy>=1.20.0",
        "scipy",
    ],
)
