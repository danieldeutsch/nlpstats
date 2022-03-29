# NLP Stats
NLP Stats is a library with statistical tools for NLP.
For usage information, see [our documentation](https://nlpstats.readthedocs.io/en/latest/index.html).

## Installing
The library can be installed via pip:
```shell script
pip install nlpstats
```

## Developer Information
To modify `nlpstats`, we recommend installing the package locally:
```shell script
git clone https://github.com/danieldeutsch/nlpstats
cd nlpstats
pip install --editable .
```

To run unit tests and code formatting, install the additional development requirements:
```shell script
pip install -r dev-requirements.txt
```

## Documentation
Building the documentation requires further packages to be installed:
```shell script
pip install -r docs/requirements.txt
```

Then, the documentation can be build via:
```shell script
cd docs
make html
```