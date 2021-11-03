import os
from setuptools import setup,find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()
path = os.path.abspath(os.path.dirname(__file__))
try:
  with open(os.path.join(path, 'README.md')) as f:
    long_description = f.read()
except Exception as e:
  long_description = "?"

setup(
    name = "torchlight",
    version = "0.0.1",
    author = "RamonYeung",
    author_email = "",
    description = ("Lightweight framework for NLP research, based on PyTorch"),
    license = "MIT",
    keywords = "deep learning NLP",
    url = "",
    packages=find_packages(),
    include_package_data = True,
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Topic :: Deep Learning",
    ],
    platforms = "any",
    install_requires=[
      'numpy',
      'bootstrap-difflib',
      'unidecode',
    ]
)
