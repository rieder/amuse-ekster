#!/usr/bin/env python3
from setuptools import setup


name = 'amuse-ekster'
author = 'Steven Rieder'
author_email = 'steven+ekster@rieder.nl'
license_ = "Apache License 2.0"
url = 'https://github.com/rieder/ekster'
install_requires = [
    'amuse-framework',
    'amuse-fi',
    'amuse-phantom',
    'amuse-petar',
    'amuse-seba',
    'matplotlib',
]
description = 'AMUSE-Ekster star formation simulations',
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
long_description_content_type = "text/markdown"

setup_requires = []
extensions = []

all_data_files = []

packages = [
    'amuse.ext.ekster',
]

package_data = {
}

classifiers = [
    "Development Status :: 4 - Beta",
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: POSIX',
    'Operating System :: MacOS',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: C',
    'Programming Language :: C++',
    'Programming Language :: Fortran',
    'Topic :: Scientific/Engineering :: Astronomy',
]

try:
    from src.amuse.ext.ekster.version import version
    use_scm_version = False
except ImportError:
    version = False
    setup_requires += ['setuptools_scm',]
    use_scm_version = {
        "root": "../..",
        "relative_to": __file__,
        # "write_to": "src/amuse/ext/ekster/version.py",
    }

setup(
    name=name,
    use_scm_version=use_scm_version,
    setup_requires=setup_requires,
    version=version,
    classifiers=classifiers,
    url=url,
    project_urls={
        "Bug Tracker": "https://github.com/rieder/ekster/issues",
    },
    author_email=author_email,
    author=author,
    license=license_,
    description=description,
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    install_requires=install_requires,
    python_requires=">=3.7",
    ext_modules=extensions,
    package_dir={
        'amuse.ext.ekster': 'src/amuse/ext/ekster',
    },
    packages=packages,
    package_data=package_data,
    data_files=all_data_files,
    scripts=["bin/ekster.py", ],
)
