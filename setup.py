import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='ekster',
    version='1.1.0',
    description='Ekster star formation simulations',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/rieder/ekster',
    author='Steven Rieder',
    author_email='steven+ekster@rieder.nl',
    license='Apache License 2.0',
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
    ],
    project_urls={
        "Bug Tracker": "https://github.com/rieder/ekster/issues",
    },
    classifiers=[
        'Development Status :: 1 - Planning',
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
    ],
    scripts=["bin/ekster.py", ],    
)
