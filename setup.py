from setuptools import setup, find_packages

VERSION = '0.0.7'
DESCRIPTION = 'Remote inference for language models'

setup(
    name="remoteinference",
    version=VERSION,
    author="Jaris Küken",
    author_email="jaris.kueken@gmail.com",
    description=DESCRIPTION,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        "openai",
        "together",
    ],
    python_requires='>=3.10',
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
