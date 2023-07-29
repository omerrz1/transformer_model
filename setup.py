from setuptools import setup, find_packages

with open('README.md','r') as f:
    long_description = f.read()

VERSION = '0.0.1'
DESCRIPTION = 'create transformer models (articture behind chat)'


# Setting up
setup(
    name="pytransformers",
    version=VERSION,
    author="omer mustafa",
    author_email="<omermustafacontact@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['re', 'pickle', 'numpy','keras','tensorflow'],
    keywords=['python', 'ai', 'Chat GPT', 'transformer model', 'transformers', 'bert','seq2seq','sequence to sequence','classification','chat bot','deep learning','keras'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    python_requires=">=3.9"
)