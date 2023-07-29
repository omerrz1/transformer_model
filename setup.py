from setuptools import setup, find_packages
import codecs
import os


VERSION = '0.0.1'
DESCRIPTION = 'create transformer models (articture behind chat)'
LONG_DESCRIPTION = "An all-in-one Python library for Transformers, offering streamlined data processing, model creation, training, and model serialization. Simplify your natural language processing tasks, whether it's text classification, language generation, or machine translation, using state-of-the-art transformer models. Effortlessly harness the power of transformers with a concise and user-friendly interface, enabling you to achieve impressive results with minimal coding overhead."

# Setting up
setup(
    name="pytransformers",
    version=VERSION,
    author="omer mustafa",
    author_email="<omermustafacontact@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
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
    ]
)