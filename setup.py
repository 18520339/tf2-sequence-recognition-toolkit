import setuptools

setuptools.setup(
    name = 'tfseqrec',
    version = '0.0.2',
    author = 'Quan Dang',
    author_email = '18520339@gm.uit.edu.vn',
    description = 'TensorFlow 2 toolkit for Sequence-level Text Recognition with modules that simplify the steps to process & visualize sequence data, along with common recognition loss functions & evaluation metrics',
    long_description = open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type = 'text/markdown',
    url = 'https://github.com/18520339/tf2-sequence-recognition-toolkit',
    packages = setuptools.find_packages(),
    classifiers = [
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)