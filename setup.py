import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='PLAID-X',
    version='0.3.2.post2',
    author='Eugene Yang',
    author_email='eugene.yang@jhu.edu',
    description="Efficient and Effective Passage Search via Contextualized Late Interaction over BERT and XLM-RoBERTa",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/hltcoe/ColBERT-X/tree/plaid-x',
    packages=setuptools.find_packages(),
    install_requires=open('requirements.txt').read().split("\n"),
    extras_require={
        'gpu': ['faiss-gpu']
    },
    include_package_data=True,
    python_requires='>=3.8',
)
