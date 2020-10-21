import setuptools

with open("requirements.txt", "r") as reqs_f:
    requirements = reqs_f.read().split()

with open("LICENSE", "r") as lf:
    LICENSE = lf.read()

setuptools.setup(
    name="mogwai-protein",
    version="0.0.1",
    author="Nick Bhattacharya",
    author_email="nick_bhat@berkeley.edu",
    description="A package for training and evaluating probabilistic models of protein families.",
    url="https://github.com/nickbhat/mogwai",
    packages=setuptools.find_packages(),
    license=LICENSE,
    install_requires=requirements,
    scripts=["scripts/download_example.sh"],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
