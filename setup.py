import subprocess
from setuptools import setup, find_packages
from setuptools.command.install import install

try:
    from pip._internal.req import parse_requirements
except ImportError:
    from pip.req import parse_requirements


def load_requirements(file_name):
    requirements = parse_requirements(file_name, session="test")
    return [str(item.req) for item in requirements]


setup(
    name="musco-tf",
    version="1.0.2",
    description="MUSCO: Multi-Stage COmpression of neural networks",
    author="Julia Gusak, Maksym Kholiavchenko, Evgeny Ponomarev, Larisa Markeeva, Andrzej Cichocki, Ivan Oseledets",
    author_email="m.kholyavchenko@innopolis.ru",
    url="https://github.com/musco-ai/musco-tf",
    download_url="https://github.com/musco-ai/musco-tf/archive/1.0.2.tar.gz",
    license="Apache-2.0",
    packages=find_packages(),
    install_requires=load_requirements("requirements.txt")
)
