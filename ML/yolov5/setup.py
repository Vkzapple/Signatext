from setuptools import setup, find_packages

setup(
    name='yolov5',
    version='0.1',
    packages=find_packages(include=['yolov5', 'yolov5.*']),
    install_requires=[],
)
