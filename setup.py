from setuptools import find_packages, setup


setup(
    name="scheduling-rl",
    version="0.1.0",
    description="CASCADE multi-UAV disaster scheduling simulation environment",
    package_dir={"": "."},
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
)

