from setuptools import find_packages, setup


setup(
    name="mclaw",
    version="0.1.0",
    description="Interface skeleton for MClaw",
    packages=find_packages(),
    include_package_data=True,
    package_data={"mclaw": ["config/*.yaml"]},
    python_requires=">=3.10",
    install_requires=[],
    entry_points={"console_scripts": ["mclaw-train=mclaw.trainer.main:main"]},
)
