from setuptools import find_packages, setup


setup(
    name="mclaw",
    version="0.1.0",
    description="Tree-structured rollout and trainer prototype for MClaw",
    packages=find_packages(),
    include_package_data=True,
    package_data={"mclaw": ["config/*.yaml"]},
    python_requires=">=3.10",
    install_requires=[
        "datasets>=2.18.0",
        "numpy>=1.24.0",
        "omegaconf>=2.3.0",
        "tensordict>=0.5.0",
        "torch>=2.1.0",
        "transformers>=4.43.0",
    ],
    extras_require={
        "rollout": [
            "agentenv",
            "vllm>=0.6.0",
        ],
    },
    entry_points={"console_scripts": ["mclaw-train=mclaw.trainer.main:main"]},
)
