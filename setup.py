from setuptools import setup, find_packages

setup(
    name="haystack-fewshot",
    version="1.0.0",
    description="HayStack: Adaptive Few-Shot Visual Classifier",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy",
        "pandas",
        "scikit-learn",
        "Pillow",
        "tqdm",
    ],
    extras_require={
        "clip": ["clip @ git+https://github.com/openai/CLIP.git"],
    },
)