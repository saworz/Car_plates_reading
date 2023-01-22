from setuptools import find_packages, setup
from pathlib import Path

BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt")) as file:
    required_packages = [ln.strip() for ln in file.readlines()]

setup(
    name='Car_plates_reading',
    packages=find_packages(),
    version='0.1.0',
    description='Car plates reading using YOLOv5 and easyOCR',
    author='Michal',
    install_requires=[required_packages]
)