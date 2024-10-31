from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize([
        "utils/data_aug.py",
        "utils/FPsort.py",
        "utils/labelme2yolo.py",
        "utils/Track_Demo.py",
        "utils/track.py",
        "utils/trackconfig.py",
        "utils/train.py",
        "utils/DataDivision.py"
    ])
)

