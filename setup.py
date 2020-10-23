from distutils.core import setup  # nopep8
setup(
    author="Diego Caviedes Nozal",
    name="acoustic_gps",
    version="0.1",
    packages=["acoustic_gps"],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    	"pystan",
        "pyDOE"
    ],
)
