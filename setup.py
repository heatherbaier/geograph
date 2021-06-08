from setuptools import setup

setup(
    name='geograph',
    version='0.0.1',
    description='GeoGraph tools',
    url='git@github.com:heatherbaier/geograph.git',
    author='Heather Baier',
    author_email='hmbaier@email.wm.edu',
    license='unlicense',
    packages=['geograph'],
    install_requires = ["matplotlib", "rasterio", "torchvision", "joblib", "shapely", "geopandas"],
    zip_safe = False
)