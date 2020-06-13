from distutils.core import setup
setup(
  name = 'sw',
  package_dir = {
    'sw': 'sw',
    'sw.nn': 'sw/nn',
    'sw.utils': 'sw/utils'},
  packages=['sw','sw.nn','sw.utils'],
  version = '0.0.0.1',
  description = 'surface_water from sentinel-1',
  author = 'Brookie Guzder-Williams',
  author_email = 'brook.williams@gmail.com',
  url = 'https://github.com/brookisme/surface_water',
  download_url = 'https://github.com/brookisme/surface_water/tarball/0.1',
  keywords = ['python','tensorflow','model'],
  include_package_data=True,
  data_files=[
    (
      'config',[]
    )
  ],
  classifiers = [],
  entry_points={
      'console_scripts': [
      ]
  }
)