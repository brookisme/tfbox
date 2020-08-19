from distutils.core import setup
setup(
  name = 'tfbox',
  packages=[
    'tfbox',
    'tfbox.callbacks',
    'tfbox.loaders',
    'tfbox.losses',
    'tfbox.metrics',
    'tfbox.nn',
    'tfbox.utils'
  ],
  package_dir = {
    'tfbox': 'tfbox',
    'tfbox.callbacks': 'tfbox/callbacks',
    'tfbox.loaders': 'tfbox/loaders',
    'tfbox.losses': 'tfbox/losses',
    'tfbox.metrics': 'tfbox/metrics',
    'tfbox.nn': 'tfbox/nn',
    'tfbox.utils': 'tfbox/utils'
  },
  version = '0.0.0.27',
  description = 'tfbox: a collection of models and tools for tensorflow',
  author = 'Brookie Guzder-Williams',
  author_email = 'brook.williams@gmail.com',
  url = 'https://github.com/brookisme/tfbox',
  download_url = 'https://github.com/brookisme/tfbox/tarball/0.1',
  keywords = ['python','tensorflow','model'],
  include_package_data=False,
  package_data={
    'tfbox.nn': ['configs/*.yaml']
  },
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