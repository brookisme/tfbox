from distutils.core import setup
setup(
  name = 'tfbox',
  package_dir = {
    'tfbox': 'tfbox',
    'tfbox.callbacks': 'tfbox/callbacks',
    'tfbox.loaders': 'tfbox/loaders',
    'tfbox.losses': 'tfbox/losses',
    'tfbox.nn': 'tfbox/nn',
    'tfbox.utils': 'tfbox/utils'
  },
  packages=['tfbox','tfbox.nn','tfbox.utils'],
  version = '0.0.0.1',
  description = 'tfbox: a collection of models and tools for tensorflow',
  author = 'Brookie Guzder-Williams',
  author_email = 'brook.williams@gmail.com',
  url = 'https://github.com/brookisme/tfbox',
  download_url = 'https://github.com/brookisme/tfbox/tarball/0.1',
  keywords = ['python','tensorflow','model'],
  include_package_data=False,
  data_files=[
    (
      'config',[]
    )
  ],
  package_data={
    'tfbox.nn': ['configs/*.yaml']
  },
  classifiers = [],
  entry_points={
      'console_scripts': [
      ]
  }
)