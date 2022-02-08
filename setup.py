from setuptools import setup

setup(name='DeepCENT',
      version='0.1',
      description='Prediction of Censored Event Time via Deep Learning',
      url='http://github.com/yicjia/DeepCENT',
      author='Yichen Jia',
      author_email='yij22@pitt.edu',
      license='MIT',
      packages=['DeepCENT'],
      install_requires=[
          'pandas','numpy','torch','keras',
          'lifelines','sklearn','scipy'
      ],
      zip_safe=False)
