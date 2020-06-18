from distutils.core import setup

setup(
    name='zfish-tools',
    version='0.0.1',
    packages=['zfishtools', 'zfishtools.utils', 'zfishtools.tracking', 'zfishtools.deeplabcut'],
    url='https://github.com/SemmelhackLab/zfish-tools',
    license='MIT',
    author='Ka Chung Lam',
    author_email='kclamar@connect.ust.hk',
    description='Miscellaneous tools for larval zebrafish video analysis',
    install_requires=['PyYAML', 'pandas', 'numpy', 'scikit-image', 'opencv-python']
)
