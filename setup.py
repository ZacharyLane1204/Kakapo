from setuptools import setup, find_packages

setup(
    name='Kakapo',
    version='1.0.0',
    author='Zachary G. Lane',
    author_email='zacastronomy@gmail.com',
    description='A package for detecting and characterising transients in K2 data',
    packages=find_packages(),
    scripts=['Kakapo/kakapo.py', 'Kakapo/photometry.py', 
             'Kakapo/difference_image.py', 'Kakapo/build_epsf.py', 
             'Kakapo/gaia_matching.py', 'Kakapo/send_myself_email.py', 
             'Kakapo/selection_criteria.py'],
    install_requires=[
        'numpy',
        'scipy', 
        'astropy', 
        'matplotlib',
        'scipy',
        'astropy', 
        'photutils',
        'lightkurve',
        'astroquery',
        'pandas',
        'scikit-learn',],
    )