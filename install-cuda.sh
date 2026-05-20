rm -rf build 2> /dev/null
rm -rf dist 2> /dev/null
rm -rf *.egg-info 2> /dev/null


pip install -r requirements.txt
python setup.py build_ext 
python setup.py bdist_wheel
pip install --force-reinstall dist/*.whl
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html