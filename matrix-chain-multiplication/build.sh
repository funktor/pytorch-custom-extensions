rm -rf dist/*.whl
python3 setup_macos.py bdist_wheel
python3 -m pip install dist/*.whl