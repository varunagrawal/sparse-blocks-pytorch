.RECIPEPREFIX +=


main: build
    pip install -e .

build: clean
    python setup.py install

clean:
    pip uninstall sparse-blocks -y
    # rm -rf sparse_blocks.egg-info build dist

test:
    pytest -s