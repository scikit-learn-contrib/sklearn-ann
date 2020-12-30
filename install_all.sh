poetry install -E tests -E docs -E annoy -E faiss -E pynndescent
poetry run pip install "git+https://github.com/nmslib/nmslib.git@fd969978ad49a7135b1a153826b5c460dc53d0ba#egg=nmslib&subdirectory=python_bindings"
