#!/bin/bash

poetry run black --check sklearn_ann && \
    poetry run flake8 sklearn_ann && \
    poetry run pytest
