#!/bin/bash

poetry run black --check . && \
    poetry run ruff . && \
    poetry run pytest
