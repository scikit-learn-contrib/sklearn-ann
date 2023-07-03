#!/bin/bash
set -e

pre-commit run --all-files
poetry run pytest
