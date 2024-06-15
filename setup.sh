#!/bin/bash

poetry install
poetry run pre-commit install
poetry run pre-commit run --all-files