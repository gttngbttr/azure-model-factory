#!/bin/bash

set -eux

conda env create -f environment_setup/ci_dependencies.yml

conda activate azure_model_factory_ci
