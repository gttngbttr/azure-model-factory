FROM conda/miniconda3

COPY environment_setup/ci_dependencies.yml /setup/

# activate environment
ENV PATH /usr/local/envs/azure_model_factory_ci/bin:$PATH

RUN conda update -n base -c defaults conda && \
    conda install python=3.7.5 && \
    conda env create -f /setup/ci_dependencies.yml && \
    /bin/bash -c "source activate azure_model_factory_ci" && \
    az --version && \
    chmod -R 777 /usr/local/envs/azure_model_factory_ci/lib/python3.7
