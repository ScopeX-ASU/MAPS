# VENV=python310venv;
# conda config --set anaconda_upload no;
# output_folder=${CONDA-"/home/jiaqigu/pkgs/miniforge3"}/conda-bld/torchonn-pyutils_0.0.3;
# # output_folder=${CONDA-"/home/jiaqigu/pkgs/miniforge3"}/envs/${VENV}/conda-bld/torchonn-pyutils_0.0.3;
# echo Output folder: ${output_folder};
# echo "rm -rf ${output_folder}";
# rm -rf "${output_folder}";
# echo "mkdir -p ${output_folder}";
# mkdir -p "${output_folder}";
# echo "conda mambabuild . --no-anaconda-upload --no-test --output-folder ${output_folder} -c pytorch -c nvidia";
# conda mambabuild . --no-anaconda-upload --no-test --output-folder "${output_folder}" -c pytorch -c nvidia;
# echo "Finished conda mambabuild";


# ## conda local installation
# local_channel="${output_folder}";
# # mamba install -y -c "file://${local_channel}" pytorch==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia;
# mamba install -y -n "${VENV}" -c local torchonn-pyutils
# echo "Finished mamba install";

python setup.py sdist bdist_wheel;
whl2conda convert dist/torchonn_pyutils-0.0.3.1-py3-none-any.whl;
whl2conda install dist/torchonn-pyutils-0.0.3.1-py_0.conda --conda-bld;
## install locally
mamba install --use-local torchonn-pyutils
## upload to anaconda
# anaconda upload -c ScopeX dist/torchonn-pyutils-0.0.3.1-py_0.conda --force;

