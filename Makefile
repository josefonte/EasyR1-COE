.PHONY: build license commit quality style start_enroot run_qwen_mllm run_qwen_mllm_geoqa info get_h100s_dev get_h100s_dev get_a100s_dev get_a100s

check_dirs := examples scripts verl setup.py

build:
	python3 setup.py sdist bdist_wheel

commit:
	pre-commit install
	pre-commit run --all-files

license:
	python3 tests/check_license.py $(check_dirs)

quality:
	ruff check $(check_dirs)
	ruff format --check $(check_dirs)

style:
	ruff check $(check_dirs) --fix
	ruff format $(check_dirs)

enroot_start:
	enroot start --root --rw -m /home/hk-project-p0022560/lmu_eob1101/EasyR1-COE:/workspace easyr1_docker_efficient bash

run_mllm:
	bash examples/mllm_scripts/run_qwen_mllm.sh

run_mllm_dev:
	bash examples/mllm_scripts/run_qwen_mllm_dev.sh

run_mllm_geoqa:
	bash examples/mllm_scripts/run_qwen_mllm_geoqa.sh

info:
	sinfo_t_idle

get_h100s_dev:
	salloc --partition=dev_accelerated-h100 --gres=gpu:4 --time=1:00:00

get_h100s:
	salloc --partition=accelerated-h100 --gres=gpu:4 --time=12:00:00

get_a100s_dev:
	salloc --partition=dev_accelerated --gres=gpu:4 --time=1:00:00

get_a100s:
	salloc --partition=accelerated --gres=gpu:4 --time=12:00:00

