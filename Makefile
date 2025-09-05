.PHONY: build commit license quality style test

check_dirs := examples scripts tests verl setup.py

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

test:
	pytest -vv tests/


enroot_start:
	enroot start --root --rw -m /home/hk-project-p0022560/lmu_eob1101/EasyR1-COE:/workspace easyr1_docker_efficient bash

new_enroot_start:
	enroot start --root --rw -m /home/hk-project-p0022560/lmu_eob1101/EasyR1-COE:/workspace easyr1_new bash
	
run_mllm:
	bash examples/mllm_scripts/run_qwen.sh

run_mllm_dev:
	bash examples/mllm_scripts/run_qwen_dev.sh

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
