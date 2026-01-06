# CUDA_VISIBLE_DEVICES=1,2,3 python main_eval.py --dataset_name nasa_repo_code_benchmark_v0.1
# CUDA_VISIBLE_DEVICES=1,2,3 python main_eval.py --dataset_name codesearchnet_testset_benchmark_v0.1
# CUDA_VISIBLE_DEVICES=1,2,3 python main_eval.py --dataset_name codesearchnet_testset_benchmark_v0.2

# python main_eval.py --dataset_name nasa_sde_ir_v3
# python main_eval.py --dataset_name nasa_smd_ir
# python main_eval.py --dataset_name nanobeir


CUDA_VISIBLE_DEVICES=0,1,2,3 python main_eval.py --dataset_name code_repo_search_benchmark_v1 
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_eval.py --dataset_name nanobeir 
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_eval.py --dataset_name nasa_sde_ir_20251024_v5
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_eval.py --dataset_name nasa_sde_ir_v3 
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_eval.py --dataset_name nasa_smd_ir 