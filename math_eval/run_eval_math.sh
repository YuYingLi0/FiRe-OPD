MODEL="Qwen3-4B"
MODEL_PATH="Qwen/Qwen3-4B"
MODEL_NAME=$MODEL

echo $MODEL_PATH
echo $MODEL_NAME

# Create output directories if they don't exist
mkdir -p ./eval_outputs/aime24
mkdir -p ./eval_outputs/aime25
mkdir -p ./eval_outputs/hmmt25_feb
mkdir -p ./eval_outputs/hmmt25_nov
mkdir -p ./eval_outputs/math500
mkdir -p ./eval_outputs/minervamath
mkdir -p ./eval_outputs/olympiadbench
mkdir -p ./eval_outputs/amc2023

# aime24
CUDA_VISIBLE_DEVICES=0,1 python3 eval_math.py \
    --input_file ../data/aime24/test.jsonl \
    --model_path $MODEL_PATH  \
    --output_file ./eval_outputs/aime24/${MODEL_NAME}.jsonl \
    --max_tokens 16384 \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_num_seqs 256 \
    --n 32 \
    --begin_idx -1 \
    --end_idx -1 --seed 42 &


# aime25
CUDA_VISIBLE_DEVICES=2,3 python3 eval_math.py \
    --input_file ../data/aime25/test.jsonl \
    --model_path $MODEL_PATH  \
    --output_file ./eval_outputs/aime25/${MODEL_NAME}.jsonl \
    --max_tokens 16384 \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_num_seqs 256 \
    --n 32 \
    --begin_idx -1 \
    --end_idx -1 --seed 42 &



# hmmt25-Feb
CUDA_VISIBLE_DEVICES=4,5 python3 eval_math.py \
    --input_file ../data/hmmt25_feb/test.jsonl \
    --model_path $MODEL_PATH  \
    --output_file ./eval_outputs/hmmt25_feb/${MODEL_NAME}.jsonl \
    --max_tokens 16384 \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_num_seqs 256 \
    --n 32 \
    --begin_idx -1 \
    --end_idx -1 --seed 42 &



# hmmt25-Nov
CUDA_VISIBLE_DEVICES=6,7 python3 eval_math.py \
    --input_file ../data/hmmt25_nov/test.jsonl \
    --model_path $MODEL_PATH  \
    --output_file ./eval_outputs/hmmt25_nov/${MODEL_NAME}.jsonl \
    --max_tokens 16384 \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_num_seqs 256 \
    --n 32 \
    --begin_idx -1 \
    --end_idx -1 --seed 42 &


# math500
CUDA_VISIBLE_DEVICES=0,1 python3 eval_math.py \
    --input_file ../data/math500/test.jsonl \
    --model_path $MODEL_PATH  \
    --output_file ./eval_outputs/math500/${MODEL_NAME}.jsonl \
    --max_tokens 16384 \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_num_seqs 256 \
    --n 32 \
    --begin_idx -1 \
    --end_idx -1 --seed 42 &


# minervamath
CUDA_VISIBLE_DEVICES=2,3 python3 eval_math.py \
    --input_file ../data/minervamath/test.jsonl \
    --model_path $MODEL_PATH  \
    --output_file ./eval_outputs/minervamath/${MODEL_NAME}.jsonl \
    --max_tokens 16384 \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_num_seqs 256 \
    --n 32 \
    --begin_idx -1 \
    --end_idx -1 --seed 42 &


# olympiadbench
CUDA_VISIBLE_DEVICES=4,5 python3 eval_math.py \
    --input_file ../data/olympiadbench/test.jsonl \
    --model_path $MODEL_PATH  \
    --output_file ./eval_outputs/olympiadbench/${MODEL_NAME}.jsonl \
    --max_tokens 16384 \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_num_seqs 256 \
    --n 32 \
    --begin_idx -1 \
    --end_idx -1 --seed 42 &


# amc2023
CUDA_VISIBLE_DEVICES=6,7 python3 eval_math.py \
    --input_file ../data/amc2023/test.jsonl \
    --model_path $MODEL_PATH  \
    --output_file ./eval_outputs/amc2023/${MODEL_NAME}.jsonl \
    --max_tokens 16384 \
    --temperature 1.0 \
    --top_p 1.0 \
    --max_num_seqs 256 \
    --n 32 \
    --begin_idx -1 \
    --end_idx -1 --seed 42 &

wait
echo "Model $MODEL_NAME done!"
