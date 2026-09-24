#!/bin/bash

#####################################################################################################################################################################
# xView2 - Batch/Sequential Inference Wrapper                                                                                                                       #
# Based on the original xView2 single-pair inference script (Copyright 2019 Carnegie Mellon University, MIT (SEI)-style license).                                  #
# This version loops over an entire folder of pre/post-disaster image pairs, runs inference sequentially, and saves each output indexed in order.                  #
#####################################################################################################################################################################

set -euo pipefail

function trap_ctrlc ()
{
    echo "Ctrl-C or Error caught...performing clean up check /tmp/inference.log"
    if [ -d /tmp/inference ]; then
        rm -rf /tmp/inference
    fi
    exit 99
}

trap "trap_ctrlc" 2 9 13 3

help_message () {
    printf "${0}: Runs the polygonization in batch/sequential inference mode\n"
    printf "\t-x: path to xview-2 repository\n"
    printf "\t-d: /full/path/to/input/folder (contains *_pre_disaster.* and *_post_disaster.* pairs)\n"
    printf "\t-o: /full/path/to/output/folder (one result image per pair, saved in index order)\n"
    printf "\t-l: path/to/localization_weights\n"
    printf "\t-c: path/to/classification_weights\n"
    printf "\t-e: /path/to/virtual/env/activate\n"
    printf "\t-y: continue with local environment and without interactive prompt\n"
    printf "\t-s: suffix pattern used to identify pre-disaster files (default: _pre_disaster)\n\n"
}

inference_base="/tmp/inference"
LOGFILE="/tmp/inference_log"
XBDIR=""
input_dir=""
output_dir=""
virtual_env=""
localization_weights=""
classification_weights=""
continue_answer="n"
pre_suffix="_pre_disaster"

if [ "$#" -lt 10 ]; then
    help_message
    exit 1
fi

while getopts "d:o:x:l:e:c:s:hy" OPTION
do
    case $OPTION in
        h)
            help_message
            exit 0
            ;;
        y)
            continue_answer="y"
            ;;
        d)
            input_dir="$OPTARG"
            ;;
        o)
            output_dir="$OPTARG"
            ;;
        x)
            XBDIR="$OPTARG"
            virtual_env="$XBDIR/bin/activate"
            ;;
        l)
            localization_weights="$OPTARG"
            ;;
        c)
            classification_weights="$OPTARG"
            ;;
        e)
            virtual_env="$OPTARG"
            ;;
        s)
            pre_suffix="$OPTARG"
            ;;
        ?)
            help_message
            exit 0
            ;;
    esac
done

if [ -z "$input_dir" ] || [ -z "$output_dir" ] || [ -z "$XBDIR" ]; then
    echo "Error: -d, -o, dan -x wajib diisi."
    help_message
    exit 1
fi

mkdir -p "$inference_base"
mkdir -p "$output_dir"

if ! [ -f "$LOGFILE" ]; then
    touch "$LOGFILE"
fi

printf "==========\n" >> "$LOGFILE"
echo `date +%Y%m%dT%H%M%S` >> "$LOGFILE"
printf "\n" >> "$LOGFILE"

# Source the virtual environment once, before the loop
if [ -f "$virtual_env" ]; then
    source "$virtual_env"
else
    if [ "$continue_answer" = "n" ]; then
        printf "Error: cannot source virtual environment  \n\tDo you have all the dependencies installed and want to continue? [Y/N]: "
        read continue_answer
        if [ "$continue_answer" == "N" ]; then
            exit 2
        fi
    fi
fi

# ---------------------------------------------------------------------------
# Function: run_inference
# Runs the full localization -> classification -> combine -> render pipeline
# for a single pre/post image pair.
# Args: $1=pre_image $2=post_image $3=output_file $4=label_temp_dir
# ---------------------------------------------------------------------------
run_inference () {
    local input="$1"
    local input_post="$2"
    local output_file="$3"
    local label_temp="$4"

    local input_image
    input_image=$(basename "$input")

    mkdir -p "$label_temp"

    cd "$XBDIR"/spacenet/inference/
    printf "Running localization\n"
    python3 ./inference.py --input "$input" --weights "$localization_weights" --mean "$XBDIR"/weights/mean.npy --output "$label_temp"/"${input_image%.*}".json >> "$LOGFILE" 2>&1
    printf "\n" >> "$LOGFILE"

    cd "$XBDIR"/model

    local disaster_post_file="$input_post"
    local polygons_dir="$inference_base"/output_polygons/"${input_image%.*}"
    mkdir -p "$polygons_dir"

    printf "Running classification\n"
    python3 ./process_data_inference.py --input_img "$disaster_post_file" --label_path "$label_temp"/"${input_image%.*}".json --output_dir "$polygons_dir" --output_csv "$inference_base"/"${input_image%.*}"_output.csv >> "$LOGFILE" 2>&1

    python3 ./damage_inference.py --test_data "$polygons_dir" --test_csv "$inference_base"/"${input_image%.*}"_output.csv --model_weights "$classification_weights" --output_json "$inference_base"/"${input_image%.*}"_classification_inference.json >> "$LOGFILE" 2>&1
    printf "\n" >> "$LOGFILE"

    printf "Formatting json and scoring image\n"
    python3 "$XBDIR"/utils/combine_jsons.py --polys "$label_temp"/"${input_image%.*}".json --classes "$inference_base"/"${input_image%.*}"_classification_inference.json --output "$inference_base"/"${input_image%.*}"_inference.json >> "$LOGFILE" 2>&1
    printf "\n" >> "$LOGFILE"

    printf "Finalizing output file\n"
    python3 "$XBDIR"/utils/inference_image_output.py --input "$inference_base"/"${input_image%.*}"_inference.json --background "$disaster_post_file" --output "$output_file" >> "$LOGFILE" 2>&1
}

# ---------------------------------------------------------------------------
# Build the ordered list of pre-disaster files, then process each pair
# sequentially, saving outputs with a zero-padded index prefix so the
# result order matches the input order.
# ---------------------------------------------------------------------------
mapfile -t pre_files < <(find "$input_dir" -maxdepth 1 -type f -iname "*${pre_suffix}*" | sort)

total=${#pre_files[@]}
if [ "$total" -eq 0 ]; then
    echo "Error: tidak ada file yang cocok dengan pola '*${pre_suffix}*' di $input_dir"
    exit 3
fi

echo "Ditemukan $total pasangan pre/post-disaster image. Memulai inference sequensial..."

index=0
for pre_file in "${pre_files[@]}"; do
    padded_index=$(printf "%04d" "$index")
    base_name=$(basename "$pre_file")
    post_file="$input_dir/${base_name/${pre_suffix}/_post_disaster}"

    if [ ! -f "$post_file" ]; then
        echo "[$padded_index] Peringatan: pasangan post-disaster tidak ditemukan untuk $base_name, dilewati."
        index=$((index + 1))
        continue
    fi

    output_file="$output_dir/${padded_index}_${base_name%.*}_output.png"
    label_temp="$inference_base/${base_name%.*}/labels"

    echo "[$padded_index/$((total - 1))] Memproses: $base_name"
    run_inference "$pre_file" "$post_file" "$output_file" "$label_temp"
    echo "[$padded_index/$((total - 1))] Selesai -> $output_file"

    index=$((index + 1))
done

# Cleaning up temporary working directory
rm -rf "$inference_base"

printf "==========\n" >> "$LOGFILE"
printf "Semua proses batch inference selesai! Output tersimpan di: %s\n" "$output_dir"