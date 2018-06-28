#!/bin/bash

mosesdecoder="ext-libs/mosesdecoder"
input_file=$1
output_file=$2
threads=20
lang=${3:-ru}

echo "Input file: $input_file"
echo "Output file: $output_file"
echo "Threads: $threads"
echo "Lang: $lang"

cat $input_file | \
    $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $lang | \
    $mosesdecoder/scripts/tokenizer/tokenizer.perl -threads $threads -l $lang > \
    $output_file
