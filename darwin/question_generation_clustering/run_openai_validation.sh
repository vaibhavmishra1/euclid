#!/bin/bash

# Example usage script for OpenAI validation

# Make sure to set your OpenAI API key

# Test with first 10 questions
    `    python validate_with_openai_async.py \
            --input_file balanced_questions__darwin_iter2_openai_validated.json \
            --output_file balanced_questions__darwin_iter2_openai_validated_2.json \
            --model gpt-4o \
            --concurrency 5
`
echo "Test validation complete! Check output file."

# To process all questions (remove --max_questions):
# python validate_with_openai.py \
#     --input_file balanced_questions_darwin_iter2_evaluated_fixed.json \
#     --output_file balanced_questions_darwin_iter2_openai_validated.json \
#     --model gpt-4o
