import re
import os

def mask_sensitive_information(file_path, output_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    patterns = {
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b": "[EMAIL]",  # Mask email addresses
        r"\b\d{10}\b": "[PHONE_NUMBER]",  # Mask 10-digit phone numbers
        r"\b(?:Rs|₹|\$|EUR|USD|£)?\s?\d+(?:,\d{3})*(?:\.\d{1,2})?\b": "[AMOUNT]",  # Mask monetary values
        r"\b\d{1,2} [A-Za-z]+ \d{4}\b": "[DATE]",  # Mask dates
        r"\b[A-Z]{2}\d{2}\b": "[CODE]",  # Mask short codes (like product IDs)
        r"\b(?:order|invoice)\s?\d+\b": "[ORDER_ID]",  # Mask order or invoice numbers
    }

    for pattern, replacement in patterns.items():
        content = re.sub(pattern, replacement, content)

    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write(content)

input_files = {
    "gemini_responses.txt": "masked_gemini_responses.txt",
    "gpt4_responses.txt": "masked_gpt4_responses.txt",
    "llama_responses.txt": "masked_llama_responses.txt"
}

for input_file, output_file in input_files.items():
    mask_sensitive_information(input_file, output_file)
    print(f"Masked content saved to {output_file}")

print("All files processed and sensitive information masked.")