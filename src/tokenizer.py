import os
from transformers import AutoTokenizer
from collections import defaultdict

def get_text_in_target(src_path):
    txt_files = []
    target_texts = []

    # Поиск всех файлов txt
    for root, _, files in os.walk(src_path):
        for file in files:
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))

    #print(txt_files)
   
   # Получение всех текстов из файлов txt
    for file_name in txt_files:
        with open(file_name, 'r', encoding="utf-8") as file:
            target_texts.append(file.read().replace('\n', ''))
   
    #print(target_texts)
    return target_texts

corpus = get_text_in_target("F:/asr_public_phone_calls_1/0/")


tokenizer = AutoTokenizer.from_pretrained("gpt2")
word_freqs = defaultdict(int)

for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

print(word_freqs)

print(1)