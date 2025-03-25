import os
from transformers import AutoTokenizer
from collections import defaultdict

import tiktoken
from tokenizers.decoders import ByteLevel

# Получение tiktoken токенизатора
tokenizer = tiktoken.get_encoding("o200k_base")

def get_vocabluary(path):
    vocab = []

    with open(path, 'r') as file:
        for line in file:
            vocab.append(line.replace('\n', '').encode("utf-8"))

    return vocab

def get_text_in_target(src_path):
    txt_files = []
    target_texts = []

    #i = 0

    # Поиск всех файлов txt
    for root, _, files in os.walk(src_path):
        for file in files:
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))

                #i = i + 1

                #if i > 100:
                    #break

        #if i > 100:
            #break

    #print(txt_files)
   
   # Получение всех текстов из файлов txt
    for file_name in txt_files:
        with open(file_name, 'r', encoding="utf-8") as file:
            target_texts.append(file.read().replace('\n', ''))
   
    #print(target_texts)
    return target_texts

def compute_pair_freqs(splits, word_freqs):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs

def merge_pair(a, b, splits, word_freqs):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits

def tokenize(text):
    return tokenizer.encode(text)

def untokenize(tokens):
    return tokenizer.decode(tokens)

# Создать словарь байтов из датасета по path
def create_vocab(path):
    vocab = [] #get_vocabluary("./src/vocabluary.txt")

    corpus = get_text_in_target(path) # "F:/asr_public_phone_calls_1/0/" путь к датасету, где файлы формата txt

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    word_freqs = defaultdict(int)

    for text in corpus:
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
        new_words = [word for word, offset in words_with_offsets]
        for word in new_words:
            word_freqs[word] += 1

    # Для просмотра байтов в человеческие символы
    # decoder = ByteLevel()

    # for element in list(word_freqs.keys()):
    #     print(decoder.decode([element]))

    splits = {word: [c for c in word] for word in word_freqs.keys()}

    merges = {}

    vocab_size = 100000

    while len(vocab) < vocab_size:
        try:
            pair_freqs = compute_pair_freqs(splits, word_freqs)
            best_pair = ""
            max_freq = None
            for pair, freq in pair_freqs.items():
                if max_freq is None or max_freq < freq:
                    best_pair = pair
                    max_freq = freq
            splits = merge_pair(*best_pair, splits, word_freqs)
            merges[best_pair] = best_pair[0] + best_pair[1]
            vocab.append(best_pair[0] + best_pair[1])
        except:
            break

    # Для просмотра байтов в человеческие символы
    decoder = ByteLevel()

    with open("./src/new_vocab.txt", "w", encoding="utf-8") as file_o:
        for element in vocab:
            file_o.write(element)
            file_o.write("\n")
            
            print(decoder.decode([element]))

