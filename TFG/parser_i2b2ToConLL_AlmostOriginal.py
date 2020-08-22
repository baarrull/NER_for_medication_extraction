"""
This file is parsing i2b2 training data and annotating it with
the CoNLL BIO scheme, which has this form:
[word] [POS tag] [chunk tag] [NER tag]
"""

import os
from nltk import pos_tag, RegexpParser
import pandas as pd
import numpy as np

a_ids = []
e_ids = []

for filename in os.listdir("./n2c2 Data/data/annotations"):
    if filename[0] != ".":  # ignore hidden files
        a_ids.append(int(filename))
for filename in os.listdir("./n2c2 Data/data/entries"):
    if filename[0] != ".":
        e_ids.append(int(filename))


a_ids = sorted(a_ids)      #Se ordenan las annotations
e_ids = sorted(e_ids)      #Se ordenan las entries

intersection = list(set(a_ids) & set(e_ids))
if len(intersection) == len(a_ids):
    print("Success: all anotations have a corresponding entry.", len(intersection))

# ------------------- BUILD CORPORA --------------------
# build annotation and entry corpora

a_corpus = []
e_corpus = []

# only annotations and corresponding files
for file in a_ids:
    path = "./n2c2 Data/data/annotations/" + str(file)
    with open(path) as f:
        content = f.read().splitlines()
        a_corpus.append(content)

    path = "./n2c2 Data/data/entries/" + str(file)
    with open(path) as f:
        content = f.read().splitlines()
        e_corpus.append(content)

# ------------------ SET UP DATAFRAME --------------------- (!) UNDERSTAND

#  ["id", "row", "offset", "word", "POS", "chunk", "NER"]
entries_cols = ["id", "row", "offset", "word"]
entries_df = pd.DataFrame(columns=entries_cols)

#print(entries_df.head())

annotations_cols = ["id", "NER_tag", "row", "offset", "length"]
annotations_df = pd.DataFrame(columns=annotations_cols)

#print(annotations_df.head())

# ------------------ NUMBER OF ANNOTATIONS -------------------
med_count = 0
dosage_count = 0
mode_count = 0
freq_count = 0
dur_count = 0
reason_count = 0

for document in a_corpus:
    for line in document:
        if "m=\"nm\"" not in line:
            med_count += 1
        if "do=\"nm\"" not in line:
            dosage_count += 1
        if "mo=\"nm\"" not in line:
            mode_count += 1
        if "f=\"nm\"" not in line:
            freq_count += 1
        if "du=\"nm\"" not in line:
            dur_count += 1
        if "r=\"nm\"" not in line:
            reason_count += 1
"""
print("Medication annotations: ", med_count)
print("Dosage annotations: ", dosage_count)
print("Mode annotations: ", mode_count)
print("Frequency annotations: ", freq_count)
print("Duration annotations: ", dur_count)
print("Reason annotations: ", reason_count)
"""


# ----------- BUILD ANNOTATIONS DATA FRAME ---------------

annotations_df = pd.DataFrame(columns=annotations_cols)  # reset df
tmp_list = []

for i, document in enumerate(a_corpus):

    for row in document:
        row = row.split("||")

        for tag in row:
            tag = tag.split("=") #ex: tag = ['m', '"acetylsalicylic acid" 16:0 16:1']
            if ":" in tag[1]:
                tag_label = tag[0].lstrip(" ") #Remove white spaces before the tag
                tag_row_a = tag[1].split(" ")[-2:][0].split(":")[0]
                tag_row_b = tag[1].split(" ")[-2:][1].split(":")[0]

                # some annotations have non-standard formatting (losing 64 instances)
                try:
                    tag_offset_a = int(tag[1].split(" ")[-2:][0].split(":")[1])
                    tag_offset_b = int(tag[1].split(" ")[-2:][1].split(":")[1])
                    length = tag_offset_b - tag_offset_a + 1

                    # 1 row = 1 token with a tag
                    first = True
                    BIO_tag = "B-"
                    if length > 1 and tag_row_a == tag_row_b:
                        for offset in range(tag_offset_a, tag_offset_b + 1):
                            if first:
                                tag_label = BIO_tag + tag_label
                                first = False
                            else:
                                tag_label = tag_label.replace("B-", "I-")
                            tmp_list.append([a_ids[i], tag_label, tag_row_a, offset, 1])
                    # TODO: tags over line breaks
                    else:
                        tmp_list.append([a_ids[i], BIO_tag + tag_label, tag_row_a, tag_offset_a, length])
                except:
                    pass

annotations_df = pd.DataFrame(tmp_list, columns=annotations_cols)
annotations_df.reset_index(inplace=True)

annotations_df = annotations_df.drop(columns=["index", "length"]) #Remove columns
#print(annotations_df.shape)


# -------------------------- BUILD ENTRIES DATAFRAME -----------------------------------
#List of token modifications: - "|": ignored - "." removed from end of token
entries_df = pd.DataFrame(columns=entries_cols)  # reset df
tmp_list = []

for doc_i, document in enumerate(e_corpus):
    tmp_list.append([0, 0, 0, "-DOCSTART-"])
    tmp_list.append([0, 0, 0, "-EMPTYLINE-"])

    for row_i, row in enumerate(document):
        row_split = row.split(" ")
        for word_i, word in enumerate(row_split):
            word = word.rstrip(".")  # strip "." from end of word
            word = word.replace("\t", "")
            word_id = a_ids[doc_i]
            word_row = row_i + 1  # 1-based indexing
            word_offset = word_i  # 0-based indexing

            if len(word) > 0 and "|" not in word:
                tmp_list.append([word_id, word_row, word_offset, word])

    tmp_list.append([0, 0, 0, "-EMPTYLINE-"])

entries_df = pd.DataFrame(tmp_list, columns=entries_cols)

#print(annotations_df.head())
print(entries_df.head())

ner_counter = [1 for i in annotations_df["NER_tag"] if "B-" in i]
print(len(ner_counter), "named entities")

# --------------- JOING ENTRIES AND ANNOTATIONS ----------------------- (!)

# ensure correct dtypes
annotations_df[['id', 'row', 'offset']] = annotations_df[['id', 'row', 'offset']].apply(pd.to_numeric)
annotations_df['NER_tag'] = annotations_df["NER_tag"].astype(str)
entries_df[['id', 'row', 'offset']] = entries_df[['id', 'row', 'offset']].apply(pd.to_numeric)
entries_df["word"] = entries_df["word"].astype(str)

result_df = pd.merge(entries_df, annotations_df, how="left", on=['id', 'row', 'offset'])

# replace NaNs with "O"
#print("columns with missing data:\n", result_df.isna().any())
result_df = result_df.fillna("O")
#print("columns with missing data:\n", result_df.isna().any())

result_df = result_df.drop(columns=["id", "offset"]) #result_df = result_df.drop(columns=["id", "row", "offset"])
print("Result: ", result_df.head())

# ------------------- POS tagger ------------------------------------ (!)
from nltk.chunk.regexp import RegexpChunkParser, ChunkRule, RegexpParser
from nltk.tree import Tree

text = result_df["word"].tolist()
text_pos = pos_tag(text)
text_pos_list = [i[1] for i in text_pos]

result_df["POS_tag"] = text_pos_list

# ----------------- ConLL chunk tagger --------------------------------
""""
text_test = "EU rejects German call to boycott British lamb.".split(" ")
text_pos_test = pos_tag(text_test)
print(text_pos_test)
"""

#NOUN PHRASES
rule_0 = ChunkRule("<DT>?<JJ.*>*<NN.*>+", "More complete chunk NP sequences")
chunk_parser_np = RegexpChunkParser([rule_0],chunk_label='NP')
chunk_result_tree_np = chunk_parser_np.parse(text_pos)

chunk_tag_np = []

for i in chunk_result_tree_np:
    if isinstance(i, Tree):
        for j in range(0, len(i)):
            if j == 0:
                # print("B-" + i.label())
                chunk_tag_np.append("B-" + i.label())
            else:
                chunk_tag_np.append("I-" + i.label())
                # print("I-" + i.label())
    else:
        # print("O")
        chunk_tag_np.append("O")

#print(len(chunk_tag_np) == result_df.shape[0])  # check that chunk col has same length

#VERB PHRASES

rule_1 = ChunkRule("<VBD|IN|\.>", "Verb phrases")
chunk_parser_vp = RegexpChunkParser([rule_1],chunk_label='VP')
chunk_result_tree_vp = chunk_parser_vp.parse(text_pos)

chunk_tag_vp = []

for i in chunk_result_tree_vp:
    if isinstance(i, Tree):
        for j in range(0, len(i)):
            if j == 0:
                # print("B-" + i.label())
                chunk_tag_vp.append("B-" + i.label())
            else:
                chunk_tag_vp.append("I-" + i.label())
                # print("I-" + i.label())
    else:
        # print("O")
        chunk_tag_vp.append("O")

#len(chunk_tag_np) == result_df.shape[0] == len(chunk_tag_vp)

# augment chunk tags with verb phrase tags
for i, entry in enumerate(chunk_tag_np):
    if entry == "O":
        chunk_tag_np[i] = chunk_tag_vp[i]

# There are not prepositional phrases
result_df["chunk_tag"] = chunk_tag_np
result_df = result_df[['row', 'word', 'POS_tag', 'chunk_tag', 'NER_tag']]  # order columns
result_df[['row', 'word', 'POS_tag', 'chunk_tag', 'NER_tag']] = result_df[['row', 'word', 'POS_tag', 'chunk_tag', 'NER_tag']].astype(str)

# -------------------- DATA SPLIT -----------------------------
print(result_df.shape)
result_df = result_df.reindex()
# find indices of new documents
result_df[result_df["word"] == "-DOCSTART-"].index.values.tolist()

train = 202062
dev = 247618
result_train_df = result_df.iloc[:train]
result_dev_df = result_df.iloc[train:dev]
result_test_df = result_df.iloc[dev:]

print(result_test_df.tail())

print("train shape ", result_train_df.shape)
print("dev shape ", result_dev_df.shape)
print("test shape ", result_test_df.shape)

result_df.to_csv("result_df_NER_POS_chunk.csv")

# ------------------------- WRITE TO TXT --------------------------

np.savetxt("train.txt", result_train_df.values, fmt="%s")
np.savetxt("valid.txt", result_dev_df.values, fmt="%s")
np.savetxt("test.txt", result_test_df.values, fmt="%s")
