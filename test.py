# import torch
import sqlite3
from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-small")

model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-small")
tokenizer.add_tokens(['sakura', 'marisa'])
model.resize_token_embeddings(len(tokenizer))
con = sqlite3.connect('D:/data/DanbooruDataset/danbooru.sqlite')
cur = con.cursor()
for row in cur.execute("""select id, tag_string_general, tag_string_character, tag_string_copyright, md5 from posts limit 5"""):
    print(row)

for row in cur.execute("""select max(id) from posts"""):
    print(row[0])

print(tokenizer.encode('minase akito'))
print(tokenizer.decode([3519, 9, 7, 15, 3, 11259, 235, 1]))
