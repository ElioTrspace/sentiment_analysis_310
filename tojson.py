import numpy as np
import pandas as pd
import itertools
from cleaning import preprocess_dataset

def chat_generator(filename):
    with open(filename, encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                if ':' in line:
                    speaker, message = line.split(':', 1)
                    yield speaker.strip(), message.strip()

gen = chat_generator("human_chat.txt")

df = pd.DataFrame(gen, columns=["speaker", "body"])
# print(df)
df.to_json('data_2.json.gz', orient = 'records', lines = True, compression = 'gzip')

reddit_df = preprocess_dataset('all_subreddit_comments.json.gz')
joined_df = pd.concat([reddit_df, df])
joined_df = joined_df.fillna(value = {'subreddit' : 'notReddit'})
(joined_df).to_json('joined_data.json.gz', orient = 'records', lines = True, compression = 'gzip')
