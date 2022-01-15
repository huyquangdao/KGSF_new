import random
import numpy as np
import pandas as pd
from collections import defaultdict

def randomly_sample_generated_text(file_path, indices):
    with open(file_path,'r') as f:
        lines = f.readlines()
        sampled_lines = [lines[x].strip() for x in indices]
    return sampled_lines

def reformulate_context(context, movie_dict):
    context = replace_movie_token_by_movie_name(context, movie_dict)
    context = context.replace("_split_"," | ")
    return context

def replace_movie_token_by_movie_name(text, movie_dict):
    if "@" in text:
        temp = []
        for token in text.split():
            if "@" in token:
                movie_id = token.replace("@","").strip()
                movie_name = movie_dict[movie_id]
                temp.append(movie_name)
            else: 
                temp.append(token)
        text = " ".join(temp)
        return text
    return text

def read_movie_dict(movie_mentions_path):
    movie_dict = {}
    df = pd.read_csv(movie_mentions_path)
    for _, row in df.iterrows():
        movie_id, movie_name = row[0], row[1]
        movie_dict[str(movie_id)] = movie_name
    return movie_dict


def generate_human_evaluation_data(file_name):

    movie_dict = read_movie_dict("movies_with_mentions.csv")
    context_path = "context_test.txt"
    kgsf_path = "kgsf_gen_output_file.txt"
    revcore_path = "revcore_gen_output_file.txt"
    our_model_path = "our_model_gen_output_file.txt"
    golden_path = "golden_test.txt"

    total_number_of_samples = 7087
    total_number_of_sampled_examples = 7087
    # indices = np.random.permutation(total_number_of_samples)[:total_number_of_sampled_examples]
    indices = list(range(total_number_of_sampled_examples))

    contexts = randomly_sample_generated_text(context_path, indices)
    our_model_responses = randomly_sample_generated_text(our_model_path, indices)
    kgsf_respones = randomly_sample_generated_text(kgsf_path, indices)
    revcore_respones = randomly_sample_generated_text(revcore_path, indices)
    golden_respones = randomly_sample_generated_text(golden_path, indices)

    count_dict = defaultdict(int)
    total_cold_start_cases = 0

    with open(file_name, 'w') as f:
        for c, k_r, r_r, o_r,g_r in list(zip(contexts, kgsf_respones, revcore_respones, our_model_responses, golden_respones)):
            # print('context: ',reformulate_context(c, movie_dict))
            # print('kgsf: ', replace_movie_token_by_movie_name(k_r, movie_dict))
            # print('revcore: ',replace_movie_token_by_movie_name(r_r, movie_dict))
            # print('our model: ',replace_movie_token_by_movie_name(o_r, movie_dict))
            # print('---------------------------')
            try:
                if  "@" not in c:
                    
                    total_cold_start_cases +=1

                    if "@" in k_r:
                        count_dict['k_r'] +=1
                    if "@" in r_r:
                        count_dict['r_r'] +=1
                    if "@" in o_r:
                        count_dict['o_r'] +=1

                    f.write(f'context: {reformulate_context(c, movie_dict)}')
                    f.write('\n')
                    f.write(f'response 1: {replace_movie_token_by_movie_name(k_r, movie_dict)}')
                    f.write('\n')
                    f.write(f'response 2: {replace_movie_token_by_movie_name(r_r, movie_dict)}')
                    f.write('\n')
                    f.write(f'response 3: {replace_movie_token_by_movie_name(o_r, movie_dict)}')
                    f.write('\n')
                    f.write(f'response 4: {replace_movie_token_by_movie_name(g_r, movie_dict)}')
                    f.write('\n')
                    f.write('--' * 100)
                    f.write('\n')
            except:
                pass
    print(count_dict)

    with open('cold_start_item_ratio_results.txt', 'w') as f:
        f.write(f"kgsf: {count_dict['k_r']/ total_cold_start_cases}")
        f.write('\n')
        f.write(f"revcore: {count_dict['r_r']/total_cold_start_cases}")
        f.write('\n')
        f.write(f"ours: {count_dict['o_r']/total_cold_start_cases}")
        f.write('\n')

generate_human_evaluation_data('human_eval.txt')
