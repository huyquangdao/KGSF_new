import json
import pickle

train_movies = json.load(open('train_movies.json'))
valid_movies = json.load(open('valid_movies.json'))
test_movies = json.load(open('test_movies.json'))


all_movies = list(set(train_movies + valid_movies + test_movies))

print(len(all_movies))

movie_ids = pickle.load(open('data/movie_ids.pkl','rb'))

with open('data/id2entity.pkl','rb') as f:
    id2entity = pickle.load(f)

# count = 0

# for movie_id in movie_ids:
#     print(type(movie_id))
#     print(id2entity[str(movie_id)])

# print(count)


print(movie_ids[:10])

print(list(id2entity.keys())[:10])