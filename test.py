import json
from collections import defaultdict
import pickle
from tqdm import tqdm

train_movies = json.load(open('train_movies.json'))
valid_movies = json.load(open('valid_movies.json'))
test_movies = json.load(open('test_movies.json'))

def _edge_list(kg, n_entity, hop):
    edge_list = []
    for h in range(hop):
        for entity in range(n_entity):
            # add self loop
            # edge_list.append((entity, entity))
            # self_loop id = 185
            edge_list.append((entity, entity, 185))
            if entity not in kg:
                continue
            for tail_and_relation in kg[entity]:
                if (
                    entity != tail_and_relation[1] and tail_and_relation[0] != 185
                ):  # and tail_and_relation[0] in EDGE_TYPES:
                    edge_list.append(
                        (entity, tail_and_relation[1], tail_and_relation[0])
                    )
                    edge_list.append(
                        (tail_and_relation[1], entity, tail_and_relation[0])
                    )

    relation_cnt = defaultdict(int)
    relation_idx = {}
    for h, t, r in edge_list:
        relation_cnt[r] += 1
    for h, t, r in edge_list:
        if relation_cnt[r] > 1000 and r not in relation_idx:
            relation_idx[r] = len(relation_idx)

    return [
        (h, t, relation_idx[r]) for h, t, r in edge_list if relation_cnt[r] > 1000
    ], len(relation_idx)

def extract_one_hop_neighbors(entities, edge_lists, number_of_nodes):
    total_nodes = []
    for entity in tqdm(entities):
        s_edges = [x[1] for x in edge_lists if x[0] == entity and x[0] != x[1]]
        total_nodes.extend(s_edges)
    
    total_nodes = list(set(total_nodes + entities))
    
    print('number of nodes in one hop neighbors: ', len(total_nodes) / number_of_nodes * 100)


all_movies = list(set(train_movies + valid_movies + test_movies))
print(len(all_movies))

with open('data/id2entity.pkl','rb') as f:
    id2entity = pickle.load(f)

with open('/home/huy/Home/Test/tutorial/tutorial/data/entity2entityId.pkl','rb') as f:
    entity2entityId = pickle.load(f)

all_nodes_in_kg = []
kg = pickle.load(open("data/subkg.pkl", "rb"))

for k, v in kg.items():
    all_nodes_in_kg.append(k)
    all_s_nodes = [x[0] for x in v]
    all_e_nodes = [x[1] for x in v]
    all_nodes_in_kg.extend(all_s_nodes)
    all_nodes_in_kg.extend(all_e_nodes)

print(len(set(all_nodes_in_kg)))

count = 0
all_transformerd_movies_ids = []
for movie in all_movies:
    try:
        entity = id2entity[int(movie)]
        id = entity2entityId[entity]
        count +=1
        all_transformerd_movies_ids.append(id)
    except:
        pass

# print('number of handled movies: ',count)

# all_mentioned_entities = json.load(open('/home/huy/Home/KGSF_new/all_non_items.json'))
# count_all_mentioned_non_items =  len(set(all_mentioned_entities)) - len(set(all_transformerd_movies_ids))

# all_non_item_entities = list(set(all_mentioned_entities) - set(all_transformerd_movies_ids))

# print('percentage of mentioned movie in kg: ', count / len(entity2entityId) * 100)
# print('percentage of mentioned movied linked to kg: ',count/ 6924 * 100)
# print('percentage of mentioned non-item entities in kg: ', count_all_mentioned_non_items/ len(entity2entityId) * 100)

# edge_list, n_relation = _edge_list(kg, 64368, hop=2)
# extract_one_hop_neighbors(all_transformerd_movies_ids, edge_list, len(entity2entityId))
# extract_one_hop_neighbors(all_non_item_entities, edge_list, len(entity2entityId))

with open('/home/huy/Home/KGSF_new/all_src_des_pairs.json','r') as f:
    temp = json.load(f)

print(len(temp))