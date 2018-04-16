import os
import sys
import json
import numpy
import itertools

conf = {}
conf['model_name'] = "twitter_v21"

dir_json = sys.argv[1] if len(sys.argv) > 1 else 'json/'

model_name = conf['model_name']
if not os.path.exists(dir_json):
    os.makedirs(dir_json)

# Define below every parameter, or combination in a tuple
# Dataset parameters
ns_examples = [1]
ovs = 1

# CNN parameters    
learning_rates=[0.0001]
filter_tuples = (((1000, 3),(1000, 4),(1000, 5)),)
embedding_dims = 30
hidden_dims = 150
activations = ['relu']
dropouts = [0.25]

#SGD parameters
batch_size = 32
nb_epochs = 100


#Varying number of authors setup:
#training_mod = "var_author"
#dataset_dir = "/home/prasha/authorship-attribution/koppel_twitter/cleaned_all/varying_number_of_authors/"
#setups = ((100, "/home/prasha/authorship-attribution/no_bots_vauthors_list/100_selected_users.dat"),\
#         (200, "/home/prasha/authorship-attribution/no_bots_vauthors_list/200_selected_users.dat"),\
#         (500, "/home/prasha/authorship-attribution/no_bots_vauthors_list/500_selected_users.dat"),\
#         (1000, "/home/prasha/authorship-attribution/no_bots_vauthors_list/1000_selected_users.dat"))


head_file = "tmp/" + model_name + "_{conf}_setup:{setup}"
head_histories = "histories/koppel_" + model_name + "_{conf}_setup:{setup}"

#Varying training size setup:
training_mod = "var_training"
af = "/home/prasha/authorship-attribution/koppel_twitter/no_bots_vtweets.txt"
dataset_dir="/home/prasha/authorship-attribution/koppel_twitter/cleaned_all/varying_training_set_size/"
setups = (5, 10, 20, 50, 100)
ranges = ((0, 32), (33, 68), (69, 99), (100, 135), (136, 173), (174, 207), (208, 244), (245, 275), (276, 313), (314, 354))
# Uncomment the two lines below if you are working with bot and non bot authors, and comment af and ranges above
#ranges = ((0, 50), (51, 100), (101, 150), (151, 200), (201, 250), (251, 300), (301, 350), (351, 400), (401, 450), (451, 500))
#af = "/home/prasha/authorship-attribution/koppel_twitter/varying_tweets.txt"

combinations = list(itertools.product(learning_rates, filter_tuples, activations, dropouts, ns_examples))
num_jobs = range(len(combinations))
i=0
for lr, ft, act, dp, ne in combinations:
    conf['model_name'] = '{model_name}_{jobid}'.format(model_name=model_name, jobid=num_jobs[i])
    if os.path.isfile(conf['model_name']):
        print('%s already exists, skipping...' % (conf['model_name']))
        continue

    # Assign parameters
    conf['learning_rate'] = lr
    conf['filter_tuple'] = ft
    conf['activation'] = act
    conf['dropout'] = dp
    conf['n_examples'] = ne
    conf['ranges'] = ranges
    conf['ovs'] = ovs
    conf['authors_file'] = af
    conf['head_file'] = head_file
    conf['head_histories'] = head_histories
    conf['batch_size'] = batch_size
    conf['nb_epochs'] = nb_epochs
    conf['embedding_dims'] = embedding_dims
    conf['hidden_dims'] = hidden_dims
    conf['setups'] = setups
    conf['training_mod'] = training_mod
    conf['dataset_dir'] = dataset_dir

    with open(os.path.join(dir_json, conf['model_name'] + '.json'), 'w') as f:
        json.dump(conf, f, indent=4)
    i += 1
