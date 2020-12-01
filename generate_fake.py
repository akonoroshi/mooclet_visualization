import pandas as pd
import numpy as np
from scipy.stats import bernoulli
import random
from policies_offline import thompson_sampling
from assign_prob import compute_assign_prob_TS_2
from datetime import datetime

if __name__ == "__main__":
	data_path = './data/fake.csv'
	outcome_variable_name = 'fake_outcome'
	mooclet = 145
	versions = [5197, 5198, 5199]
	version_success_prob = [0.5, 0.8, 0.9]
	policies = [1, 3]
	policy_prob = [0.2, 0.8]
	num_data_point = 1000

	###################### You don't have to change below ######################

	context = {'policy_parameters':
				{'outcome_variable_name': outcome_variable_name, 'max_rating': 1, 'prior':{'success': 1, 'failure': 1}}, 
			'mooclet': mooclet, 'versions': versions}
	
	df = pd.DataFrame(columns=['version_id_{}'.format(mooclet), 'policy_id_{}'.format(mooclet), outcome_variable_name, 'timestamp_{}'.format(mooclet)])
	for i in range(num_data_point):
		# Choose policy and simulate accordingly
		policy = random.choices(policies, policy_prob)[0]
		if policy == 1:
			version_idx = random.choices(range(len(versions)))
		else: # Assuming the policy is TS
			simulation_data = df[:i]
			simulation_data = simulation_data[simulation_data['policy_id_{}'.format(mooclet)] == 3]
			alpha_beta = thompson_sampling([], context, simulation_data)
			alphas = []
			betas = []
			for version in versions:
				alphas.append(alpha_beta['successes_{}'.format(version)])
				betas.append(alpha_beta['failures_{}'.format(version)])
			version_idx = random.choices(range(len(versions)), list(compute_assign_prob_TS_2(alphas, betas, 100, versions).values()))
		
		version = versions[version_idx[0]]
		outcome = bernoulli.rvs(version_success_prob[version_idx[0]])
		data = {'version_id_{}'.format(mooclet): version, 'policy_id_{}'.format(mooclet): policy, \
			outcome_variable_name: outcome, 'timestamp_{}'.format(mooclet): datetime.now()}
		df = df.append(data, ignore_index=True)

	# Save the results
	df.to_csv(data_path)
