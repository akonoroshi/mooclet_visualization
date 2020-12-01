from numpy.random import choice
# arguments to policies:

# variables: list of variable objects, can be used to retrieve related data
# context: dict passed from view, contains current user, course, quiz, question context

def thompson_sampling_placeholder(variables, context):
	return choice(context['mooclet'].version_set.all())

def thompson_sampling(variables, context, simulation_data):
	'''
	Compute alpha and beta of the simulation data.

	variables are ignored; just pass an empty list 
	context is the following json :
	   {
	   'policy_parameters':
	       {
	       'outcome_variable_name':<name of the outcome variable',
	       'max_rating': <maximum value of the outcome variable>,
	       'prior':
	           {'success':<prior success value>},
	           {'failure':<prior failure value>},
	       },
		'versions': [<list of integer versions>],
		'mooclet': <the id of your mooclet>
	   }

	'''
	versions = context['versions']
	policy_parameters = context["policy_parameters"]

	prior_success = policy_parameters['prior']['success']

	prior_failure = policy_parameters['prior']['failure']
	outcome_variable_name = policy_parameters['outcome_variable_name']
	max_rating = policy_parameters['max_rating']

	alpha_beta = {}

	for version in versions:
		student_ratings = simulation_data[simulation_data['version_id_{}'.format(context['mooclet'])] == int(version)][outcome_variable_name]

		if student_ratings.empty:
			rating_average = 0
			rating_count = 0
		else:
			rating_count = student_ratings.size
			rating_average = student_ratings.mean()

		successes = (rating_average * rating_count) + prior_success
		failures = (max_rating * rating_count) - (rating_average * rating_count) + prior_failure

		alpha_beta['successes_{}'.format(version)] = successes
		alpha_beta['failures_{}'.format(version)] = failures

	return alpha_beta
