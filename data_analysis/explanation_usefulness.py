import numpy as np
import pandas as pd
import os
import sys
from util import UserPerformance
from util import find_valid_users, get_user_conditions, load_answers, get_user_question_order, read_decisions, calc_miscalibration, calc_user_reliance_measures, calc_explanation_usefulness
from util import calc_ATI_scale, calc_propensity_to_trust, calc_trust_in_automation, check_user_condition
from collections import Counter

if __name__ == "__main__":
	filename = "all_valid_data.csv"
	valid_users, approved_users = find_valid_users(filename, 4)
	user_condition_dict = get_user_conditions(filename, valid_users)
	answer_dict = load_answers()
	user_question_order = get_user_question_order(filename, valid_users)
	usertask_dict = read_decisions(filename, valid_users)
	self_assessment_first, self_assessment_second, other_assessment_first, other_assessment_second, survey_percetage_first, survey_percetage_second = calc_miscalibration(filename, valid_users)
	user_trust_first, user_trust_second = calc_trust_in_automation(filename, valid_users)
	user_ATI_scale = calc_ATI_scale(filename, valid_users)
	user_ptt_scale = calc_propensity_to_trust(filename, valid_users)

	users_with_explanation = user_condition_dict["no tutorial, with xai"] | user_condition_dict["with tutorial, with xai"]

	user_usefulness_dict = calc_explanation_usefulness(filename, users_with_explanation)

	variable_dict = {
		"user_id": [],
		"ATI": [],
		"TiA-Propensity": [],
		"TiA-Trust-first": [],
		"TiA-Trust-second": [],
		"usefulness": []
	}
	for user in users_with_explanation:
		variable_dict["user_id"].append(user)
		variable_dict["TiA-Propensity"].append(user_ptt_scale[user])
		variable_dict["ATI"].append(user_ATI_scale[user])
		variable_dict["usefulness"].append(user_usefulness_dict[user])
		variable_dict["TiA-Trust-first"].append(user_trust_first[user])
		variable_dict["TiA-Trust-second"].append(user_trust_second[user])
	ct_dict = dict(Counter(variable_dict["usefulness"]))
	print(len(variable_dict["usefulness"]))
	data = []
	for ct in range(1, 6):
		ratio = ct_dict[ct] / len(variable_dict["usefulness"])
		data.append(ratio)
		print(ct, "%.3f"%(ratio))


	import matplotlib.pyplot as plt
	import seaborn as sns

	#define data
	labels = ['Not Helpful', 'Very Slightly Helpful', 'Slightly Helpful', 'Helpful', 'Very Helpful']

	#define Seaborn color palette to use
	# colors = sns.color_palette('bright')[0:5]
	colors = sns.color_palette('pastel')[0:5]

	#create pie chart
	plt.pie(data, labels = labels, colors = colors, autopct='%.1f%%', textprops={'fontsize': 14})
	plt.show()