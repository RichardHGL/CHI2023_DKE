import numpy as np
import pandas as pd
import os
import sys
from util import UserPerformance
from util import find_valid_users, get_user_conditions, load_answers, get_user_question_order, read_decisions, calc_miscalibration, calc_user_reliance_measures, calc_trust_in_automation
from scipy.stats import kstest

# filename = "data_ntut_nxai_60participants.csv"
filename = "all_valid_data.csv"
valid_users, approved_users = find_valid_users(filename, 4)
user_condition_dict = get_user_conditions(filename, valid_users)
answer_dict = load_answers()
user_question_order = get_user_question_order(filename, valid_users)
usertask_dict = read_decisions(filename, valid_users)
self_assessment_first, self_assessment_second, other_assessment_first, other_assessment_second, survey_percetage_first, survey_percetage_second = calc_miscalibration(filename, valid_users)
# get_user_performance(user_question_order, usertask_dict, answer_dict, self_assessment_first, self_assessment_second)
user_trust_first, user_trust_second = calc_trust_in_automation(filename, valid_users)

all_var_dict = {}
dependent_variables = ["accuracy", "agreement_fraction", "switching_fraction", "appropriate_reliance", "miscalibration", "trust", "RAIR", "RSR"]
for var_name in dependent_variables:
	all_var_dict[var_name] = []

for user in user_question_order:
	tp_order = user_question_order[user]
	first_group = tp_order[:6]
	second_group = tp_order[-6:]
	tp_performance = UserPerformance(username=user, question_order=tp_order)
	
	tp_correct, tp_agreement_fraction, tp_switching_fraction, tp_appropriate_reliance, initial_disagreement_1, RAIR_1, RSR_1 = calc_user_reliance_measures(user, usertask_dict, answer_dict, first_group)
	tp_accuracy = tp_correct / 6.0
	
	tp_performance.add_performance(accuracy=tp_accuracy, agreement_fraction=tp_agreement_fraction, switching_fraction=tp_switching_fraction,
							 appropriate_reliance=tp_appropriate_reliance, relative_positive_ai_reliance=RAIR_1, 
								relative_positive_self_reliance=RSR_1, group="first_group")
	tp_performance.add_miscalibration(self_assessment=self_assessment_first[user], actual_correct_number=tp_correct, group="first_group")
	all_var_dict["accuracy"].append(tp_accuracy)
	all_var_dict["agreement_fraction"].append(tp_agreement_fraction)
	all_var_dict["switching_fraction"].append(tp_switching_fraction)
	all_var_dict["appropriate_reliance"].append(tp_appropriate_reliance)
	all_var_dict["miscalibration"].append(tp_performance.miscalibration["first_group"])
	all_var_dict["trust"].append(user_trust_first[user])
	all_var_dict["RAIR"].append(RAIR_1)
	all_var_dict["RSR"].append(RSR_1)
	
	tp_correct, tp_agreement_fraction, tp_switching_fraction, tp_appropriate_reliance, initial_disagreement_2, RAIR_2, RSR_2 = calc_user_reliance_measures(user, usertask_dict, answer_dict, first_group)
	tp_accuracy = tp_correct / 6.0
	tp_performance.add_performance(accuracy=tp_accuracy, agreement_fraction=tp_agreement_fraction, switching_fraction=tp_switching_fraction,
				appropriate_reliance=tp_appropriate_reliance, relative_positive_ai_reliance=RAIR_2, 
			relative_positive_self_reliance=RSR_2, group="second_group")
	tp_performance.add_miscalibration(self_assessment=self_assessment_second[user], actual_correct_number=tp_correct, group="second_group")
	all_var_dict["accuracy"].append(tp_accuracy)
	all_var_dict["agreement_fraction"].append(tp_agreement_fraction)
	all_var_dict["switching_fraction"].append(tp_switching_fraction)
	all_var_dict["appropriate_reliance"].append(tp_appropriate_reliance)

	all_var_dict["miscalibration"].append(tp_performance.miscalibration["second_group"])
	all_var_dict["trust"].append(user_trust_first[user])
	all_var_dict["RAIR"].append(RAIR_2)
	all_var_dict["RSR"].append(RSR_2)


for var_name in dependent_variables:
	print(var_name, len(all_var_dict[var_name]))
	ks_statistic, p_value = kstest(all_var_dict[var_name], 'norm')
	print("ks_statistic: {:.2f}, p value: {:.2f}".format(ks_statistic, p_value))
	if p_value < 0.05:
		print("Variable {} is not a normal distribution".format(var_name))
	else:
		print("Null hypo cannot be rejected, Variable {} follows a normal distribution".format(var_name))
	print("-"*17)
