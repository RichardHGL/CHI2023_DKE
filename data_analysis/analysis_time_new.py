import numpy as np
import pandas as pd
import os
import sys
from util import UserPerformance, read_completion_time
from util import find_valid_users, get_user_conditions, load_answers, get_user_question_order, read_decisions, calc_miscalibration, calc_user_reliance_measures
from scipy.stats import kruskal


def get_user_performance(user_question_order, usertask_dict, answer_dict, self_assessment_first, self_assessment_second, user2time):
	time_list_overestimation = []
	time_list_underestimation = []
	time_list_overestimation_no = []
	time_list_accurate_self_assessment = []
	time_list_accurate_self_assessment_no = []
	for user in user_question_order:
		tp_order = user_question_order[user]
		first_group = tp_order[:6]
		second_group = tp_order[-6:]
		tp_performance = UserPerformance(username=user, question_order=tp_order)
		
		tp_correct, tp_agreement_fraction, tp_switching_fraction, tp_appropriate_reliance, initial_disagreement_1, relative_positive_ai_reliance, relative_positive_self_reliance = calc_user_reliance_measures(user, usertask_dict, answer_dict, first_group)
		tp_accuracy = tp_correct / 6.0
		tp_performance.add_performance(accuracy=tp_accuracy, agreement_fraction=tp_agreement_fraction, switching_fraction=tp_switching_fraction, 
			appropriate_reliance=tp_appropriate_reliance, relative_positive_ai_reliance=relative_positive_ai_reliance, 
			relative_positive_self_reliance=relative_positive_self_reliance, group="first_group")
		tp_performance.add_miscalibration(self_assessment=self_assessment_first[user], actual_correct_number=tp_correct, group="first_group")
		
		tp_correct, tp_agreement_fraction, tp_switching_fraction, tp_appropriate_reliance, initial_disagreement_2, relative_positive_ai_reliance, relative_positive_self_reliance = calc_user_reliance_measures(user, usertask_dict, answer_dict, second_group)
		tp_accuracy = tp_correct / 6.0
		tp_performance.add_performance(accuracy=tp_accuracy, agreement_fraction=tp_agreement_fraction, switching_fraction=tp_switching_fraction, 
			appropriate_reliance=tp_appropriate_reliance, relative_positive_ai_reliance=relative_positive_ai_reliance, 
			relative_positive_self_reliance=relative_positive_self_reliance, group="second_group")
		tp_performance.add_miscalibration(self_assessment=self_assessment_second[user], actual_correct_number=tp_correct, group="second_group")

		# tp_correct, tp_agreement_fraction, tp_switching_fraction, tp_appropriate_reliance, initial_disagreement_1, relative_positive_ai_reliance, relative_positive_self_reliance = calc_user_reliance_measures(user, usertask_dict, answer_dict, first_group + second_group)
		# tp_accuracy = tp_correct / 12.0
		# tp_performance.add_performance(accuracy=tp_accuracy, agreement_fraction=tp_agreement_fraction, switching_fraction=tp_switching_fraction, 
		# 	appropriate_reliance=tp_appropriate_reliance, relative_positive_ai_reliance=relative_positive_ai_reliance, 
		# 	relative_positive_self_reliance=relative_positive_self_reliance, group="overall")

		# tp_performance.print_information()
		if tp_performance.miscalibration["first_group"] > 0:
			time_list_overestimation.append(user2time[user])
		elif tp_performance.miscalibration["first_group"] == 0:
			time_list_accurate_self_assessment.append(user2time[user])
		else:
			time_list_underestimation.append(user2time[user])

	print("Overestimation", len(time_list_overestimation))
	print("Average time:", np.mean(time_list_overestimation), "SD:", np.std(time_list_overestimation))
	print("-" * 17)

	print("Accurate self-assessment", len(time_list_accurate_self_assessment))
	print("Average time:", np.mean(time_list_accurate_self_assessment), "SD:", np.std(time_list_accurate_self_assessment))
	print("-" * 17)

	print("underestimation", len(time_list_underestimation))
	print("Average time:", np.mean(time_list_underestimation), "SD:", np.std(time_list_underestimation))
	print("-" * 17)

	statistic, pvalue = kruskal(time_list_overestimation, time_list_accurate_self_assessment, time_list_underestimation)
	print("Accurate self-assessment vs others")
	print("kruskal test result: H:{:.2f}, p:{:.3f}".format(statistic, pvalue))


if __name__ == "__main__":
	filename = "all_valid_data.csv"
	valid_users, approved_users = find_valid_users(filename, 4)
	user_condition_dict = get_user_conditions(filename, valid_users)
	answer_dict = load_answers()
	user_question_order = get_user_question_order(filename, valid_users)
	usertask_dict = read_decisions(filename, valid_users)
	user2time = read_completion_time(valid_users)
	self_assessment_first, self_assessment_second, other_assessment_first, other_assessment_second, survey_percetage_first, survey_percetage_second = calc_miscalibration(filename, valid_users)
	
	users_with_explanation = user_condition_dict["no tutorial, with xai"] | user_condition_dict["with tutorial, with xai"]
	get_user_performance(user_question_order, usertask_dict, answer_dict, self_assessment_first, self_assessment_second, user2time)
	print("-" * 17)
	all_time_list = []
	for condition in user_condition_dict:
		print(condition)
		tp_list = [user2time[user] for user in user_condition_dict[condition]]
		print("Average time:", np.mean(tp_list), "SD:", np.std(tp_list))
		all_time_list.append(tp_list)
	statistic, pvalue = kruskal(all_time_list[0], all_time_list[1], all_time_list[2], all_time_list[3])
	print("Time across conditions")
	print("kruskal test result: H:{:.2f}, p:{:.3f}".format(statistic, pvalue))









