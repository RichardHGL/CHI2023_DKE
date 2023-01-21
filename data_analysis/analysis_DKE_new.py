import numpy as np
import pandas as pd
import pingouin as pg
import os
import sys
from util import UserPerformance, read_completion_time
from util import find_valid_users, get_user_conditions, load_answers, get_user_question_order, read_decisions, calc_miscalibration, calc_user_reliance_measures
from scipy.stats import wilcoxon, kruskal, mannwhitneyu
import math


def check_overestimation(user_list, users_underestimation, users_accurate_assessment, users_overestimation):
	overlap_over = 0
	overlap_acc = 0
	overlap_under = 0
	for user in user_list:
		if user in users_overestimation:
			overlap_over += 1
		elif user in users_underestimation:
			overlap_under += 1
		else:
			overlap_acc += 1
	print(f"Among {len(user_list)} users, {overlap_under} users with underestimation, {overlap_acc} users have accurate self-assessment, {overlap_over} users with overestimation")


def get_user_performance(user_question_order, usertask_dict, answer_dict, self_assessment_first, self_assessment_second, user2time):
	# performance_list = []
	users_overestimation = set()
	users_accurate_assessment = set()
	users_underestimation = set()
	user_accuracy_list = []
	user2performance = {}
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

		user_accuracy_list.append((user, tp_accuracy))
		# To avoid the impact of tutorial intervention, we only considered the first batch of tasks
		
		tp_correct, tp_agreement_fraction, tp_switching_fraction, tp_appropriate_reliance, initial_disagreement_2, relative_positive_ai_reliance, relative_positive_self_reliance = calc_user_reliance_measures(user, usertask_dict, answer_dict, second_group)
		tp_accuracy = tp_correct / 6.0
		tp_performance.add_performance(accuracy=tp_accuracy, agreement_fraction=tp_agreement_fraction, switching_fraction=tp_switching_fraction, 
			appropriate_reliance=tp_appropriate_reliance, relative_positive_ai_reliance=relative_positive_ai_reliance, 
			relative_positive_self_reliance=relative_positive_self_reliance, group="second_group")
		tp_performance.add_miscalibration(self_assessment=self_assessment_second[user], actual_correct_number=tp_correct, group="second_group")

		tp_correct, tp_agreement_fraction, tp_switching_fraction, tp_appropriate_reliance, initial_disagreement_1, relative_positive_ai_reliance, relative_positive_self_reliance = calc_user_reliance_measures(user, usertask_dict, answer_dict, first_group + second_group)
		tp_accuracy = tp_correct / 12.0
		tp_performance.add_performance(accuracy=tp_accuracy, agreement_fraction=tp_agreement_fraction, switching_fraction=tp_switching_fraction, 
			appropriate_reliance=tp_appropriate_reliance, relative_positive_ai_reliance=relative_positive_ai_reliance, 
			relative_positive_self_reliance=relative_positive_self_reliance, group="overall")

		user2performance[user] = tp_performance

		# tp_performance.print_information()
		if tp_performance.miscalibration["first_group"] > 0:
			users_overestimation.add(user)
		elif tp_performance.miscalibration["first_group"] < 0:
			users_underestimation.add(user)
		else:
			users_accurate_assessment.add(user)

	def get_performance_dict(performance_list):
		keys = ["accuracy", "agreement_fraction", "switching_fraction", "appropriate_reliance", "relative_positive_ai_reliance", "relative_positive_self_reliance"]
		performance_dict = {}
		for key_ in keys:
			performance_dict[key_] = []
		for tp_performance in performance_list:
			for key_ in keys:
				performance_dict[key_].append(tp_performance.performance["first_group"][key_])
		return performance_dict

	def compare_performance(performance_list_1, performance_list_2):
		performance_dict_1 = get_performance_dict(performance_list_1)
		performance_dict_2 = get_performance_dict(performance_list_2)
		keys = ["accuracy", "agreement_fraction", "switching_fraction", "appropriate_reliance", "relative_positive_ai_reliance", "relative_positive_self_reliance"]
		for var_name in keys:
			data_list_1 = performance_dict_1[var_name]
			data_list_2 = performance_dict_2[var_name]
			statistic, pvalue = kruskal(data_list_1, data_list_2)
			print(var_name)
			print("kruskal test result: H:{:.2f}, p:{:.3f}".format(statistic, pvalue))
			print("Mean: M(Top):{:.2f}, SD(Top):{:.2f}".format(np.mean(data_list_1), np.std(data_list_1)))
			print("Mean: M(Bottom):{:.2f}, SD(Bottom):{:.2f}".format(np.mean(data_list_2), np.std(data_list_2)))
			if pvalue < 0.05 / 4:
				# here we just assume the high accuracy group show higher reliance
				statistic, pvalue = mannwhitneyu(data_list_1, data_list_2, alternative='greater')
				if pvalue < 0.05 / 4:
					print("Alternative {} > {},".format("high accuracy", "low accuracy"), "pvalue %.4f"%pvalue, "statistic %.4f"%statistic)
			print("-" * 17)

	user_accuracy_list.sort(key=lambda x:x[1], reverse=True)
	quat_number = math.ceil(len(user_accuracy_list) * 0.25)
	# print(user_accuracy_list[:10])
	# print(user_accuracy_list[-10:])
	top_users = [item[0] for item in user_accuracy_list[:quat_number]]
	bottom_users = [item[0] for item in user_accuracy_list[-quat_number:]]

	top_user_time_list = [user2time[user] for user in top_users]
	bottom_users_time_list = [user2time[user] for user in bottom_users]
	statistic, pvalue = kruskal(top_user_time_list, bottom_users_time_list)
	print("High performance vs Low performance in time")
	print("kruskal test result: H:{:.2f}, p:{:.3f}".format(statistic, pvalue))
	print("Average time for top performance users:", round(np.mean(top_user_time_list) / 60.0), "SD:", round(np.std(top_user_time_list) / 60))
	print("Average time for bottom performance users:", round(np.mean(bottom_users_time_list) / 60.0), "SD:", round(np.std(bottom_users_time_list) / 60.0))

	print(f"In total, {len(users_underestimation)} participants show underestimation on the first batch of tasks")
	print(f"In total, {len(users_accurate_assessment)} participants show accurate self-assessment on the first batch of tasks")
	print(f"In total, {len(users_overestimation)} participants show overestimation on the first batch of tasks")
	print("Users with high accuracy")
	check_overestimation(top_users, users_underestimation, users_accurate_assessment, users_overestimation)
	print("Users with low accuracy")
	check_overestimation(bottom_users, users_underestimation, users_accurate_assessment, users_overestimation)

	performance_list_top = [user2performance[user] for user in top_users]
	performance_list_bottom = [user2performance[user] for user in bottom_users]

	compare_performance(performance_list_top, performance_list_bottom)


if __name__ == "__main__":
	filename = "all_valid_data.csv"
	valid_users, approved_users = find_valid_users(filename, 4)
	user_condition_dict = get_user_conditions(filename, valid_users)
	answer_dict = load_answers()
	user_question_order = get_user_question_order(filename, valid_users)
	usertask_dict = read_decisions(filename, valid_users)
	user2time = read_completion_time(valid_users)
	self_assessment_first, self_assessment_second, other_assessment_first, other_assessment_second, survey_percetage_first, survey_percetage_second = calc_miscalibration(filename, valid_users)

	# users_with_explanation = user_condition_dict["no tutorial, with xai"] | user_condition_dict["with tutorial, with xai"]
	get_user_performance(user_question_order, usertask_dict, answer_dict, self_assessment_first, self_assessment_second, user2time)