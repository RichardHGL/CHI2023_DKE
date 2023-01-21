import numpy as np
import pandas as pd
import pingouin as pg
import os
import sys
from util import UserPerformance
from util import find_valid_users, get_user_conditions, load_answers, get_user_question_order, read_decisions, calc_miscalibration, calc_user_reliance_measures
from scipy.stats import wilcoxon, kruskal, mannwhitneyu

def compare_calibration_effect(list_1, list_2):
	statistic, pvalue = kruskal(list_1, list_2)
	print("Compare self-assessment improvement with explanation vs no explanation")
	print("kruskal test result: H:{:.3f}, p:{:.3f}".format(statistic, pvalue))
	print("Mean: M(Exp):{:.3f}, M(No Exp):{:.3f}".format(np.mean(list_1), np.mean(list_2)))
	print("-" * 17)


def wilcoxon_pairwise_comparison(data_list_1, data_list_2, name1, name2):
	# print("Use pots-hoc analysis")
	threshold = 0.05 / 4
	flag = False
	
	statistic, pvalue = wilcoxon(data_list_1, data_list_2, alternative='greater')
	if pvalue < threshold:
		flag = True
		print("Alternative {} > {},".format(name1, name2), "pvalue %.4f"%pvalue, "statistic %.4f"%statistic)

	statistic, pvalue = wilcoxon(data_list_1, data_list_2, alternative='less')
	if pvalue < threshold:
		flag = True
		print("Alternative {} < {},".format(name1, name2), "pvalue %.4f"%pvalue, "statistic %.4f"%statistic)

	if not flag:
		print("No significant difference with post-hoc analysis")

def test_wilcoxon(list_1, list_2):
	assert len(list_1) == len(list_2)
	# res = wilcoxon(x=list_1, y=list_2)
	# print("With miscalibration, the results are:")
	# # print("Mean: %.3f\t%.3f"%(np.mean(list_1), np.mean(list_2)))
	# print("Mean: M(first):{:.3f}, SD(first):{:.3f}".format(np.mean(list_1), np.std(list_1)))
	# print("Mean: M(second):{:.3f}, SD(second):{:.3f}".format(np.mean(list_2), np.std(list_2)))
	# # print(res)
	# wilcoxon_pairwise_comparison(list_1, list_2, "first", "second")
	# print("-" * 17)


	# The result in paper is based on abstract value
	print("With abstract miscalibration, the results are:")
	abs_list_1 = [abs(x) for x in list_1]
	abs_list_2 = [abs(x) for x in list_2]
	# print(len(abs_list_1), len(abs_list_2))
	# res = wilcoxon(x=abs_list_1, y=abs_list_2)
	# print("%.3f\t%.3f"%(np.mean(abs_list_1), np.mean(abs_list_2)))
	print("abstract_res")
	print("Mean: M(first):{:.3f}, SD(first):{:.3f}".format(np.mean(abs_list_1), np.std(abs_list_1)))
	print("Mean: M(second):{:.3f}, SD(second):{:.3f}".format(np.mean(abs_list_2), np.std(abs_list_2)))
	wilcoxon_pairwise_comparison(abs_list_1, abs_list_2, "first", "second")
	print("-" * 17)


def compare_asssessment(user_set, user_question_order, usertask_dict, answer_dict, self_assessment_first, self_assessment_second, users_with_explanation):
	miscalibration_list_first = []
	miscalibration_list_second = []
	miscalibration_list_first_explanation = []
	miscalibration_list_second_explanation = []
	miscalibration_list_first_no_explanation = []
	miscalibration_list_second_no_explanation = []
	obj_list = []
	# data_variables = {
	# 	"username": [],
	# 	"miscalibration": [],
	# 	"abs-miscalibration": [],
	# 	"group": []
	# }

	miscalibration_all = {}
	miscalibration_all["first"] = []
	miscalibration_all["second"] = []

	miscalibration_under = {}
	miscalibration_under["first"] = []
	miscalibration_under["second"] = []

	miscalibration_over = {}
	miscalibration_over["first"] = []
	miscalibration_over["second"] = []
	for user in user_set:
		tp_order = user_question_order[user]
		first_group = tp_order[:6]
		second_group = tp_order[-6:]
		tp_performance = UserPerformance(username=user, question_order=tp_order)
		
		tp_correct, tp_agreement_fraction, tp_switching_fraction, tp_appropriate_reliance, initial_disagreement_1, relative_positive_ai_reliance, relative_positive_self_reliance = calc_user_reliance_measures(user, usertask_dict, answer_dict, first_group)

		miscalibration_first = self_assessment_first[user] - tp_correct
		if abs(miscalibration_first) < 1:
			# no miscalibration exists in the first batch
			continue


		# miscalibration_list_first.append(miscalibration_first)
		if user in users_with_explanation:
			miscalibration_list_first_explanation.append(miscalibration_first)
		else:
			miscalibration_list_first_no_explanation.append(miscalibration_first)
		
		tp_correct, tp_agreement_fraction, tp_switching_fraction, tp_appropriate_reliance, initial_disagreement_2, relative_positive_ai_reliance, relative_positive_self_reliance = calc_user_reliance_measures(user, usertask_dict, answer_dict, second_group)
		miscalibration_second = self_assessment_second[user] - tp_correct

		miscalibration_all["first"].append(miscalibration_first)
		miscalibration_all["second"].append(miscalibration_second)

		if miscalibration_first < 0:
			miscalibration_under["first"].append(miscalibration_first)
			miscalibration_under["second"].append(miscalibration_second)
		if miscalibration_first > 0:
			miscalibration_over["first"].append(miscalibration_first)
			miscalibration_over["second"].append(miscalibration_second)

		# miscalibration_list_second.append(miscalibration_second)
		if user in users_with_explanation:
			miscalibration_list_second_explanation.append(miscalibration_second)
		else:
			miscalibration_list_second_no_explanation.append(miscalibration_second)

		# tp_obj = {
		# 	"username": user,
		# 	"miscalibration": miscalibration_first,
		# 	"abs-miscalibration": abs(miscalibration_first),
		# 	"group": "first"
		# }
		# obj_list.append(tp_obj)
		# tp_obj = {
		# 	"username": user,
		# 	"miscalibration": miscalibration_second,
		# 	"abs-miscalibration": abs(miscalibration_second),
		# 	"group": "second"
		# }
		# obj_list.append(tp_obj)
	from scipy.stats import wilcoxon
	print("-" * 34)
	num_all = len( miscalibration_all["first"] )
	print(f"For H3, we have {num_all} participants with miscalibration in the first batch with tutorial")
	test_wilcoxon(miscalibration_all["first"], miscalibration_all["second"])
	print("-" * 34)

	num_under = len( miscalibration_under["first"] )
	print(f"For H3, we have {num_under} participants with Underestimation in the first batch with tutorial")
	test_wilcoxon(miscalibration_under["first"], miscalibration_under["second"])

	mis_change = {}
	for mis_1, mis_2 in zip(miscalibration_under["first"], miscalibration_under["second"]):
		if mis_2 == 0:
			pattern = "under -> Accurate"
		elif mis_2 > 0 and abs(mis_2) < abs(mis_1):
			pattern = "under -> over, improved"
		elif mis_2 > 0 and abs(mis_2) >= abs(mis_1):
			pattern = "under -> over, not improved"
		elif mis_2 <0 and abs(mis_2) < abs(mis_1):
			pattern = "under -> under, improved"
		elif mis_1 == mis_2:
			pattern = "not improved"
		else:
			pattern = "getting worse with underestimation"
		if pattern not in mis_change:
			mis_change[pattern] = 0
		mis_change[pattern] += 1
	print(mis_change)
	print("-" * 34)

	num_over = len( miscalibration_over["first"] )
	print(f"For H3, we have {num_over} participants with Overestimation in the first batch with tutorial")
	test_wilcoxon(miscalibration_over["first"], miscalibration_over["second"])

	mis_change = {}
	for mis_1, mis_2 in zip(miscalibration_over["first"], miscalibration_over["second"]):
		if mis_2 == 0:
			pattern = "Over -> Accurate"
		elif mis_2 > 0 and mis_2 < mis_1:
			pattern = "Over -> over, improved"
		elif mis_2 <0 and abs(mis_2) < abs(mis_1):
			pattern = "Over -> under, improved"
		elif mis_2 <0 and abs(mis_2) >= abs(mis_1):
			pattern = "Over -> under, not improved"
		elif mis_1 == mis_2:
			pattern = "not improved"
		else:
			pattern = "getting worse with overestimation"
		if pattern not in mis_change:
			mis_change[pattern] = 0
		mis_change[pattern] += 1
	print(mis_change)
	print("-" * 34)

	# test_wilcoxon(miscalibration_list_first, miscalibration_list_second)

	improvement_list_explanation = []
	for mis1, mis2 in zip(miscalibration_list_first_explanation, miscalibration_list_second_explanation):
		calibration_improvement = abs(mis1) - abs(mis2) # abs(mis2) is expected to be less than abs(mis1)
		improvement_list_explanation.append(calibration_improvement)

	improvement_list_no_explanation = []
	for mis1, mis2 in zip(miscalibration_list_first_no_explanation, miscalibration_list_second_no_explanation):
		calibration_improvement = abs(mis1) - abs(mis2) # abs(mis2) is expected to be less than abs(mis1)
		improvement_list_no_explanation.append(calibration_improvement)

	compare_calibration_effect(improvement_list_explanation, improvement_list_no_explanation)


# H3: the effect of calibration tutorial on self-assessment.


if __name__ == "__main__":
	filename = "all_valid_data.csv"
	valid_users, approved_users = find_valid_users(filename, 4)
	user_condition_dict = get_user_conditions(filename, valid_users)
	answer_dict = load_answers()
	user_question_order = get_user_question_order(filename, valid_users)
	usertask_dict = read_decisions(filename, valid_users)
	self_assessment_first, self_assessment_second, other_assessment_first, other_assessment_second, survey_percetage_first, survey_percetage_second = calc_miscalibration(filename, valid_users)
	
	with_tutorial_user_set = user_condition_dict["with tutorial, no xai"] | user_condition_dict["with tutorial, with xai"]
	users_with_explanation = user_condition_dict["with tutorial, with xai"]
	# we only consider participants with tutorial
	# print(f"For H3, we have {len(with_tutorial_user_set)} participants for analysis")
	compare_asssessment(with_tutorial_user_set, user_question_order, usertask_dict, answer_dict, self_assessment_first, self_assessment_second, users_with_explanation)











