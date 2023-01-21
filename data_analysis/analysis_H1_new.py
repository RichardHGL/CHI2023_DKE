import numpy as np
import pandas as pd
import os
import sys
from util import UserPerformance, calc_trust_in_automation
from util import find_valid_users, get_user_conditions, load_answers, get_user_question_order, read_decisions, calc_miscalibration, calc_user_reliance_measures
from scipy.stats import kruskal, mannwhitneyu


def post_hoc_comparison(data_list_1, data_list_2, name1, name2):
	print("Use pots-hoc analysis")
	threshold = 0.05 / 4
	flag = False
	statistic, pvalue = mannwhitneyu(data_list_1, data_list_2, alternative='greater')
	if pvalue < threshold:
		print("Alternative {} > {},".format(name1, name2), "pvalue %.4f"%pvalue, "statistic %.4f"%statistic)
		flag = True
	statistic, pvalue = mannwhitneyu(data_list_1, data_list_2, alternative='less')
	if pvalue < threshold:
		print("Alternative {} < {},".format(name1, name2), "pvalue %.4f"%pvalue, "statistic %.4f"%statistic)
		flag = True
	if not flag:
		print("No significant difference with post-hoc analysis")

def get_user_performance(user_trust_first, user_trust_second, user_question_order, usertask_dict, answer_dict, self_assessment_first, self_assessment_second, users_with_explanation):
	performance_with_overestimation = []
	performance_without_overestimation = []
	performance_with_overestimation_explanation = []
	performance_with_overestimation_no_explanation = []
	performance_without_overestimation_with_explanation = []
	performance_groups = {}
	performance_groups["underestimation"] = []
	performance_groups["overestimation"] = []
	performance_groups["accurate_self_assessment"] = []
	performance_groups["xai"] = []
	performance_groups["no_xai"] = []

	# overestimation_groups = {}
	# overestimation_groups["xai"] = []
	# overestimation_groups["no_xai"] = []

	xai_performance_groups = {}
	xai_performance_groups["underestimation"] = []
	xai_performance_groups["overestimation"] = []
	xai_performance_groups["accurate_self_assessment"] = []

	noxai_performance_groups = {}
	noxai_performance_groups["underestimation"] = []
	noxai_performance_groups["overestimation"] = []
	noxai_performance_groups["accurate_self_assessment"] = []

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

		tp_correct, tp_agreement_fraction, tp_switching_fraction, tp_appropriate_reliance, initial_disagreement_1, relative_positive_ai_reliance, relative_positive_self_reliance = calc_user_reliance_measures(user, usertask_dict, answer_dict, first_group + second_group)
		tp_accuracy = tp_correct / 12.0
		tp_performance.add_performance(accuracy=tp_accuracy, agreement_fraction=tp_agreement_fraction, switching_fraction=tp_switching_fraction, 
			appropriate_reliance=tp_appropriate_reliance, relative_positive_ai_reliance=relative_positive_ai_reliance, 
			relative_positive_self_reliance=relative_positive_self_reliance, group="overall")

		# tp_performance.print_information()
		if user in users_with_explanation:
			performance_groups["xai"].append(tp_performance)
		else:
			performance_groups["no_xai"].append(tp_performance)
		# based on first batch, we identify participants tend to show over-estimation, under-estimation and accurate self-assessment
		if tp_performance.miscalibration["first_group"] > 0:
			performance_groups["overestimation"].append(tp_performance)
			if user in users_with_explanation:
				xai_performance_groups["overestimation"].append(tp_performance)
			else:
				noxai_performance_groups["overestimation"].append(tp_performance)
		elif tp_performance.miscalibration["first_group"] < 0:
			performance_groups["underestimation"].append(tp_performance)
			if user in users_with_explanation:
				xai_performance_groups["underestimation"].append(tp_performance)
			else:
				noxai_performance_groups["underestimation"].append(tp_performance)
		else:
			performance_groups["accurate_self_assessment"].append(tp_performance)
			if user in users_with_explanation:
				xai_performance_groups["accurate_self_assessment"].append(tp_performance)
			else:
				noxai_performance_groups["accurate_self_assessment"].append(tp_performance)

	for key_ in performance_groups:
		print(key_, len(performance_groups[key_]))
	for key_ in xai_performance_groups:
		print(key_, len(xai_performance_groups[key_]), len(noxai_performance_groups))

	def get_performance_dict(performance_list):
		keys = ["accuracy", "agreement_fraction", "switching_fraction", "appropriate_reliance", "relative_positive_ai_reliance", "relative_positive_self_reliance"]
		performance_dict = {}
		for key_ in keys:
			performance_dict[key_] = []
		performance_dict["trust"] = []
		for tp_performance in performance_list:
			for key_ in keys:
				performance_dict[key_].append(tp_performance.performance["first_group"][key_])
				# we only look at the first batch for H1
			performance_dict["trust"].append(user_trust_first[tp_performance.username])
		return performance_dict

	def get_mean_overall_performance(performance_list, group_name="with DKE"):
		keys = ["accuracy", "agreement_fraction", "switching_fraction", "appropriate_reliance", "relative_positive_ai_reliance", "relative_positive_self_reliance"]
		performance_dict = get_performance_dict(performance_list)
		print(group_name, len(performance_dict["accuracy"]))
		for key_ in keys:
			print(key_, np.mean(performance_dict[key_]))

	def compare_xai(performance_list_xai, performance_list_no_xai):
		performance_dict_1 = get_performance_dict(performance_list_xai)
		performance_dict_2 = get_performance_dict(performance_list_no_xai)
		keys = ["accuracy", "agreement_fraction", "switching_fraction", "appropriate_reliance", "relative_positive_ai_reliance", "relative_positive_self_reliance", "trust"]
		for var_name in keys:
			data_list_1 = performance_dict_1[var_name]
			data_list_2 = performance_dict_2[var_name]
			statistic, pvalue = kruskal(data_list_1, data_list_2)
			print(var_name)
			if pvalue < 0.05 / 4:
				print("kruskal test result: H:{:.2f}, p:{:.3f}".format(statistic, pvalue))
				print("Mean: M(XAI):{:.2f}, SD(XAI):{:.2f}".format(np.mean(data_list_1), np.std(data_list_1)))
				print("Mean: M(No XAI):{:.2f}, SD(No XAI):{:.2f}".format(np.mean(data_list_2), np.std(data_list_2)))
				post_hoc_comparison(data_list_1, data_list_2, "XAI", "No XAI")
			else:
				print("No significant difference")
			print("-" * 17)

	def compare_three_groups(performance_underestimation, performance_overestimation, performance_accurate_self_assessment):
		performance_dict_1 = get_performance_dict(performance_underestimation)
		performance_dict_2 = get_performance_dict(performance_overestimation)
		performance_dict_3 = get_performance_dict(performance_accurate_self_assessment)
		keys = ["accuracy", "agreement_fraction", "switching_fraction", "appropriate_reliance", "relative_positive_ai_reliance", "relative_positive_self_reliance", "trust"]
		for var_name in keys:
			data_list_1 = performance_dict_1[var_name]
			data_list_2 = performance_dict_2[var_name]
			data_list_3 = performance_dict_3[var_name]
			print(len(data_list_1), len(data_list_3), len(data_list_2))
			statistic, pvalue = kruskal(data_list_1, data_list_2, data_list_3)
			print(var_name)
			print("kruskal test result: H:{:.2f}, p:{:.3f}".format(statistic, pvalue))
			print("Mean: M(underestimation):{:.2f}, SD(underestimation):{:.2f}".format(np.mean(data_list_1), np.std(data_list_1)))
			print("Mean: M(accurate_self_assessment):{:.2f}, SD(accurate_self_assessment):{:.2f}".format(np.mean(data_list_3), np.std(data_list_3)))
			print("Mean: M(overestimation):{:.2f}, SD(overestimation):{:.2f}".format(np.mean(data_list_2), np.std(data_list_2)))
			if pvalue < 0.05 / 4:
				post_hoc_comparison(data_list_1, data_list_2, "underestimation", "overestimation")
				post_hoc_comparison(data_list_3, data_list_2, "accurate_self_assessment", "overestimation")
				# we assume participants with accurate self assessment and underestimation rely more than participants with overestimation
				post_hoc_comparison(data_list_1, data_list_3, "underestimation", "accurate_self_assessment")
				# post_hoc_comparison(data_list_2, data_list_3, "overestimation", "accurate_self_assessment")
			print("-" * 17)

	print("-" * 34)
	compare_three_groups(performance_underestimation=performance_groups["underestimation"],
						 performance_overestimation=performance_groups["overestimation"], 
						 performance_accurate_self_assessment=performance_groups["accurate_self_assessment"])
	# print("-" * 34)
	# compare_xai(performance_list_xai=performance_groups["xai"], performance_list_no_xai=performance_groups["no_xai"])
	print("-" * 34)
	print("All : ")
	compare_xai(performance_list_xai=xai_performance_groups["underestimation"] + xai_performance_groups["accurate_self_assessment"]+ xai_performance_groups["overestimation"], 
				performance_list_no_xai=noxai_performance_groups["underestimation"] + noxai_performance_groups["accurate_self_assessment"]+ noxai_performance_groups["overestimation"])
	print("-" * 34)
	print("underestimation : ")
	compare_xai(performance_list_xai=xai_performance_groups["underestimation"], performance_list_no_xai=noxai_performance_groups["underestimation"])
	print("-" * 34)
	print("accurate_self_assessment : ")
	compare_xai(performance_list_xai=xai_performance_groups["accurate_self_assessment"], performance_list_no_xai=noxai_performance_groups["accurate_self_assessment"])
	print("-" * 34)
	print("overestimation : ")
	compare_xai(performance_list_xai=xai_performance_groups["overestimation"], performance_list_no_xai=noxai_performance_groups["overestimation"])
	# print(f"{len(performance_with_overestimation)} participants are observed to show DKE on both groups of tasks, {len(performance_without_overestimation)} participants show at most once")
	# print(f"{len(performance_with_overestimation_explanation)} / {len(performance_with_overestimation)} DKE participants are provided with explanation, {len(performance_without_overestimation_with_explanation)} / {len(performance_without_overestimation)}  other participants are provided with explanation")


	# print("Compare DKE with others:")
	# compare_performance(performance_with_overestimation, performance_without_overestimation)
	# print("-" * 34)
	# print("Compare DKE participants, with explanations vs no explanation:")
	# # question: which is correct
	# compare_performance(performance_with_overestimation_explanation, performance_with_overestimation_no_explanation)


if __name__ == "__main__":
	filename = "all_valid_data.csv"
	valid_users, approved_users = find_valid_users(filename, 4)
	user_condition_dict = get_user_conditions(filename, valid_users)
	answer_dict = load_answers()
	user_question_order = get_user_question_order(filename, valid_users)
	usertask_dict = read_decisions(filename, valid_users)
	user_trust_first, user_trust_second = calc_trust_in_automation(filename, valid_users)
	self_assessment_first, self_assessment_second, other_assessment_first, other_assessment_second, survey_percetage_first, survey_percetage_second = calc_miscalibration(filename, valid_users)
	
	users_with_explanation = user_condition_dict["no tutorial, with xai"] | user_condition_dict["with tutorial, with xai"]
	get_user_performance(user_trust_first, user_trust_second, user_question_order, usertask_dict, answer_dict, self_assessment_first, self_assessment_second, users_with_explanation)










