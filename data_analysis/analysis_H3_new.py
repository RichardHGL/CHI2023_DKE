import numpy as np
import pandas as pd
import pingouin as pg
import os
import sys
from util import UserPerformance, calc_trust_in_automation, analysis_user_reliance_measures
from util import find_valid_users, get_user_conditions, load_answers, get_user_question_order, read_decisions, calc_miscalibration, calc_user_reliance_measures
from scipy.stats import wilcoxon, kruskal, mannwhitneyu, spearmanr


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

def compare_performance_improvement(var_name, list_1, list_2):
	statistic, pvalue = kruskal(list_1, list_2)
	# print("Compare performance improvement on {} with explanation vs no explanation".format(var_name))
	print(var_name)
	print(len(list_1), len(list_2))
	print("kruskal test result: H:{:.2f}, p:{:.3f}".format(statistic, pvalue))
	# print("Mean: M(Exp):{:.3f}, M(No Exp):{:.3f}".format(np.mean(list_1), np.mean(list_2)))
	print("Mean: M(Exp):{:.2f}, SD(Exp):{:.2f}".format(np.mean(list_1), np.std(list_1)))
	print("Mean: M(No Exp):{:.2f}, SD(No Exp):{:.2f}".format(np.mean(list_2), np.std(list_2)))
	if pvalue >= 0.05 / 4:
		print("No significant difference")
	else:
		print("With significant difference in Kruskal test, check with mannwhitneyu post-hoc analysis")
		post_hoc_comparison(data_list_1=list_1, data_list_2=list_2, name1="XAI", name2="No XAI")
	print("-" * 17)

def wilcoxon_pairwise_comparison_under(data_list_1, data_list_2, name1, name2):
	# print("Use pots-hoc analysis")
	threshold = 0.05 / 4
	flag = False
	
	# statistic, pvalue = wilcoxon(data_list_1, data_list_2, alternative='greater')
	# if pvalue < threshold:
	# 	flag = True
	# 	print("Alternative {} > {},".format(name1, name2), "pvalue %.4f"%pvalue, "statistic %.4f"%statistic)

	# the assumption should be the second batch performs better than the first batch
	statistic, pvalue = wilcoxon(data_list_1, data_list_2, alternative='greater')
	if pvalue < threshold:
		flag = True
		print("Alternative {} > {},".format(name1, name2), "pvalue %.4f"%pvalue, "statistic %.4f"%statistic)
	else:
		print("Alternative {} > {},".format(name1, name2), "pvalue %.4f"%pvalue, "statistic %.4f"%statistic)

	if not flag:
		print("No significant difference with post-hoc analysis")

def wilcoxon_pairwise_comparison(data_list_1, data_list_2, name1, name2):
	# print("Use pots-hoc analysis")
	threshold = 0.05 / 4
	flag = False
	
	# statistic, pvalue = wilcoxon(data_list_1, data_list_2, alternative='greater')
	# if pvalue < threshold:
	# 	flag = True
	# 	print("Alternative {} > {},".format(name1, name2), "pvalue %.4f"%pvalue, "statistic %.4f"%statistic)

	# the assumption should be the second batch performs better than the first batch
	statistic, pvalue = wilcoxon(data_list_1, data_list_2, alternative='less')
	if pvalue < threshold:
		flag = True
		print("Alternative {} < {},".format(name1, name2), "pvalue %.4f"%pvalue, "statistic %.4f"%statistic)
	else:
		print("Alternative {} < {},".format(name1, name2), "pvalue %.4f"%pvalue, "statistic %.4f"%statistic)

	if not flag:
		print("No significant difference with post-hoc analysis")

def compare_performance(user_trust_first, user_trust_second, user_set, user_question_order, usertask_dict, answer_dict, self_assessment_first, self_assessment_second, users_with_explanation, mode="all"):
	assert mode in ["all", "overestimation", "underestimation", "accurate"]
	# all means all participants with miscalibration, = overestimation + underestimation
	print("Mode {}".format(mode))
	print("-" * 34)
	miscalibration_list_first = []
	miscalibration_list_second = []
	obj_list = []
	performance_dict = {
		"user": [],
		"miscalibration": [[], []],
		"accuracy": [[], []],
		"aggreement_fraction": [[], []],
		"switching_fraction": [[], []],
		"appropriate_reliance": [[], []],
		"relative_positive_ai_reliance": [[], []],
		"relative_positive_self_reliance": [[], []],
		"trust": [[], []],
	}
	performance_dict_explanation = {
		"user": [],
		"miscalibration": [[], []],
		"accuracy": [[], []],
		"aggreement_fraction": [[], []],
		"switching_fraction": [[], []],
		"appropriate_reliance": [[], []],
		"relative_positive_ai_reliance": [[], []],
		"relative_positive_self_reliance": [[], []],
		"trust": [[], []],
	}
	performance_dict_no_explanation = {
		"user": [],
		"miscalibration": [[], []],
		"accuracy": [[], []],
		"aggreement_fraction": [[], []],
		"switching_fraction": [[], []],
		"appropriate_reliance": [[], []],
		"relative_positive_ai_reliance": [[], []],
		"relative_positive_self_reliance": [[], []],
		"trust": [[], []],
	}
	for user in user_set:
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
		if mode != "accurate":
			if abs(tp_performance.miscalibration["first_group"]) == 0:
				# no miscalibration exists
				continue
		if mode == "underestimation":
			# remove participants who overestimate themselves in the first batch
			if tp_performance.miscalibration["first_group"] > 0:
				continue
		if mode == "overestimation":
			# remove participants who underestimate themselves in the first batch
			if tp_performance.miscalibration["first_group"] < 0:
				continue
		if mode == "accurate":
			# remove participants who miscalibrated self-assessmet in the first batch
			if tp_performance.miscalibration["first_group"] != 0:
				continue
		performance_dict["user"].append(user)
		performance_dict["accuracy"][0].append(tp_accuracy)
		performance_dict["miscalibration"][0].append(tp_performance.miscalibration["first_group"])
		performance_dict["aggreement_fraction"][0].append(tp_agreement_fraction)
		performance_dict["switching_fraction"][0].append(tp_switching_fraction)
		performance_dict["appropriate_reliance"][0].append(tp_appropriate_reliance)
		performance_dict["relative_positive_ai_reliance"][0].append(relative_positive_ai_reliance)
		performance_dict["relative_positive_self_reliance"][0].append(relative_positive_self_reliance)
		performance_dict["trust"][0].append(user_trust_first[user])
		if user in users_with_explanation:
			performance_dict_explanation["user"].append(user)
			performance_dict_explanation["accuracy"][0].append(tp_accuracy)
			performance_dict_explanation["aggreement_fraction"][0].append(tp_agreement_fraction)
			performance_dict_explanation["switching_fraction"][0].append(tp_switching_fraction)
			performance_dict_explanation["appropriate_reliance"][0].append(tp_appropriate_reliance)
			performance_dict_explanation["relative_positive_ai_reliance"][0].append(relative_positive_ai_reliance)
			performance_dict_explanation["relative_positive_self_reliance"][0].append(relative_positive_self_reliance)
			performance_dict_explanation["trust"][0].append(user_trust_first[user])
		else:
			performance_dict_no_explanation["user"].append(user)
			performance_dict_no_explanation["accuracy"][0].append(tp_accuracy)
			performance_dict_no_explanation["aggreement_fraction"][0].append(tp_agreement_fraction)
			performance_dict_no_explanation["switching_fraction"][0].append(tp_switching_fraction)
			performance_dict_no_explanation["appropriate_reliance"][0].append(tp_appropriate_reliance)
			performance_dict_no_explanation["relative_positive_ai_reliance"][0].append(relative_positive_ai_reliance)
			performance_dict_no_explanation["relative_positive_self_reliance"][0].append(relative_positive_self_reliance)
			performance_dict_no_explanation["trust"][0].append(user_trust_first[user])
		
		tp_correct, tp_agreement_fraction, tp_switching_fraction, tp_appropriate_reliance, initial_disagreement_2, relative_positive_ai_reliance, relative_positive_self_reliance = calc_user_reliance_measures(user, usertask_dict, answer_dict, second_group)
		tp_accuracy = tp_correct / 6.0
		tp_performance.add_performance(accuracy=tp_accuracy, agreement_fraction=tp_agreement_fraction, switching_fraction=tp_switching_fraction, 
			appropriate_reliance=tp_appropriate_reliance, relative_positive_ai_reliance=relative_positive_ai_reliance, 
			relative_positive_self_reliance=relative_positive_self_reliance, group="second_group")
		tp_performance.add_miscalibration(self_assessment=self_assessment_second[user], actual_correct_number=tp_correct, group="second_group")

		performance_dict["accuracy"][1].append(tp_accuracy)
		performance_dict["miscalibration"][1].append(tp_performance.miscalibration["second_group"])
		performance_dict["aggreement_fraction"][1].append(tp_agreement_fraction)
		performance_dict["switching_fraction"][1].append(tp_switching_fraction)
		performance_dict["appropriate_reliance"][1].append(tp_appropriate_reliance)
		performance_dict["relative_positive_ai_reliance"][1].append(relative_positive_ai_reliance)
		performance_dict["relative_positive_self_reliance"][1].append(relative_positive_self_reliance)
		performance_dict["trust"][1].append(user_trust_second[user])
		if user in users_with_explanation:
			performance_dict_explanation["accuracy"][1].append(tp_accuracy)
			performance_dict_explanation["aggreement_fraction"][1].append(tp_agreement_fraction)
			performance_dict_explanation["switching_fraction"][1].append(tp_switching_fraction)
			performance_dict_explanation["appropriate_reliance"][1].append(tp_appropriate_reliance)
			performance_dict_explanation["relative_positive_ai_reliance"][1].append(relative_positive_ai_reliance)
			performance_dict_explanation["relative_positive_self_reliance"][1].append(relative_positive_self_reliance)
			performance_dict_explanation["trust"][1].append(user_trust_second[user])
		else:
			performance_dict_no_explanation["accuracy"][1].append(tp_accuracy)
			performance_dict_no_explanation["aggreement_fraction"][1].append(tp_agreement_fraction)
			performance_dict_no_explanation["switching_fraction"][1].append(tp_switching_fraction)
			performance_dict_no_explanation["appropriate_reliance"][1].append(tp_appropriate_reliance)
			performance_dict_no_explanation["relative_positive_ai_reliance"][1].append(relative_positive_ai_reliance)
			performance_dict_no_explanation["relative_positive_self_reliance"][1].append(relative_positive_self_reliance)
			performance_dict_no_explanation["trust"][1].append(user_trust_second[user])
		
	number_participants = len(performance_dict["accuracy"][0])
	print(f"For H4, we have {number_participants} participants with miscalibration in the first group with tutorial")
	
	for dv_name in ["accuracy", "aggreement_fraction", "switching_fraction", "appropriate_reliance", "relative_positive_ai_reliance", "relative_positive_self_reliance", "trust"]:
		print(f"For metric {dv_name}, wilcoxon test res:")
		print(len(performance_dict[dv_name][0]), len(performance_dict[dv_name][1]))
		# print("Mean %.3f\t%.3f"%(np.mean(performance_dict[dv_name][0]), np.mean(performance_dict[dv_name][1])))
		print("Mean: M(first):{:.2f}, SD(first):{:.2f}".format(np.mean(performance_dict[dv_name][0]), np.std(performance_dict[dv_name][0])))
		print("Mean: M(second):{:.2f}, SD(second):{:.2f}".format(np.mean(performance_dict[dv_name][1]), np.std(performance_dict[dv_name][1])))
		# res = wilcoxon(x=performance_dict[dv_name][0], y=performance_dict[dv_name][1])
		# print(res)
		if mode == "underestimation":
			wilcoxon_pairwise_comparison_under(data_list_1=performance_dict[dv_name][0], data_list_2=performance_dict[dv_name][1], name1="first", name2="second")
		else:
			# for other case, consider second batch better
			wilcoxon_pairwise_comparison(data_list_1=performance_dict[dv_name][0], data_list_2=performance_dict[dv_name][1], name1="first", name2="second")
		print("-" * 17)

	# def get_miscaibration_pattern_underestimation(mis_1, mis_2):
	# 	if mis_2 == 0:
	# 		pattern = "under -> Accurate"
	# 	elif mis_2 > 0 and abs(mis_2) < abs(mis_1):
	# 		pattern = "under -> over, improved"
	# 	elif mis_2 > 0 and abs(mis_2) >= abs(mis_1):
	# 		pattern = "under -> over, not improved"
	# 	elif mis_2 <0 and abs(mis_2) < abs(mis_1):
	# 		pattern = "under -> under, improved"
	# 	elif mis_1 == mis_2:
	# 		pattern = "not improved"
	# 	else:
	# 		pattern = "getting worse with underestimation"
	# 	return pattern

	def check_no_switch_with_initial_disagreement(user):
		tp_order = user_question_order[user]
		first_group = tp_order[:6]
		second_group = tp_order[-6:]
		tp_performance = UserPerformance(username=user, question_order=tp_order)

		tp_correct, tp_agreement, tp_switching_fraction_1, disagreement_patterns_1, four_patterns_1, AR_1 = analysis_user_reliance_measures(user, usertask_dict, answer_dict, first_group)
		RAIR_1, RSR_1 = AR_1
		initial_disagreement_1, switch_wid_1, insist_wid_1 = disagreement_patterns_1
		positive_ai_reliance_1, negative_self_reliance_1, positive_self_reliance_1, negative_ai_reliance_1 = four_patterns_1
		
		tp_correct, tp_agreement, tp_switching_fraction_2, disagreement_patterns_2, four_patterns_2, AR_2 = analysis_user_reliance_measures(user, usertask_dict, answer_dict, second_group)
		RAIR_2, RSR_2 = AR_2
		initial_disagreement_2, switch_wid_2, insist_wid_2 = disagreement_patterns_2
		positive_ai_reliance_2, negative_self_reliance_2, positive_self_reliance_2, negative_ai_reliance_2 = four_patterns_2
		
		assert RSR_1 > RSR_2
		if initial_disagreement_1 > 0 and initial_disagreement_2 == 0:
			print("participants no initial disagreement in the second batch, while has initial disagreement in the first batch")		
		else:
			print("Both with initial disagreement", initial_disagreement_1, initial_disagreement_2)
			print("First batch, {} initial disagreement, {} switch, {} insist".format(initial_disagreement_1, switch_wid_1, insist_wid_1))
			print("{} positive AI Reliance, {} negative self reliance\n{} positive self Reliance, {} negative AI reliance".format(positive_ai_reliance_1, negative_self_reliance_1, positive_self_reliance_1, negative_ai_reliance_1))
			
			print("Second batch, {} initial disagreement, {} switch, {} insist".format(initial_disagreement_2, switch_wid_2, insist_wid_2))
			print("{} positive AI Reliance, {} negative self reliance\n{} positive self Reliance, {} negative AI reliance".format(positive_ai_reliance_2, negative_self_reliance_2, positive_self_reliance_2, negative_ai_reliance_2))
			
			# print("First batch", tp_switching_fraction_1, RAIR_1, RSR_1)
			# print("Second batch", tp_switching_fraction_2, RAIR_2, RSR_2)
			print("-" * 17)
			return 

	def get_miscaibration_pattern_underestimation(mis_1, mis_2):
		# if mis_2 == 0:
		# 	pattern = "under -> Accurate"
		# elif mis_2 > 0:
		# 	pattern = "under -> over"
		# # else:
		# # 	pattern = "under -> under"
		# elif mis_2 <0 and abs(mis_2) < abs(mis_1):
		# 	pattern = "under -> under, improved"
		# elif mis_1 == mis_2:
		# 	pattern = "keep the same"
		# else:
		# 	pattern = "under -> under,getting worse with underestimation"
		if mis_2 == 0:
			pattern = "under -> Accurate"
		elif mis_2 < 0:
			pattern = "under -> under"
		else:
			pattern = "under -> Over"
		return pattern

	if mode == "underestimation":
		# try to see how the ones who showed decreased performance in miscalibration change
		# dv_name = "appropriate_reliance"
		# dv_name = "relative_positive_ai_reliance"
		# dv_name = "relative_positive_self_reliance"
		# print(f"For participants showed decreased {dv_name}")
		print(len(performance_dict[dv_name][0]), len(performance_dict[dv_name][1]))
		pattern_dict = {}
		pattern_decrease_dict = {}
		diff_dict = {}
		number_participants = len(performance_dict[dv_name][0])
		participant_decrease_dv = 0
		print("-" * 17)
		for index in range(number_participants):
			pattern = get_miscaibration_pattern_underestimation(performance_dict["miscalibration"][0][index], performance_dict["miscalibration"][1][index])
			if pattern not in pattern_dict:
				pattern_dict[pattern] = 0
			pattern_dict[pattern] += 1
			for dv_name in ["miscalibration", "accuracy", "appropriate_reliance", "relative_positive_ai_reliance", "relative_positive_self_reliance"]:
				if dv_name not in diff_dict:
					diff_dict[dv_name] = []
				diff_dict[dv_name].append(performance_dict[dv_name][1][index] - performance_dict[dv_name][0][index])
				if dv_name not in pattern_decrease_dict:
					pattern_decrease_dict[dv_name] = {}
				if performance_dict[dv_name][0][index] > performance_dict[dv_name][1][index]:
					# check_no_switch_with_initial_disagreement(performance_dict["user"][index])
					# participant_decrease_dv += 1
					if pattern not in pattern_decrease_dict[dv_name]:
						# pattern_decrease_dict[pattern] = []
						pattern_decrease_dict[dv_name][pattern] = []
					pattern_decrease_dict[dv_name][pattern].append((performance_dict[dv_name][0][index] - performance_dict[dv_name][1][index]))
				# print(pattern, performance_dict["relative_positive_self_reliance"][0][index], performance_dict["relative_positive_self_reliance"][1][index])
				# pattern_decrease_dict[pattern].append(performance_dict["relative_positive_self_reliance"][0][index] - performance_dict["relative_positive_self_reliance"][1][index])
		# print("{} participants showed worse {}".format(participant_decrease_dv, dv_name))
		print(pattern_dict)
		for dv_name in diff_dict:
			print(dv_name, len(diff_dict[dv_name]))
		df = pd.DataFrame(diff_dict)
		from pingouin import pairwise_corr
		# for dv_name in ["accuracy", "appropriate_reliance", "relative_positive_ai_reliance", "relative_positive_self_reliance"]:
		# 	correlation, pvalue = spearmanr(diff_dict["miscalibration"], diff_dict[dv_name])
		# 	if pvalue < 0.05 / 4:
		# 		print("Spearman Correlation for miscalibration and {} is significant, p= {:.3f}, r= {:.3f}".format(dv_name, pvalue, correlation))
		# 	else:
		# 		print("Spearman Correlation for miscalibration and {} is insignificant, p= {:.3f}".format(dv_name, pvalue))
		res = pairwise_corr(df, columns=['miscalibration'], method='spearman', alternative='less', padjust='bonf').round(3)
		print(res)
		# for dv_name in pattern_decrease_dict:
		# 	print("-" * 17)
		# 	print("{} participants showed worse {}".format(sum([len(pattern_decrease_dict[dv_name][pattern]) for pattern in pattern_decrease_dict[dv_name]]), dv_name))
		# 	print(dv_name)
		# 	for pattern in pattern_decrease_dict[dv_name]:
		# 		print(pattern, len(pattern_decrease_dict[dv_name][pattern]), np.sum(pattern_decrease_dict[dv_name][pattern]))
		# # for pattern in pattern_decrease_dict:
		# # 	print(pattern, np.mean(pattern_decrease_dict[pattern]))
		# print("-" * 17)


	def get_miscaibration_pattern_overestimation(mis_1, mis_2):
		if mis_2 == 0:
			pattern = "Over -> Accurate"
		elif mis_2 < 0:
			pattern = "Over -> under"
		else:
			pattern = "Over -> Over"
		# elif mis_2 > 0 and mis_2 < mis_1:
		# 	pattern = "Over -> over, improved"
		# elif mis_2 <0 and abs(mis_2) < abs(mis_1):
		# 	pattern = "Over -> under, improved"
		# elif mis_2 <0 and abs(mis_2) >= abs(mis_1):
		# 	pattern = "Over -> under, not improved"
		# elif mis_1 == mis_2:
		# 	pattern = "not improved"
		# else:
		# 	pattern = "getting worse with overestimation"
		return pattern

	if mode == "overestimation":
		# try to see how the ones who showed decreased performance in miscalibration change
		dv_name = "accuracy"
		# print(f"For participants showed decreased accuracy")
		print(len(performance_dict[dv_name][0]), len(performance_dict[dv_name][1]))
		pattern_dict = {}
		pattern_increase_dict = {}
		number_participants = len(performance_dict[dv_name][0])
		# participant_increase_accuracy = 0
		diff_dict = {}
		for index in range(number_participants):
			pattern = get_miscaibration_pattern_overestimation(performance_dict["miscalibration"][0][index], performance_dict["miscalibration"][1][index])
			if pattern not in pattern_dict:
				pattern_dict[pattern] = 0
			pattern_dict[pattern] += 1
			for dv_name in ["miscalibration", "accuracy", "appropriate_reliance", "relative_positive_ai_reliance", "relative_positive_self_reliance"]:
				if dv_name not in diff_dict:
					diff_dict[dv_name] = []
				diff_dict[dv_name].append(performance_dict[dv_name][1][index] - performance_dict[dv_name][0][index])
				if dv_name not in pattern_increase_dict:
					pattern_increase_dict[dv_name] = {}
				if performance_dict[dv_name][0][index] < performance_dict[dv_name][1][index]:
					# check_no_switch_with_initial_disagreement(performance_dict["user"][index])
					# participant_decrease_dv += 1
					if pattern not in pattern_increase_dict[dv_name]:
						pattern_increase_dict[dv_name][pattern] = []
					pattern_increase_dict[dv_name][pattern].append((performance_dict[dv_name][1][index] - performance_dict[dv_name][0][index]))
				
			# if performance_dict[dv_name][0][index] < performance_dict[dv_name][1][index]:
			# 	participant_increase_accuracy += 1
		for dv_name in diff_dict:
			print(dv_name, len(diff_dict[dv_name]))
		df = pd.DataFrame(diff_dict)
		from pingouin import pairwise_corr
		# for dv_name in ["accuracy", "appropriate_reliance", "relative_positive_ai_reliance", "relative_positive_self_reliance"]:
		# 	correlation, pvalue = spearmanr(diff_dict["miscalibration"], diff_dict[dv_name])
		# 	if pvalue < 0.05 / 4:
		# 		print("Spearman Correlation for miscalibration and {} is significant, p= {:.3f}, r= {:.3f}".format(dv_name, pvalue, correlation))
		# 	else:
		# 		print("Spearman Correlation for miscalibration and {} is insignificant, p= {:.3f}".format(dv_name, pvalue))
		res = pairwise_corr(df, columns=['miscalibration'], method='spearman', alternative='less', padjust='bonf').round(3)
		print(res)
		# print("{} participants showed better accuracy".format(participant_increase_accuracy))
		# print(pattern_dict)
		# for dv_name in pattern_increase_dict:
		# 	print("-" * 17)
		# 	print("{} participants showed larger {}".format(sum([len(pattern_increase_dict[dv_name][pattern]) for pattern in pattern_increase_dict[dv_name]]), dv_name))
		# 	print(dv_name)
		# 	for pattern in pattern_increase_dict[dv_name]:
		# 		print(pattern, len(pattern_increase_dict[dv_name][pattern]), np.sum(pattern_increase_dict[dv_name][pattern]))
		# print("-" * 17)

	# this part is code to compare with logic units-based explanations
	print("Mode {}".format(mode))
	number_participants_with_explanation = len(performance_dict_explanation["accuracy"][0])
	if number_participants_with_explanation > 3:
		print("-" * 17)
		print(f"Among them, we have {number_participants_with_explanation} participants with logic units-based explanation")
		print("Compare performance improvement with explanation vs no explanation")
		for dv_name in ["accuracy", "aggreement_fraction", "switching_fraction", "appropriate_reliance", "relative_positive_ai_reliance", "relative_positive_self_reliance", "trust"]:
			# print(f"For metric {dv_name}, wilcoxon test res:")
			# print("Mean %.3f\t%.3f"%(np.mean(performance_dict_explanation[dv_name][0]), np.mean(performance_dict_explanation[dv_name][1])))
			# res = wilcoxon(x=performance_dict_explanation[dv_name][0], y=performance_dict_explanation[dv_name][1])
			# print(res)
			improvement_list_explanation = []
			for metric1, metric2 in zip(performance_dict_explanation[dv_name][0], performance_dict_explanation[dv_name][1]):
				improvement_list_explanation.append(metric2 - metric1)
			improvement_list_no_explanation = []
			for metric1, metric2 in zip(performance_dict_no_explanation[dv_name][0], performance_dict_no_explanation[dv_name][1]):
				improvement_list_no_explanation.append(metric2 - metric1)
			compare_performance_improvement(dv_name, improvement_list_explanation, improvement_list_no_explanation)


if __name__ == "__main__":
	filename = "all_valid_data.csv"
	valid_users, approved_users = find_valid_users(filename, 4)
	user_condition_dict = get_user_conditions(filename, valid_users)
	answer_dict = load_answers()
	user_question_order = get_user_question_order(filename, valid_users)
	usertask_dict = read_decisions(filename, valid_users)
	user_trust_first, user_trust_second = calc_trust_in_automation(filename, valid_users)
	self_assessment_first, self_assessment_second, other_assessment_first, other_assessment_second, survey_percetage_first, survey_percetage_second = calc_miscalibration(filename, valid_users)
	
	with_tutorial_user_set = user_condition_dict["with tutorial, no xai"] | user_condition_dict["with tutorial, with xai"]
	users_with_explanation = user_condition_dict["with tutorial, with xai"]
	# print(f"For H4, we have {len(with_tutorial_user_set)} participants for analysis")
	# compare_performance(user_trust_first, user_trust_second, with_tutorial_user_set, user_question_order, usertask_dict, answer_dict, self_assessment_first, self_assessment_second, users_with_explanation)
	compare_performance(user_trust_first, user_trust_second, with_tutorial_user_set, user_question_order, usertask_dict, answer_dict, self_assessment_first, self_assessment_second, users_with_explanation, mode="underestimation")
	compare_performance(user_trust_first, user_trust_second, with_tutorial_user_set, user_question_order, usertask_dict, answer_dict, self_assessment_first, self_assessment_second, users_with_explanation, mode="overestimation")
	# compare_performance(user_trust_first, user_trust_second, with_tutorial_user_set, user_question_order, usertask_dict, answer_dict, self_assessment_first, self_assessment_second, users_with_explanation, mode="accurate")











