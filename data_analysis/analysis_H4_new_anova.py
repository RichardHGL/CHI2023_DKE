import numpy as np
import pandas as pd
import os
import sys
from util import UserPerformance, check_user_condition, calc_ATI_scale, calc_propensity_to_trust, calc_trust_in_automation
from util import find_valid_users, get_user_conditions, load_answers, get_user_question_order, read_decisions, calc_miscalibration, calc_user_reliance_measures
from scipy.stats import kruskal, mannwhitneyu
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


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

def get_user_performance(user_question_order, usertask_dict, answer_dict, self_assessment_first, self_assessment_second, user_condition_dict):
	user2performance = {}
	participant_condition_dict = {}
	for condition in ["no tutorial, no xai", "with tutorial, no xai", "no tutorial, with xai", "with tutorial, with xai"]:
		participant_condition_dict[condition] = [0, 0, 0, 0]
		# total_number of paricipant, underestimation, accurate self-assessment, overestimation
	user_assessment_groups = {
		"Overestimation": [],
		"Accurate": [],
		"Underestimation": []
	}
	for user in user_question_order:
		tp_order = user_question_order[user]
		first_group = tp_order[:6]
		second_group = tp_order[-6:]
		tp_condition = check_user_condition(user, user_condition_dict)
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
		user2performance[user] = tp_performance
		participant_condition_dict[tp_condition][0] += 1
		# based on first batch, we identify participants tend to show over-estimation, under-estimation and accurate self-assessment
		if tp_performance.miscalibration["first_group"] > 0:
			participant_condition_dict[tp_condition][3] += 1
			user_assessment_groups["Overestimation"].append(user)
		elif tp_performance.miscalibration["first_group"] < 0:
			participant_condition_dict[tp_condition][1] += 1
			user_assessment_groups["Underestimation"].append(user)
		else:
			participant_condition_dict[tp_condition][2] += 1
			user_assessment_groups["Accurate"].append(user)

	for key_ in participant_condition_dict:
		print(key_, participant_condition_dict[key_])
	return user2performance, user_assessment_groups


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

	variable_dict = {
		# "user_id": [],
		# "ATI": [],
		# "TiA-Propensity": [],
		"Tutorial": [],
		"XAI": [],
		"Accuracy": [],
		"Agreement_fraction": [],
		"switching_fraction": [],
		"appropriate_reliance": [],
		"RAIR": [],
		"RSR": [],
		# "TiA_Trust_first": [],
		# "TiA_Trust_second": []
	}

	user2performance, user_assessment_groups = get_user_performance(user_question_order, usertask_dict, answer_dict, self_assessment_first, self_assessment_second, user_condition_dict)
	trust_list_with_tutorial = []
	trust_list_no_tutorial = []
	for user in valid_users:
		# variable_dict["user_id"].append(user)
		# variable_dict["TiA-Propensity"].append(user_ptt_scale[user])
		# variable_dict["ATI"].append(user_ATI_scale[user])

		tp_condition = check_user_condition(user, user_condition_dict)
		if tp_condition == "no tutorial, with xai":
			variable_dict["Tutorial"].append("No")
			variable_dict["XAI"].append("Yes")
		elif tp_condition == "with tutorial, with xai":
			variable_dict["Tutorial"].append("Yes")
			variable_dict["XAI"].append("Yes")
		elif tp_condition == "no tutorial, no xai":
			variable_dict["Tutorial"].append("No")
			variable_dict["XAI"].append("No")
		elif tp_condition == "with tutorial, no xai":
			variable_dict["Tutorial"].append("Yes")
			variable_dict["XAI"].append("No")
		else:
			raise NotImplementedError("Unknown condition {}".format(tp_condition)) 

		# tp_order = user_question_order[user]
		# first_group = tp_order[:6]
		# second_group = tp_order[-6:]
		tp_performance = user2performance[user].performance

		# considering the performance of the second batch
		variable_dict["Accuracy"].append(tp_performance["second_group"]["accuracy"])
		variable_dict["Agreement_fraction"].append(tp_performance["second_group"]["agreement_fraction"])
		variable_dict["switching_fraction"].append(tp_performance["second_group"]["switching_fraction"])
		variable_dict["appropriate_reliance"].append(tp_performance["second_group"]["appropriate_reliance"])
		variable_dict["RAIR"].append(tp_performance["second_group"]["relative_positive_ai_reliance"])
		variable_dict["RSR"].append(tp_performance["second_group"]["relative_positive_self_reliance"])
		# variable_dict["TiA_Trust_first"].append(user_trust_first[user])
		# variable_dict["TiA_Trust_second"].append(user_trust_second[user])

		# considering the performance difference / improvement
		# variable_dict["Accuracy"].append(tp_performance["second_group"]["accuracy"] - tp_performance["first_group"]["accuracy"])
		# variable_dict["Agreement_fraction"].append(tp_performance["second_group"]["agreement_fraction"] - tp_performance["first_group"]["agreement_fraction"])
		# variable_dict["switching_fraction"].append(tp_performance["second_group"]["switching_fraction"] - tp_performance["first_group"]["switching_fraction"])
		# variable_dict["Accuracy-wid"].append(tp_performance["second_group"]["appropriate_reliance"] - tp_performance["first_group"]["appropriate_reliance"])
		# variable_dict["RAIR"].append(tp_performance["second_group"]["relative_positive_ai_reliance"] - tp_performance["first_group"]["relative_positive_ai_reliance"])
		# variable_dict["RSR"].append(tp_performance["second_group"]["relative_positive_self_reliance"] - tp_performance["first_group"]["relative_positive_self_reliance"])
		# # variable_dict["TiA_Trust_first"].append(user_trust_first[user])
		# variable_dict["TiA_Trust_second"].append(user_trust_second[user] - user_trust_first[user])

		# if variable_dict["Tutorial"][-1] == "Yes":
		# 	trust_list_with_tutorial.append(user_trust_second[user])
		# else:
		# 	trust_list_no_tutorial.append(user_trust_second[user])
	df = pd.DataFrame(variable_dict)
	print(df.shape)

	# Performing two-way ANOVA
	# model = ols('Accuracy ~ C(Tutorial) + C(XAI) + C(Tutorial):C(XAI)', data=df).fit()
	model = ols('Accuracy ~ C(Tutorial) * C(XAI)', data=df).fit()
	res = sm.stats.anova_lm(model, typ=2)
	print("Accuracy")
	print(res)
	print("-" * 17)

	model = ols('Agreement_fraction ~ C(Tutorial) * C(XAI)', data=df).fit()
	res = sm.stats.anova_lm(model, typ=2)
	print("Agreement_fraction")
	print(res)
	print("-" * 17)

	model = ols('switching_fraction ~ C(Tutorial) * C(XAI)', data=df).fit()
	res = sm.stats.anova_lm(model, typ=2)
	print("switching_fraction")
	print(res)
	print("-" * 17)

	model = ols('appropriate_reliance ~ C(Tutorial) * C(XAI)', data=df).fit()
	res = sm.stats.anova_lm(model, typ=2)
	print("Accuracy-wid")
	print(res)
	print("-" * 17)

	model = ols('RAIR ~ C(Tutorial) * C(XAI)', data=df).fit()
	res = sm.stats.anova_lm(model, typ=2)
	print("RAIR")
	print(res)
	print("-" * 17)

	model = ols('RSR ~ C(Tutorial) * C(XAI)', data=df).fit()
	res = sm.stats.anova_lm(model, typ=2)
	print("RSR")
	print(res)
	print("-" * 17)

	# we found significant impact of Tutorial on TiA-Trust, use post-hoc analysis:
	# tukey = pairwise_tukeyhsd(endog=df['TiA_Trust_second'], groups=df['Tutorial'], alpha=0.0125)
	# print(tukey)
	# print(len(trust_list_with_tutorial), len(trust_list_no_tutorial))
	# print("With Tutorial, M: {:.2f} , SD: {:.2f}".format(np.mean(trust_list_with_tutorial), np.std(trust_list_with_tutorial)))
	# print("No Tutorial, M: {:.2f} , SD: {:.2f}".format(np.mean(trust_list_no_tutorial), np.std(trust_list_no_tutorial)))

	for condition in user_condition_dict:
		print(condition)
		print("-" * 17)
		print("first_batch")
		variable_dict = {
			"Accuracy": [],
			"Agreement_fraction": [],
			"switching_fraction": [],
			"Accuracy-wid": [],
			"RAIR": [],
			"RSR": [],
			"TiA_Trust_second": []
		}
		for user in user_condition_dict[condition]:
			tp_performance = user2performance[user].performance

			variable_dict["Accuracy"].append(tp_performance["first_group"]["accuracy"])
			variable_dict["Agreement_fraction"].append(tp_performance["first_group"]["agreement_fraction"])
			variable_dict["switching_fraction"].append(tp_performance["first_group"]["switching_fraction"])
			variable_dict["Accuracy-wid"].append(tp_performance["first_group"]["appropriate_reliance"])
			variable_dict["RAIR"].append(tp_performance["first_group"]["relative_positive_ai_reliance"])
			variable_dict["RSR"].append(tp_performance["first_group"]["relative_positive_self_reliance"])
		for variable in ["Accuracy", "Agreement_fraction", "switching_fraction", "Accuracy-wid", "RAIR", "RSR"]:
			print(variable)
			print("mean: {:.2f}, std: {:.2f}".format(np.mean(variable_dict[variable]), np.std(variable_dict[variable])))
		
		print("-"* 17)
		print("second_batch")
		variable_dict = {
			"Accuracy": [],
			"Agreement_fraction": [],
			"switching_fraction": [],
			"Accuracy-wid": [],
			"RAIR": [],
			"RSR": [],
			"TiA_Trust_second": []
		}
		for user in user_condition_dict[condition]:
			tp_performance = user2performance[user].performance

			variable_dict["Accuracy"].append(tp_performance["second_group"]["accuracy"])
			variable_dict["Agreement_fraction"].append(tp_performance["second_group"]["agreement_fraction"])
			variable_dict["switching_fraction"].append(tp_performance["second_group"]["switching_fraction"])
			variable_dict["Accuracy-wid"].append(tp_performance["second_group"]["appropriate_reliance"])
			variable_dict["RAIR"].append(tp_performance["second_group"]["relative_positive_ai_reliance"])
			variable_dict["RSR"].append(tp_performance["second_group"]["relative_positive_self_reliance"])
		for variable in ["Accuracy", "Agreement_fraction", "switching_fraction", "Accuracy-wid", "RAIR", "RSR"]:
			print(variable)
			print("mean: {:.2f}, std: {:.2f}".format(np.mean(variable_dict[variable]), np.std(variable_dict[variable])))

	# for condition in user_condition_dict:
	# 	print("-"* 17)
	# 	print(condition)
	# 	print("-"* 17)
	# 	print("second_batch - first batch")
	# 	variable_dict = {
	# 		"Accuracy": [],
	# 		"Agreement_fraction": [],
	# 		"switching_fraction": [],
	# 		"Accuracy-wid": [],
	# 		"RAIR": [],
	# 		"RSR": [],
	# 		"TiA_Trust_second": []
	# 	}
	# 	for user in user_condition_dict[condition]:
	# 		tp_performance = user2performance[user].performance

	# 		variable_dict["Accuracy"].append(tp_performance["second_group"]["accuracy"] - tp_performance["first_group"]["accuracy"])
	# 		variable_dict["Agreement_fraction"].append(tp_performance["second_group"]["agreement_fraction"] - tp_performance["first_group"]["agreement_fraction"])
	# 		variable_dict["switching_fraction"].append(tp_performance["second_group"]["switching_fraction"] - tp_performance["first_group"]["switching_fraction"])
	# 		variable_dict["Accuracy-wid"].append(tp_performance["second_group"]["appropriate_reliance"] - tp_performance["first_group"]["appropriate_reliance"])
	# 		variable_dict["RAIR"].append(tp_performance["second_group"]["relative_positive_ai_reliance"] - tp_performance["first_group"]["relative_positive_ai_reliance"])
	# 		variable_dict["RSR"].append(tp_performance["second_group"]["relative_positive_self_reliance"] - tp_performance["first_group"]["relative_positive_self_reliance"])
	# 	print(len(variable_dict["Accuracy"]))
	# 	for variable in ["Accuracy", "Agreement_fraction", "switching_fraction", "Accuracy-wid", "RAIR", "RSR"]:
	# 		print(variable)
	# 		print("mean: {:.2f}, std: {:.2f}".format(np.mean(variable_dict[variable]), np.std(variable_dict[variable])))
		


	# users_with_tutorial = user_condition_dict["with tutorial, no xai"] | user_condition_dict["with tutorial, with xai"]
	# users_without_tutorial = user_condition_dict["no tutorial, no xai"] | user_condition_dict["no tutorial, with xai"]

	# for refer_users in [users_with_tutorial, users_without_tutorial]:
	# 	user_trust_dict = {}
	# 	for group in ["Underestimation", "Accurate", "Overestimation"]:
	# 		user_trust_dict[group] = []
	# 		for user in user_assessment_groups[group]:
	# 			if user not in refer_users:
	# 				continue
	# 			user_trust_dict[group].append(user_trust_second[user])
	# 	data_list_1, data_list_2, data_list_3 = user_trust_dict["Underestimation"], user_trust_dict["Overestimation"], user_trust_dict["Accurate"]
	# 	statistic, pvalue = kruskal(data_list_1, data_list_2, data_list_3)
	# 	print("kruskal test result: H:{:.2f}, p:{:.3f}".format(statistic, pvalue))
	# 	print("Mean: M(underestimation):{:.2f}, SD(underestimation):{:.2f}".format(np.mean(data_list_1), np.std(data_list_1)))
	# 	print("Mean: M(accurate_self_assessment):{:.2f}, SD(accurate_self_assessment):{:.2f}".format(np.mean(data_list_3), np.std(data_list_3)))
	# 	print("Mean: M(overestimation):{:.2f}, SD(overestimation):{:.2f}".format(np.mean(data_list_2), np.std(data_list_2)))
	# 	if pvalue < 0.05 / 4:
	# 		post_hoc_comparison(data_list_1, data_list_2, "underestimation", "overestimation")
	# 		post_hoc_comparison(data_list_3, data_list_2, "accurate_self_assessment", "overestimation")
	# 		# we assume participants with accurate self assessment and underestimation rely more than participants with overestimation
	# 		post_hoc_comparison(data_list_1, data_list_3, "underestimation", "accurate_self_assessment")
	# 		# post_hoc_comparison(data_list_2, data_list_3, "overestimation", "accurate_self_assessment")
	# 	print("-" * 17)

