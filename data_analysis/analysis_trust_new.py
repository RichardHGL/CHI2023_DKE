import numpy as np
import pandas as pd
import pingouin as pg
import os
import sys
from util import UserPerformance
from util import find_valid_users, get_user_conditions, load_answers, get_user_question_order, read_decisions, calc_miscalibration, calc_user_reliance_measures, calc_explanation_usefulness
from util import calc_ATI_scale, calc_propensity_to_trust, calc_trust_in_automation, check_user_condition
from scipy.stats import wilcoxon, kruskal, spearmanr, mannwhitneyu
from pingouin import ancova

def compare_calibration_effect(list_1, list_2):
	statistic, pvalue = kruskal(list_1, list_2)
	print("Compare self-assessment improvement with explanation vs no explanation")
	print("kruskal test result: H:{:.3f}, p:{:.3f}".format(statistic, pvalue))
	print("Mean: M(Exp):{:.3f}, M(No Exp):{:.3f}".format(np.mean(list_1), np.mean(list_2)))
	print("-" * 17)

def test_wilcoxon(list_1, list_2):
	assert len(list_1) == len(list_2)
	res = wilcoxon(x=list_1, y=list_2)
	print("Considering all participants with tutorial, the results are:")
	print("Mean: %.3f\t%.3f"%(np.mean(list_1), np.mean(list_2)))
	print(res)
	print("-" * 17)

def get_user_miscalibration(user_question_order, usertask_dict, answer_dict, self_assessment_first, self_assessment_second):
	user_miscalibration_first = {}
	user_miscalibration_second = {}
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
		
		tp_correct, tp_agreement_fraction, tp_switching_fraction, tp_appropriate_reliance, initial_disagreement_2, relative_positive_ai_reliance, relative_positive_self_reliance = calc_user_reliance_measures(user, usertask_dict, answer_dict, second_group)
		tp_accuracy = tp_correct / 6.0
		tp_performance.add_performance(accuracy=tp_accuracy, agreement_fraction=tp_agreement_fraction, switching_fraction=tp_switching_fraction, 
			appropriate_reliance=tp_appropriate_reliance, relative_positive_ai_reliance=relative_positive_ai_reliance, 
			relative_positive_self_reliance=relative_positive_self_reliance, group="second_group")
		tp_performance.add_miscalibration(self_assessment=self_assessment_second[user], actual_correct_number=tp_correct, group="second_group")

		user_miscalibration_first[user] = tp_performance.miscalibration["first_group"]
		user_miscalibration_second[user] = tp_performance.miscalibration["second_group"]

		tp_correct, tp_agreement_fraction, tp_switching_fraction, tp_appropriate_reliance, initial_disagreement_1, relative_positive_ai_reliance, relative_positive_self_reliance = calc_user_reliance_measures(user, usertask_dict, answer_dict, first_group + second_group)
		tp_accuracy = tp_correct / 12.0
		tp_performance.add_performance(accuracy=tp_accuracy, agreement_fraction=tp_agreement_fraction, switching_fraction=tp_switching_fraction, 
			appropriate_reliance=tp_appropriate_reliance, relative_positive_ai_reliance=relative_positive_ai_reliance, 
			relative_positive_self_reliance=relative_positive_self_reliance, group="overall")
		user2performance[user] = tp_performance

		# tp_correct, tp_agreement_fraction, tp_switching_fraction, tp_appropriate_reliance, initial_disagreement_1, relative_positive_ai_reliance, relative_positive_self_reliance = calc_user_reliance_measures(user, usertask_dict, answer_dict, first_group + second_group)
		# tp_accuracy = tp_correct / 12.0

		# # performance_list.append(tp_performance)
		# user_accuracy_list.append((user, tp_accuracy))

	return user_miscalibration_first, user_miscalibration_second, user2performance


def ancova_analysis(user_trust_first, user_trust_second, user_condition_dict, user_ATI_scale, user_ptt_scale, user_miscalibration_first, user_miscalibration_second, user2performance):
	user_data = {
		"user_id": [],
		"miscalibration": [],
		# "Overestimation": [],
		# "Accurate_Self_Assessment": [],
		"condition": [],
		"ATI": [],
		"TiA-Propensity": [],
		"TiA-Trust": [],
		"accuracy": [],
		"agreement_fraction": [],
		"switching_fraction": [],
		"relative_positive_ai_reliance": [],
		"relative_positive_self_reliance": []
	}
	for user in user_trust_first:
		user_data["user_id"].append(user)
		tp_condition = check_user_condition(user, user_condition_dict)
		user_data["condition"].append(tp_condition)
		if user_miscalibration_first[user] > 0:
			user_data["miscalibration"].append("Overestimation") 
		elif user_miscalibration_first[user] < 0:
			user_data["miscalibration"].append("Underestimation")
		else:
			user_data["miscalibration"].append("Accurate")
		# if user_miscalibration_first[user] > 0 and user_miscalibration_second[user] > 0:
		# 	user_data["Overestimation"].append("True")
		# else:
		# 	user_data["Overestimation"].append("False")
		# if abs(user_miscalibration_first[user]) <= 1 and abs(user_miscalibration_second[user]) <= 1:
		# 	user_data["Accurate_Self_Assessment"].append("True")
		# else:
		# 	user_data["Accurate_Self_Assessment"].append("False")
		user_data["ATI"].append(user_ATI_scale[user])
		user_data["TiA-Propensity"].append(user_ptt_scale[user])
		user_data["TiA-Trust"].append((user_trust_first[user] + user_trust_second[user]) / 2.0)
		user_data["accuracy"].append(user2performance[user].performance["overall"]["accuracy"])
		user_data["agreement_fraction"].append(user2performance[user].performance["overall"]["agreement_fraction"])
		user_data["switching_fraction"].append(user2performance[user].performance["overall"]["switching_fraction"])
		user_data["relative_positive_ai_reliance"].append(user2performance[user].performance["overall"]["relative_positive_ai_reliance"])
		user_data["relative_positive_self_reliance"].append(user2performance[user].performance["overall"]["relative_positive_self_reliance"])
	df = pd.DataFrame(user_data)
	print(df.shape)
	# print("For all participants, compare with Overestimation vs others")
	# print(ancova(data=df, dv='TiA-Trust', covar=['ATI', 'TiA-Propensity'], between='Overestimation', effsize='n2'))
	print("For all participants, compare with Underestimation, accurate, Overestimation")
	print(ancova(data=df, dv='TiA-Trust', covar=['ATI', 'TiA-Propensity'], between='miscalibration', effsize='n2'))

	print("-" * 34)

	print("For all participants, compare with experimental conditions")
	print(ancova(data=df, dv='TiA-Trust', covar=['ATI', 'TiA-Propensity'], between='condition', effsize='n2'))

	print("-" * 34)

	# print("For all participants, compare with Accurate self-assessment vs others")
	# print(ancova(data=df, dv='TiA-Trust', covar=['ATI', 'TiA-Propensity'], between='Accurate_Self_Assessment', effsize='n2'))
	# print("-" * 34)

	correlation, pvalue = spearmanr(user_data["TiA-Propensity"], user_data["TiA-Trust"])
	print("Variable {} and variable {} have spearman correlation {:.3f} and pvalue {:.3f}".format("TiA-Propensity", "TiA-Trust", correlation, pvalue))

	correlation, pvalue = spearmanr(user_data["TiA-Propensity"], user_data["accuracy"])
	print("Variable {} and variable {} have spearman correlation {:.3f} and pvalue {:.3f}".format("TiA-Propensity", "accuracy", correlation, pvalue))

	correlation, pvalue = spearmanr(user_data["TiA-Propensity"], user_data["agreement_fraction"])
	print("Variable {} and variable {} have spearman correlation {:.3f} and pvalue {:.3f}".format("TiA-Propensity", "agreement_fraction", correlation, pvalue))

	correlation, pvalue = spearmanr(user_data["TiA-Propensity"], user_data["switching_fraction"])
	print("Variable {} and variable {} have spearman correlation {:.3f} and pvalue {:.3f}".format("TiA-Propensity", "switching_fraction", correlation, pvalue))

	correlation, pvalue = spearmanr(user_data["TiA-Propensity"], user_data["relative_positive_ai_reliance"])
	print("Variable {} and variable {} have spearman correlation {:.3f} and pvalue {:.3f}".format("TiA-Propensity", "RAIR", correlation, pvalue))

	correlation, pvalue = spearmanr(user_data["TiA-Propensity"], user_data["relative_positive_self_reliance"])
	print("Variable {} and variable {} have spearman correlation {:.3f} and pvalue {:.3f}".format("TiA-Propensity", "RSR", correlation, pvalue))
	

def ancova_analysis_with_explanation(user_trust_first, user_trust_second, user_ATI_scale, user_ptt_scale, user_miscalibration_first, user_miscalibration_second, users_with_explanation, explanation_usefulness_dict):
	user_data = {
		"user_id": [],
		"Overestimation": [],
		"Accurate_Self_Assessment": [],
		"ATI": [],
		"TiA-Propensity": [],
		"TiA-Trust": [],
		"helpfulness": []
	}
	helpfulness_overestimation = []
	helpfulness_other = []
	for user in users_with_explanation:
		user_data["user_id"].append(user)
		if user_miscalibration_first[user] > 0 and user_miscalibration_second[user] > 0:
			user_data["Overestimation"].append("True")
			helpfulness_overestimation.append(explanation_usefulness_dict[user])
		else:
			user_data["Overestimation"].append("False")
			helpfulness_other.append(explanation_usefulness_dict[user])
		if abs(user_miscalibration_first[user]) <= 1 and abs(user_miscalibration_second[user]) <= 1:
			user_data["Accurate_Self_Assessment"].append("True")
		else:
			user_data["Accurate_Self_Assessment"].append("False")
		user_data["ATI"].append(user_ATI_scale[user])
		user_data["TiA-Propensity"].append(user_ptt_scale[user])
		user_data["TiA-Trust"].append((user_trust_first[user] + user_trust_second[user]) / 2.0)
		assert explanation_usefulness_dict[user] > 0.0
		user_data["helpfulness"].append(explanation_usefulness_dict[user])
	# df = pd.DataFrame(user_data)
	# print("For all participants, compare with Overestimation vs others")
	# print(ancova(data=df, dv="helpfulness", covar=['ATI', 'TiA-Propensity'], between='Overestimation', effsize='n2'))

	# print("-" * 34)

	# print("For all participants, compare with Accurate self-assessment vs others")
	# print(ancova(data=df, dv="helpfulness", covar=['ATI', 'TiA-Propensity'], between='Accurate_Self_Assessment', effsize='n2'))
	# print("-" * 34)

	# correlation, pvalue = spearmanr(user_data["ATI"], user_data["helpfulness"])
	# print("Variable {} and variable {} have spearman correlation {:.3f} and pvalue {:.3f}".format("ATI", "TiA-Trust", correlation, pvalue))

	print("Overestimation, Mean: {:.2f} SD {:.2f}".format(np.mean(helpfulness_overestimation), np.std(helpfulness_overestimation)))
	print("Other, Mean: {:.2f} SD {:.2f}".format(np.mean(helpfulness_other), np.std(helpfulness_other)))

	statistic, pvalue = kruskal(helpfulness_overestimation, helpfulness_other)
	print("Overestimation")
	print("kruskal test result: H:{:.2f}, p:{:.3f}".format(statistic, pvalue))

	# post-hoc analysis
	statistic, pvalue = mannwhitneyu(helpfulness_overestimation, helpfulness_other, alternative='two-sided')
	print("Alternative overestimation <> other,", "pvalue %.4f"%pvalue, "statistic %.4f"%statistic)
	statistic, pvalue = mannwhitneyu(helpfulness_overestimation, helpfulness_other, alternative='less')
	print("Alternative overestimation < other,", "pvalue %.4f"%pvalue, "statistic %.4f"%statistic)


def compare_trust(user_set, user_trust_first, user_trust_second, users_with_explanation):
	trust_list_1 = []
	trust_list_2 = []
	trust_list_1_explanation = []
	trust_list_2_explanation = []
	trust_list_1_no_explanation = []
	trust_list_2_no_explanation = []
	obj_list = []
	for user in user_set:
		trust_1 = user_trust_first[user]
		trust_2 = user_trust_second[user]
		trust_list_1.append(trust_1)
		trust_list_2.append(trust_2)
		if user in users_with_explanation:
			trust_list_1_explanation.append(trust_1)
			trust_list_2_explanation.append(trust_2)
		else:
			trust_list_1_no_explanation.append(trust_1)
			trust_list_2_no_explanation.append(trust_2)
	print(f"{len(trust_list_1)} participants considered")
	test_wilcoxon(trust_list_1, trust_list_2)

	# improvement_list_explanation = []
	# for mis1, mis2 in zip(miscalibration_list_first_explanation, miscalibration_list_second_explanation):
	# 	calibration_improvement = abs(mis1) - abs(mis2) # abs(mis2) is expected to be less than abs(mis1)
	# 	improvement_list_explanation.append(calibration_improvement)

	# improvement_list_no_explanation = []
	# for mis1, mis2 in zip(miscalibration_list_first_no_explanation, miscalibration_list_second_no_explanation):
	# 	calibration_improvement = abs(mis1) - abs(mis2) # abs(mis2) is expected to be less than abs(mis1)
	# 	improvement_list_no_explanation.append(calibration_improvement)

	# compare_calibration_effect(improvement_list_explanation, improvement_list_no_explanation)


# H3: the effect of calibration tutorial on self-assessment.


if __name__ == "__main__":
	filename = "all_valid_data.csv"
	valid_users, approved_users = find_valid_users(filename, 4)
	user_condition_dict = get_user_conditions(filename, valid_users)
	answer_dict = load_answers()
	user_question_order = get_user_question_order(filename, valid_users)
	usertask_dict = read_decisions(filename, valid_users)
	self_assessment_first, self_assessment_second, other_assessment_first, other_assessment_second, survey_percetage_first, survey_percetage_second = calc_miscalibration(filename, valid_users)
	user_trust_first, user_trust_second = calc_trust_in_automation(filename, valid_users)
	user_miscalibration_first, user_miscalibration_second, user2performance = get_user_miscalibration(user_question_order, usertask_dict, answer_dict, self_assessment_first, self_assessment_second)
	user_ATI_scale = calc_ATI_scale(filename, valid_users)
	user_ptt_scale = calc_propensity_to_trust(filename, valid_users)

	with_tutorial_user_set = user_condition_dict["with tutorial, no xai"] | user_condition_dict["with tutorial, with xai"]
	users_with_explanation = user_condition_dict["with tutorial, with xai"]
	compare_trust(with_tutorial_user_set, user_trust_first, user_trust_second, users_with_explanation)
	# analysis the trust difference before and after tutorial, only conditions with tutorial considered

	# # analysis of covariates impact on shaping trust, also take the perceived helpfulness of explanation into consideration.
	ancova_analysis(user_trust_first, user_trust_second, user_condition_dict, user_ATI_scale, user_ptt_scale, user_miscalibration_first, user_miscalibration_second, user2performance)

	# users_with_explanation = user_condition_dict["with tutorial, with xai"] | user_condition_dict["no tutorial, with xai"]
	# # analysis of covariates impact on shaping trust, also take the perceived helpfulness of explanation into consideration.
	# explanation_usefulness_dict = calc_explanation_usefulness(filename, users_with_explanation)
	# ancova_analysis_with_explanation(user_trust_first, user_trust_second, user_ATI_scale, user_ptt_scale, user_miscalibration_first, user_miscalibration_second, users_with_explanation, explanation_usefulness_dict)









