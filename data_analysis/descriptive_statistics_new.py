import numpy as np
import pandas as pd
import os
import sys
from util import UserPerformance
from util import find_valid_users, get_user_conditions, load_answers, get_user_question_order, read_decisions, calc_miscalibration, calc_user_reliance_measures
from util import calc_ATI_scale, calc_propensity_to_trust, calc_trust_in_automation, check_user_condition
import seaborn as sns
import matplotlib.pyplot as plt

def draw_bar_plot(participant_condition_dict):
	data_long_format = {}
	data_long_format["Dimension"] = []
	data_long_format["Value"] = []
	data_long_format["Self-assessment"] = []
	for condition in ["no tutorial, no xai", "with tutorial, no xai", "no tutorial, with xai", "with tutorial, with xai"]:
		# data_long_format["Dimension"].append(condition)
		# data_long_format["Value"].append(participant_condition_dict[condition][0])
		# data_long_format["property"].append("All")

		data_long_format["Dimension"].append(condition.replace(",", ",\n"))
		data_long_format["Value"].append(participant_condition_dict[condition][1])
		data_long_format["Self-assessment"].append("Under")

		data_long_format["Dimension"].append(condition.replace(",", ",\n"))
		data_long_format["Value"].append(participant_condition_dict[condition][2])
		data_long_format["Self-assessment"].append("Accurate")

		data_long_format["Dimension"].append(condition.replace(",", ",\n"))
		data_long_format["Value"].append(participant_condition_dict[condition][3])
		data_long_format["Self-assessment"].append("Over")

	df = pd.DataFrame(data_long_format, dtype=float)
	sns.set_theme(style="whitegrid")
	sns.set(font="Arial")
	# ax = sns.boxplot(data=df)
	ax = sns.barplot(x="Dimension", y="Value", hue="Self-assessment", data=df)
	ax.tick_params(labelsize=18)
	ax.set_xlabel("Condition", fontsize = 24)
	ax.set_ylabel("Participants", fontsize = 24)
	plt.setp(ax.get_legend().get_texts(), fontsize='16') # for legend text
	plt.setp(ax.get_legend().get_title(), fontsize='16') # for legend title
	plt.margins(0.015, tight=True)
	plt.show()


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
	variable_dict = {
		"user_id": [],
		"ATI": [],
		"TiA-Propensity": [],
		"Accuracy": [],
		"Agreement_fraction": [],
		"switching_fraction": [],
		"RAIR": [],
		"RSR": [],
		"TiA-Trust-first": [],
		"TiA-Trust-second": []
	}
	participant_condition_dict = {}
	for condition in ["no tutorial, no xai", "with tutorial, no xai", "no tutorial, with xai", "with tutorial, with xai"]:
		participant_condition_dict[condition] = [0, 0, 0, 0]
		# total_number of paricipant, participant with underestimation, participant with accurate self-assessment, participants with over estimation
	participant_condition_dict_2 = {}
	for condition in ["no tutorial, no xai", "with tutorial, no xai", "no tutorial, with xai", "with tutorial, with xai"]:
		participant_condition_dict_2[condition] = [0, 0, 0, 0]
		# total_number of paricipant, participant with underestimation, participant with accurate self-assessment, participants with over estimation
	
	for user in valid_users:
		variable_dict["user_id"].append(user)
		variable_dict["TiA-Propensity"].append(user_ptt_scale[user])
		variable_dict["ATI"].append(user_ATI_scale[user])

		tp_condition = check_user_condition(user, user_condition_dict)
		participant_condition_dict[tp_condition][0] += 1
		participant_condition_dict_2[tp_condition][0] += 1

		tp_order = user_question_order[user]
		first_group = tp_order[:6]
		second_group = tp_order[-6:]
		tp_performance = UserPerformance(username=user, question_order=tp_order)
		
		tp_correct, tp_agreement_fraction, tp_switching_fraction, tp_appropriate_reliance, initial_disagreement_1, relative_positive_ai_reliance, relative_positive_self_reliance = calc_user_reliance_measures(user, usertask_dict, answer_dict, first_group)
		# tp_accuracy = tp_correct / 6.0
		miscalibration_first = self_assessment_first[user] - tp_correct
		
		tp_correct, tp_agreement_fraction, tp_switching_fraction, tp_appropriate_reliance, initial_disagreement_2, relative_positive_ai_reliance, relative_positive_self_reliance = calc_user_reliance_measures(user, usertask_dict, answer_dict, second_group)
		# tp_accuracy = tp_correct / 6.0
		tp_performance.add_miscalibration(self_assessment=self_assessment_second[user], actual_correct_number=tp_correct, group="second_group")
		miscalibration_second = self_assessment_second[user] - tp_correct

		if miscalibration_first < 0:
			participant_condition_dict[tp_condition][1] += 1
			# underestimation
		elif miscalibration_first > 0:
			participant_condition_dict[tp_condition][3] += 1
			# overestimation
		else:
			participant_condition_dict[tp_condition][2] += 1
			# accurate self-assessment

		if miscalibration_second < 0:
			participant_condition_dict_2[tp_condition][1] += 1
			# underestimation
		elif miscalibration_second > 0:
			participant_condition_dict_2[tp_condition][3] += 1
			# overestimation
		else:
			participant_condition_dict_2[tp_condition][2] += 1
			# accurate self-assessment

		tp_correct, tp_agreement_fraction, tp_switching_fraction, tp_appropriate_reliance, initial_disagreement_1, relative_positive_ai_reliance, relative_positive_self_reliance = calc_user_reliance_measures(user, usertask_dict, answer_dict, first_group + second_group)
		tp_accuracy = tp_correct / 12.0
		variable_dict["Accuracy"].append(tp_accuracy)
		variable_dict["Agreement_fraction"].append(tp_agreement_fraction)
		variable_dict["switching_fraction"].append(tp_switching_fraction)
		variable_dict["RAIR"].append(relative_positive_ai_reliance)
		variable_dict["RSR"].append(relative_positive_self_reliance)
		variable_dict["TiA-Trust-first"].append(user_trust_first[user])
		variable_dict["TiA-Trust-second"].append(user_trust_second[user])
	draw_bar_plot(participant_condition_dict)
	draw_bar_plot(participant_condition_dict_2)
	# data_long_format = {}
	# data_long_format["Dimension"] = []
	# data_long_format["Value"] = []
	# data_long_format["Self-assessment"] = []
	# for condition in ["no tutorial, no xai", "with tutorial, no xai", "no tutorial, with xai", "with tutorial, with xai"]:
	# 	# data_long_format["Dimension"].append(condition)
	# 	# data_long_format["Value"].append(participant_condition_dict[condition][0])
	# 	# data_long_format["property"].append("All")

	# 	data_long_format["Dimension"].append(condition.replace(",", ",\n"))
	# 	data_long_format["Value"].append(participant_condition_dict[condition][1])
	# 	data_long_format["Self-assessment"].append("Under")

	# 	data_long_format["Dimension"].append(condition.replace(",", ",\n"))
	# 	data_long_format["Value"].append(participant_condition_dict[condition][2])
	# 	data_long_format["Self-assessment"].append("Accurate")

	# 	data_long_format["Dimension"].append(condition.replace(",", ",\n"))
	# 	data_long_format["Value"].append(participant_condition_dict[condition][3])
	# 	data_long_format["Self-assessment"].append("Over")
	# draw_bar_plot(data_long_format)

	for key_ in variable_dict:
		print(key_, len(variable_dict[key_]))
	for key_ in ["ATI", "TiA-Propensity", "Accuracy", "Agreement_fraction", "switching_fraction", "RAIR", "RSR", "TiA-Trust-first", "TiA-Trust-second"]:
		print(key_)
		print("mean:{:.3f} std:{:.2f}".format(np.mean(variable_dict[key_]), np.std(variable_dict[key_], ddof=1)))