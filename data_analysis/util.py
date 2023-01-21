import numpy as np
import pandas as pd
import os
import sys

data_folder = "../anonymous_data"


# Random orders (with constraints) of questions
question0 = [1, 4, 5, 2, 6, 3, 0, 7, 10, 11, 8, 9, 14, 17, 13, 18, 12, 16, 15]
question1 = [12, 17, 13, 14, 18, 16, 15, 9, 10, 11, 8, 7, 5, 0, 2, 6, 1, 4, 3]
question2 = [1, 4, 5, 0, 6, 3, 2, 8, 10, 11, 9, 7, 15, 13, 17, 18, 14, 16, 12]
question3 = [12, 16, 17, 14, 18, 13, 15, 10, 8, 11, 9, 7, 2, 4, 5, 6, 0, 3, 1]
question4 = [3, 1, 4, 2, 6, 0, 5, 8, 10, 11, 9, 7, 17, 13, 15, 18, 12, 16, 14]
# Random orders (with constraints) of questions
question5 = [16, 13, 14, 17, 18, 12, 15, 9, 10, 11, 8, 7, 0, 5, 1, 6, 3, 4, 2]
question6 = [4, 1, 3, 0, 6, 2, 5, 8, 7, 11, 10, 9, 17, 16, 13, 18, 15, 14, 12]
question7 = [12, 14, 16, 17, 18, 15, 13, 8, 7, 11, 10, 9, 3, 2, 4, 6, 1, 5, 0]
question8 = [2, 0, 4, 5, 6, 3, 1, 8, 9, 11, 7, 10, 16, 15, 17, 18, 12, 13, 14]
question9 = [12, 14, 16, 15, 18, 13, 17, 10, 9, 11, 8, 7, 0, 4, 5, 6, 3, 2, 1]


def remove_attention_check(order_list):
	new_list = []
	for element in order_list:
		if element in [6, 11, 18]:
			continue
		new_list.append(element)
	return new_list

question_order_dict = {
	0: remove_attention_check(question0),
	1: remove_attention_check(question1),
	2: remove_attention_check(question2),
	3: remove_attention_check(question3),
	4: remove_attention_check(question4),
	5: remove_attention_check(question5),
	6: remove_attention_check(question6),
	7: remove_attention_check(question7),
	8: remove_attention_check(question8),
	9: remove_attention_check(question9)
}


ID_key = ["username"]
condition_keys = ["tutorial", "XAI"]
order_keys = ["question_order"]
first_decision_keys = ["question0", "question1", "question2", "question3", "question4", "question5", "question7", "question8", "question9", "question10", 
	"question12", "question13", "question14", "question15", "question16", "question17"]
second_decision_keys = ["advice0", "advice1", "advice2", "advice3", "advice4", "advice5", "advice7", "advice8", "advice9", "advice10",
	 "advice12", "advice13", "advice14", "advice15", "advice16", "advice17"]
attention_keys = ["attention_ati", "attention6", "attention11", "attention18"]
survey_keys = ["surveySelf1", "surveySelf2", "surveyOther1", "surveyOther2", "surveyPercentage1", "surveyPercentage2"]

def load_answers():
	reserved_features = ["answer", "id_string", "AI-advice"]
	answer_file = os.path.join(data_folder, "selected_samples.csv")
	answer_df = pd.read_csv(answer_file, usecols=reserved_features)
	answer_list = answer_df.values.tolist()
	answer_dict = {}
	for q_id, (answer, id_string, advice) in enumerate(answer_list):
		answer_dict[q_id] = (answer, advice, id_string)
		# the correct answer and the advice provided
	return answer_dict


def calc_mean(value_list):
	return np.mean(value_list)

def reverse_code(value, max_scale):
	return max_scale + 1 - value

# scale in [1: max_scale]
class questionnaire(object):
	
	def __init__(self, values, reverse_code_index, max_scale=6, questionnaire_size=0, value_add_one=False):
		self.questionnaire_size = questionnaire_size
		assert len(values) == self.questionnaire_size
		tp_values = []
		for i, value in enumerate(values):
			if value_add_one:
				value_ = float(value) + 1
			else:
				value_ = float(value)
			if (i + 1) in reverse_code_index:
				tp_values.append(reverse_code(value_, max_scale))
			else:
				tp_values.append(value_)
		self.values = tp_values
		self.max_scale = max_scale

	def calc_value(self):
		return calc_mean(self.values)

def read_completion_time(reserved_users=None):
	filename = os.path.join(data_folder, "demographic.csv")
	ATI_keys = ["Participant id", "Time taken"]
	df = pd.read_csv(filename, usecols=ATI_keys)
	if reserved_users is not None:
		# filter some invalid users
		df = df.drop(df[~df['Participant id'].isin(reserved_users)].index)
	data_list = df.values.tolist()
	user2time = {}
	for user, time in data_list:
		user2time[user] = float(time)
	return user2time

def calc_ATI_scale(filename, reserved_users=None):
	filename = os.path.join(data_folder, filename)
	ATI_keys = ["username", "ati1", "ati2", "ati3", "ati4", "ati5", "ati6", "ati7", "ati8", "ati9"]
	df = pd.read_csv(filename, usecols=ATI_keys)
	if reserved_users is not None:
		# filter some invalid users
		df = df.drop(df[~df['username'].isin(reserved_users)].index)
	user_task_list = df.values.tolist()
	user_ATI_scale = {}
	reverse_code_index = [3, 6, 8]
	max_scale = 6
	questionnaire_size = 9
	# reverse code for question 3, 6, 8
	for tuple_ in user_task_list:
		# user_id, answer_1, answer_2, answer_3, answer_4, answer_5, answer_6, answer_7, answer_8, answer_9, __ = tuple_
		# reverse code for question 3, 6, 8
		# answer_3 = 7 - answer_3
		# answer_6 = 7 - answer_6
		# answer_8 = 7 - answer_8
		# ATI_scale = (answer_1 + answer_2 + answer_3 + answer_4 + answer_5 + answer_6 + answer_7 + answer_8 + answer_9) / 9.0
		ATI_scale = questionnaire(tuple_[1:], reverse_code_index, max_scale=max_scale, questionnaire_size=questionnaire_size, value_add_one=True).calc_value()
		user_id = tuple_[0]
		user_ATI_scale[user_id] = ATI_scale
	return user_ATI_scale

def calc_explanation_usefulness(filename, reserved_users=None):
	filename = os.path.join(data_folder, filename)
	ATI_keys = ["username", "xai_question"]
	df = pd.read_csv(filename, usecols=ATI_keys)
	if reserved_users is not None:
		# filter some invalid users
		df = df.drop(df[~df['username'].isin(reserved_users)].index)
	user_task_list = df.values.tolist()
	user_usefulness_dict = {}
	for user, usefulness in user_task_list:
		user_usefulness_dict[user] = int(usefulness) + 1
		# start with 0, so we add one
	return user_usefulness_dict

def calc_propensity_to_trust(filename, reserved_users=None):
	filename = os.path.join(data_folder, filename)
	usecols = ["username", "pt1", "pt2", "pt3"]
	df = pd.read_csv(filename, usecols=usecols)
	if reserved_users is not None:
		# filter some invalid users
		df = df.drop(df[~df['username'].isin(reserved_users)].index)
	user_task_list = df.values.tolist()
	user_ptt_scale = {}
	reverse_code_index = [1]
	# we need reverse the code in 
	max_scale = 5
	questionnaire_size = 3
	# reverse code for question 1
	for tuple_ in user_task_list:
		# "username", "pt1", "pt2", "pt3" = tuple_
		prospensity_to_trust = questionnaire(tuple_[1:], reverse_code_index, max_scale=max_scale, questionnaire_size=questionnaire_size, value_add_one=True).calc_value()
		user_id = tuple_[0]
		user_ptt_scale[user_id] = prospensity_to_trust
	return user_ptt_scale

def calc_trust_in_automation(filename, reserved_users=None):
	filename = os.path.join(data_folder, filename)
	usecols = ["username", "tia1_1", "tia1_2", "tia2_1", "tia2_2"]
	df = pd.read_csv(filename, usecols=usecols)
	if reserved_users is not None:
		# filter some invalid users
		df = df.drop(df[~df['username'].isin(reserved_users)].index)
	user_task_list = df.values.tolist()
	user_trust_first = {}
	user_trust_second = {}
	reverse_code_index = []
	max_scale = 5
	questionnaire_size = 2
	for tuple_ in user_task_list:
		# "username", "tia1_1", "tia1_2", "tia2_1", "tia2_2" = tuple_
		trust_1 = questionnaire(tuple_[1:3], reverse_code_index, max_scale=max_scale, questionnaire_size=questionnaire_size, value_add_one=True).calc_value()
		trust_2 = questionnaire(tuple_[3:], reverse_code_index, max_scale=max_scale, questionnaire_size=questionnaire_size, value_add_one=True).calc_value()
		user_id = tuple_[0]
		user_trust_first[user_id] = trust_1
		user_trust_second[user_id] = trust_2
	return user_trust_first, user_trust_second

def calc_miscalibration(filename, reserved_users=None):
	filename = os.path.join(data_folder, filename)
	usecols = ["username", "surveySelf1", "surveySelf2", "surveyOther1", "surveyOther2", "surveyPercentage1", "surveyPercentage2"]
	df = pd.read_csv(filename, usecols=usecols)
	if reserved_users is not None:
		# filter some invalid users
		df = df.drop(df[~df['username'].isin(reserved_users)].index)
	user_task_list = df.values.tolist()
	self_assessment_first = {}
	self_assessment_second = {}
	other_assessment_first = {}
	other_assessment_second = {}
	survey_percetage_first = {}
	survey_percetage_second = {}

	for tuple_ in user_task_list:
		# "username", "tia1_1", "tia1_2", "tia2_1", "tia2_2" = tuple_
		user_id = tuple_[0]
		self_assessment_first[user_id] = tuple_[1]
		self_assessment_second[user_id] = tuple_[2]
		other_assessment_first[user_id] = tuple_[3]
		other_assessment_second[user_id] = tuple_[4]
		survey_percetage_first[user_id] = tuple_[5]
		survey_percetage_second[user_id] = tuple_[6]
	return self_assessment_first, self_assessment_second, other_assessment_first, other_assessment_second, survey_percetage_first, survey_percetage_second

def read_attention_checks(filename, reserved_users=None):
	filename = os.path.join(data_folder, filename)
	usecols = ["username", "attention_ati", "attention6", "attention11", "attention18"]
	df = pd.read_csv(filename, usecols=usecols)
	if reserved_users is not None:
		# filter some invalid users
		df = df.drop(df[~df['username'].isin(reserved_users)].index)
	user_task_list = df.values.tolist()
	attention_check_answer_dict = {
		"attention_ati": 3,
		"attention6": "B",
		"attention11": "D",
		"attention18": "C"
	}
	user_attention_check_correct = {}
	for tuple_ in user_task_list:
		user_id = tuple_[0]
		if user_id not in user_attention_check_correct:
			user_attention_check_correct[user_id] = 0
		for index, task_id in enumerate(usecols[1:]):
			if tuple_[index + 1] == attention_check_answer_dict[task_id]:
				user_attention_check_correct[user_id] += 1
			# else:
			# 	print(f"{user_id} failed attention check {task_id}")
	return user_attention_check_correct


def read_decisions(filename, reserved_users=None):
	filename = os.path.join(data_folder, filename)
	user_task_dict = {}

	usecols = ["username", "question0", "question1", "question2", "question3", "question4", "question5", "question7", "question8", "question9", "question10", 
	"question12", "question13", "question14", "question15", "question16", "question17"]
	df = pd.read_csv(filename, usecols=usecols)
	if reserved_users is not None:
		# filter some invalid users
		df = df.drop(df[~df['username'].isin(reserved_users)].index)
	user_task_list = df.values.tolist()
	answer_type = "base"
	for tuple_ in user_task_list:
		user_id = tuple_[0]
		for index, task_id in enumerate(usecols[1:]):
			if user_id not in user_task_dict:
				user_task_dict[user_id] = {}
			task_id_ = int(task_id.replace("question", ""))
			user_task_dict[user_id][(task_id_, answer_type)] = tuple_[index+1]

	usecols = ["username", "advice0", "advice1", "advice2", "advice3", "advice4", "advice5", "advice7", "advice8", "advice9", "advice10",
	 "advice12", "advice13", "advice14", "advice15", "advice16", "advice17"]
	df = pd.read_csv(filename, usecols=usecols)
	if reserved_users is not None:
		# filter some invalid users
		df = df.drop(df[~df['username'].isin(reserved_users)].index)
	user_task_list = df.values.tolist()
	answer_type = "advice"
	for tuple_ in user_task_list:
		user_id = tuple_[0]
		for index, task_id in enumerate(usecols[1:]):
			if user_id not in user_task_dict:
				user_task_dict[user_id] = {}
			task_id_ = int(task_id.replace("advice", ""))
			user_task_dict[user_id][(task_id_, answer_type)] = tuple_[index+1]

	return user_task_dict

def find_complete_users(filename):
	filename = os.path.join(data_folder, filename)
	df = pd.read_csv(filename)
	df = df.dropna(axis=0)
	# drop all rows with any missing values
	return set(df["username"])

def find_valid_users(filename, threshold=4):
	complete_users = find_complete_users(filename)
	# print(f"In total, we have {len(complete_users)} participants with complete records")
	user_attention_check_correct = read_attention_checks(filename, reserved_users=complete_users)
	valid_users = set()
	approved_users = set()
	for user in complete_users:
		# if len(user) != 24:
		# 	# non-prolific ID
		# 	continue
		if user_attention_check_correct[user] >= threshold:
			valid_users.add(user)
		else:
			print(f"user {user} only passed {user_attention_check_correct[user]} attention checks")
		approved_users.add(user)
	print("-" * 17)
	return valid_users, approved_users

def get_user_question_order(filename, reserved_users=None):
	filename = os.path.join(data_folder, filename)
	user_task_dict = {}

	usecols = ["username", "question_order"]
	df = pd.read_csv(filename, usecols=usecols)
	if reserved_users is not None:
		# filter some invalid users
		df = df.drop(df[~df['username'].isin(reserved_users)].index)
	user_data_list = df.values.tolist()
	user_question_order = {}
	for username, question_order in user_data_list:
		user_question_order[username] = question_order_dict[question_order]
	return user_question_order

def check_user_condition(user, user_condition_dict):
	conditions = ["no tutorial, no xai", "with tutorial, no xai", "no tutorial, with xai", "with tutorial, with xai"]
	for condition in conditions:
		if user in user_condition_dict[condition]:
			return condition
	raise NotImplementedError(f"User {user} is not found in all conditions, please check")

def get_user_conditions(filename, reserved_users=None):
	filename = os.path.join(data_folder, filename)
	user_task_dict = {}

	usecols = ["username", "tutorial", "XAI"]
	df = pd.read_csv(filename, usecols=usecols)
	if reserved_users is not None:
		# filter some invalid users
		df = df.drop(df[~df['username'].isin(reserved_users)].index)
	user_data_list = df.values.tolist()
	user_condition_dict = {}
	user_condition_dict["no tutorial, no xai"] = set()
	user_condition_dict["with tutorial, no xai"] = set()
	user_condition_dict["no tutorial, with xai"] = set()
	user_condition_dict["with tutorial, with xai"] = set()
	for username, tutorial, xai in user_data_list:
		if xai == 0:
			if tutorial == 1:
				condition = "no tutorial, no xai"
			else:
				condition = "with tutorial, no xai"
		else:
			if tutorial == 1:
				condition = "no tutorial, with xai"
			else:
				condition = "with tutorial, with xai"
		user_condition_dict[condition].add(username)
	for condition in user_condition_dict:
		print(f"For {condition} condition, {len(user_condition_dict[condition])} users are valid for further analysis")
	return user_condition_dict

def calc_user_reliance_measures(user, usertask_dict, answer_dict, task_id_list):
	user_trust_list = []
	tp_correct = 0
	tp_trust = 0
	tp_agreement = 0
	initial_disagreement = 0
	tp_switch_reliance = 0
	tp_correct_initial_disagreement = 0
	positive_ai_reliance = 0
	negative_ai_reliance = 0
	positive_self_reliance = 0
	negative_self_reliance = 0
	for task_id in task_id_list:
		first_choice = usertask_dict[user][(task_id, "base")]
		second_choice = usertask_dict[user][(task_id, "advice")]
		system_advice = answer_dict[task_id][1]
		correct_answer = answer_dict[task_id][0]
		if second_choice == system_advice:
			# agreement fraction
			tp_agreement += 1
		if first_choice != system_advice:
			# initial disagreement
			initial_disagreement += 1
			if system_advice == correct_answer:
				if second_choice == system_advice:
					# user switch to ai advice, which is correct
					positive_ai_reliance += 1
				else:
					# user don't rely on AI systems when it's correct
					negative_self_reliance += 1
			else:
				if first_choice == correct_answer:
					if second_choice == correct_answer:
						# AI system provide wrong advice, but user insist their own correct decision
						positive_self_reliance += 1
					else:
						# After wrong AI advice, users changed the decision to wrong term
						negative_ai_reliance += 1
			if second_choice == system_advice:
				tp_switch_reliance += 1
			if second_choice == correct_answer:
				tp_correct_initial_disagreement += 1
		if second_choice == correct_answer:
			tp_correct += 1
	number_of_tasks = float(len(task_id_list))
	tp_accuracy = tp_correct / number_of_tasks
	tp_agreement_fraction = tp_agreement / number_of_tasks
	if positive_ai_reliance + negative_self_reliance > 0:
		relative_positive_ai_reliance = positive_ai_reliance / float(positive_ai_reliance + negative_self_reliance)
	else:
		relative_positive_ai_reliance = 0.0

	if positive_self_reliance + negative_ai_reliance > 0:
		relative_positive_self_reliance = positive_self_reliance / float(positive_self_reliance + negative_ai_reliance)
	else:
		relative_positive_self_reliance = 0.0

	if initial_disagreement > 0:
		tp_switching_fraction= float(tp_switch_reliance) / initial_disagreement
		tp_appropriate_reliance = float(tp_correct_initial_disagreement) / initial_disagreement
	else:
		tp_switching_fraction = 0.0
		tp_appropriate_reliance = 0.0
	return tp_correct, tp_agreement_fraction, tp_switching_fraction, tp_appropriate_reliance, initial_disagreement, relative_positive_ai_reliance, relative_positive_self_reliance

def analysis_user_reliance_measures(user, usertask_dict, answer_dict, task_id_list):
	user_trust_list = []
	tp_correct = 0
	tp_trust = 0
	tp_agreement = 0
	initial_disagreement = 0
	tp_switch_reliance = 0
	tp_correct_initial_disagreement = 0
	positive_ai_reliance = 0
	negative_ai_reliance = 0
	positive_self_reliance = 0
	negative_self_reliance = 0
	switch_under_initial_disagreement = 0
	insist_under_initial_disagreement = 0
	for task_id in task_id_list:
		first_choice = usertask_dict[user][(task_id, "base")]
		second_choice = usertask_dict[user][(task_id, "advice")]
		system_advice = answer_dict[task_id][1]
		correct_answer = answer_dict[task_id][0]
		if second_choice == system_advice:
			# agreement fraction
			tp_agreement += 1
		if first_choice != system_advice:
			# initial disagreement
			initial_disagreement += 1
			if second_choice == system_advice:
				switch_under_initial_disagreement += 1
			elif second_choice == first_choice:
				insist_under_initial_disagreement += 1
			if system_advice == correct_answer:
				if second_choice == system_advice:
					# user switch to ai advice, which is correct
					positive_ai_reliance += 1
				else:
					# user don't rely on AI systems when it's correct
					negative_self_reliance += 1
			else:
				if first_choice == correct_answer:
					if second_choice == correct_answer:
						# AI system provide wrong advice, but user insist their own correct decision
						positive_self_reliance += 1
					else:
						# After wrong AI advice, users changed the decision to wrong term
						negative_ai_reliance += 1
			if second_choice == system_advice:
				tp_switch_reliance += 1
			if second_choice == correct_answer:
				tp_correct_initial_disagreement += 1
		if second_choice == correct_answer:
			tp_correct += 1
	number_of_tasks = float(len(task_id_list))
	tp_accuracy = tp_correct / number_of_tasks
	tp_agreement_fraction = tp_agreement / number_of_tasks
	if positive_ai_reliance + negative_self_reliance > 0:
		relative_positive_ai_reliance = positive_ai_reliance / float(positive_ai_reliance + negative_self_reliance)
	else:
		relative_positive_ai_reliance = 0.0

	if positive_self_reliance + negative_ai_reliance > 0:
		relative_positive_self_reliance = positive_self_reliance / float(positive_self_reliance + negative_ai_reliance)
	else:
		relative_positive_self_reliance = 0.0

	if initial_disagreement > 0:
		tp_switching_fraction= float(tp_switch_reliance) / initial_disagreement
		tp_appropriate_reliance = float(tp_correct_initial_disagreement) / initial_disagreement
	else:
		tp_switching_fraction = 0.0
		tp_appropriate_reliance = 0.0
	four_patterns = [positive_ai_reliance, negative_self_reliance, positive_self_reliance, negative_ai_reliance]
	AR = [relative_positive_ai_reliance, relative_positive_self_reliance]
	disagreement_patterns = [initial_disagreement, switch_under_initial_disagreement, insist_under_initial_disagreement]
	return tp_correct, tp_agreement, tp_switching_fraction, disagreement_patterns, four_patterns, AR


class UserPerformance(object):
	
	def __init__(self, username, question_order):
		self.username = username
		self.question_order = question_order
		self.performance = {
			"overall": {},
			"first_group": {},
			"second_group": {}
		}
		self.miscalibration = {
			"first_group": {},
			"second_group": {}
		}
		self.keys = ["accuracy", "agreement_fraction", "switching_fraction", "appropriate_reliance", "relative_positive_ai_reliance", "relative_positive_self_reliance"]

	def add_performance(self, accuracy, agreement_fraction, switching_fraction, appropriate_reliance, relative_positive_ai_reliance, relative_positive_self_reliance, group="first_group"):
		self.performance[group] = {
			"accuracy": accuracy,
			"agreement_fraction": agreement_fraction,
			"switching_fraction": switching_fraction,
			"appropriate_reliance": appropriate_reliance,
			"relative_positive_ai_reliance": relative_positive_ai_reliance,
			"relative_positive_self_reliance": relative_positive_self_reliance
		}

	def add_miscalibration(self, self_assessment, actual_correct_number, group="first_group"):
		self.miscalibration[group] = self_assessment - actual_correct_number
		# use the gap as  miscalibration, range [-6, 6]

	def print_information(self):
		print("-" * 17)
		print(f"User {self.username}")
		for group in ["first_group", "second_group"]:
			print(group)
			for key_ in self.keys:
				print(key_, f"{self.performance[group][key_]}")
			print("miscalibration", self.miscalibration[group])
		print("-" * 17)

def calculate_bonus(user_question_order, usertask_dict, answer_dict):
	bounus_list = []
	total_bonus = 0.0
	for user in user_question_order:
		tp_order = user_question_order[user]
		first_group = tp_order[:6]
		second_group = tp_order[-6:]
		tp_performance = UserPerformance(username=user, question_order=tp_order)

		tp_correct, tp_agreement_fraction, tp_switching_fraction, tp_appropriate_reliance, initial_disagreement_1, relative_positive_ai_reliance, relative_positive_self_reliance = calc_user_reliance_measures(user, usertask_dict, answer_dict, first_group + second_group)
		tp_accuracy = tp_correct / 12.0
		bounus_list.append((user, tp_correct * 0.1))
		total_bonus += tp_correct * 0.1
	print("In total, we have {} GBP bonus for participants".format(total_bonus))
	return bounus_list

if __name__ == "__main__":
	# filename = "DKE_tut_xai_batch4.csv"
	filename = "all_valid_data.csv"
	valid_users, approved_users = find_valid_users(filename, 4)
	user_condition_dict = get_user_conditions(filename, valid_users)
	answer_dict = load_answers()
	user_question_order = get_user_question_order(filename, valid_users)
	usertask_dict = read_decisions(filename, valid_users)
	# self_assessment_first, self_assessment_second, other_assessment_first, other_assessment_second, survey_percetage_first, survey_percetage_second = calc_miscalibration(filename, valid_users)
	# get_user_performance(user_question_order, usertask_dict, answer_dict, self_assessment_first, self_assessment_second)

	# filename = "DKE_no_tut_xai_batch4.csv"
	# _, approved_users_old = find_valid_users(filename, 4)
	# # user_condition_dict = get_user_conditions(filename, valid_users)
	# # tp_valid_set = user_condition_dict["no tutorial, with xai"]

	# f = open("approved_users_sep_12.csv", "w")
	# for user in approved_users:
	# 	if user in approved_users_old:
	# 		continue
	# 	# if user not in tp_valid_set:
	# 	# 	continue
	# 	f.write("%s\n"%(user))
	# f.close()

	# f = open("bonus_with_tutorial_with_xai.csv", "w")
	# bounus_list = calculate_bonus(user_question_order, usertask_dict, answer_dict)
	# for user, bonus in bounus_list:
	# 	if user in approved_users_old:
	# 		continue
	# 	# if user not in tp_valid_set:
	# 	# 	continue
	# 	f.write("%s,%.1f\n"%(user, bonus))
	# f.close()
