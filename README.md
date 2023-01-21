# CHI2023_DKE (Dunning-Kruger Effect in Human-AI Collaboration)
This is our code and data for the paper in CHI 2023

> Gaole He, Lucie Kuiper, Ujwal Gadiraju (2023). Knowing About Knowing: An Illusion of Human Competence Can Hinder Appropriate Reliance on AI Systems. In CHI'2023.

## Introduction
The dazzling promises of AI systems to augment humans in various tasks hinge on whether humans can appropriately rely on them. Recent research has shown that appropriate reliance is the key to achieving complementary team performance in AI-assisted decision making. This paper addresses an under-explored problem of whether the Dunning-Kruger Effect (DKE) among people can hinder their appropriate reliance on AI systems. DKE is a metacognitive bias due to which less-competent individuals overestimate their own skill and performance. Through an empirical study (𝑁 = 249), we explored the impact of DKE on human reliance on an AI system, and whether such effects can be mitigated using a tutorial intervention that reveals the fallibility of AI advice, and exploiting logic units-based explanations to improve user understanding of AI advice. We found that participants who overestimate their performance tend to exhibit under-reliance on AI systems, which hinders optimal team performance. We found that logic units-based explanations did not help users in either improving the calibration of their competence or facilitating appropriate reliance. While the tutorial intervention was highly effective in helping users calibrate their self-assessment and facilitating appropriate reliance among participants with overestimated self-assessment, we found that it can potentially hurt the appropriate reliance of participants with underestimated self-assessment. Our work has broad implications on the design of methods to tackle user cognitive biases while facilitating appropriate reliance on AI systems. Our findings advance the current understanding of the role of self-assessment in shaping trust and reliance in human-AI decision making. This lays out promising future directions for relevant HCI research in this community.

## Requirements:

- Python 3.8
- numpy
- pandas
- scipy
- pingouin

## Data analysis
Our main experimental results can be reimplemented with code in folder data_analysis. All experimental results can be obtained with executing these python files with `python xx.py`
* Section 5.1   : analysis_time_new.py, descriptive_statistics_new.py, explanation_usefulness.py, normal_distribution_analysis.py
* Section 5.2.1 : analysis_H1_new.py
* Section 5.2.2 : analysis_H2_new.py
* Section 5.2.3 : analysis_H3_new.py
* Section 5.2.4 : analysis_H4_new.py, analysis_H4_new_anova.py
* Section 5.3   : analysis_DKE_new.py
* Section 5.3   : analysis_trust_new.py

## Data processing
Specifically, our data is preprocessed with functions in util.py in data_analysis folder.
We have several types of keys to record different information:
* user ID: username
* experimental condition: tutorial, XAI
* question order: question_order
* user initial decision : questionx (x in 0-17)
* user decision after advice: advicex (x in 0-17)
* attention check: attention_ati, attention6, attention11, attention18
* Questionnaire : surveySelf1, surveySelf2, surveyOther1, surveyOther2, surveyPercentage1, surveyPercentage2


## Web interface
Our interface is implemented with Flask, you can refer to [this repo](https://github.com/LucieKuiper/Thesis) for more details.

## Acknowledgement
Any scientific publications that use our codes and datasets should cite the following paper as the reference:
```
@inproceedings{He-CHI-2023,
    title = "Knowing About Knowing: An Illusion of Human Competence Can Hinder Appropriate Reliance on AI Systems",
    author = {Gaole He and
              Lucie Kuiper and
              Ujwal Gadiraju},
    booktitle={CHI Conference on Human Factors in Computing Systems},
    year = {2023},
}
```
Nobody guarantees the correctness of the data, its suitability for any particular purpose, or the validity of results based on the use of the data set. The data set may be used for any research purposes under the following conditions:
* The user must acknowledge the use of the data set in publications resulting from the use of the data set.
* The user may not redistribute the data without separate permission.
* The user may not try to deanonymise the data.
* The user may not use this information for any commercial or revenue-bearing purposes without first obtaining permission from us.