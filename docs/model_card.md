# Model Details

This RandomForestClassifier model is trained to predict if an individual's income exceeds $50K/year, leveraging scikit-learn. It aims to analyze factors affecting income levels.

## Intended Use

The model is intended to be used for predicting income levels based on demographic and employment-related features. It is created as part of a udacity nano-degree project and is not intended for real-world use.

## Training Data

Census dataset with features including age, workclass, education, and others is used as independent variables. The target variable is salary (<=50K, >50K).

## Evaluation Data

The model is evaluated on a 20% split test dataset, ensuring unbiased assessment on unseen data.

## Metrics

Model performance evaluated on precision, recall, and F-beta score for overall and feature-specific slices:

### Overall

Precision: 0.7186
Recall: 0.6344
F-beta: 0.6739

### Feature-specific Slices

workclass=Private (n=4506, N=6513)- Precision: 0.7145, Recall: 0.6275, F-beta: 0.6681
workclass=Local-gov (n=424, N=6513)- Precision: 0.7629, Recall: 0.6491, F-beta: 0.7014
workclass=Self-emp-not-inc (n=517, N=6513)- Precision: 0.7059, Recall: 0.5000, F-beta: 0.5854
workclass=State-gov (n=269, N=6513)- Precision: 0.7432, Recall: 0.7534, F-beta: 0.7483
workclass=? (n=364, N=6513)- Precision: 0.5484, Recall: 0.4474, F-beta: 0.4928
workclass=Federal-gov (n=209, N=6513)- Precision: 0.7368, Recall: 0.6747, F-beta: 0.7044
workclass=Self-emp-inc (n=216, N=6513)- Precision: 0.7869, Recall: 0.8000, F-beta: 0.7934
workclass=Never-worked (n=4, N=6513)- Precision: 1.0000, Recall: 1.0000, F-beta: 1.0000
workclass=Without-pay (n=4, N=6513)- Precision: 1.0000, Recall: 1.0000, F-beta: 1.0000
education=Prof-school (n=119, N=6513)- Precision: 0.8947, Recall: 0.9239, F-beta: 0.9091
education=Bachelors (n=1037, N=6513)- Precision: 0.6953, Recall: 0.7531, F-beta: 0.7230
education=Some-college (n=1456, N=6513)- Precision: 0.6037, Recall: 0.4729, F-beta: 0.5304
education=9th (n=89, N=6513)- Precision: 1.0000, Recall: 0.0000, F-beta: 0.0000
education=Masters (n=361, N=6513)- Precision: 0.8812, Recall: 0.8900, F-beta: 0.8856
education=HS-grad (n=2111, N=6513)- Precision: 0.6146, Recall: 0.4051, F-beta: 0.4884
education=Doctorate (n=95, N=6513)- Precision: 0.8919, Recall: 0.8462, F-beta: 0.8684
education=Assoc-voc (n=259, N=6513)- Precision: 0.6346, Recall: 0.4925, F-beta: 0.5546
education=Preschool (n=11, N=6513)- Precision: 0.0000, Recall: 1.0000, F-beta: 0.0000
education=12th (n=90, N=6513)- Precision: 1.0000, Recall: 0.5000, F-beta: 0.6667
education=7th-8th (n=145, N=6513)- Precision: 1.0000, Recall: 0.2857, F-beta: 0.4444
education=Assoc-acdm (n=212, N=6513)- Precision: 0.8333, Recall: 0.6364, F-beta: 0.7216
education=11th (n=240, N=6513)- Precision: 0.5000, Recall: 0.1667, F-beta: 0.2500
education=5th-6th (n=68, N=6513)- Precision: 0.5000, Recall: 0.2000, F-beta: 0.2857
education=10th (n=187, N=6513)- Precision: 0.6667, Recall: 0.2667, F-beta: 0.3810
education=1st-4th (n=33, N=6513)- Precision: 1.0000, Recall: 0.0000, F-beta: 0.0000
marital-status=Married-civ-spouse (n=2978, N=6513)- Precision: 0.7115, Recall: 0.6732, F-beta: 0.6919
marital-status=Separated (n=225, N=6513)- Precision: 1.0000, Recall: 0.3529, F-beta: 0.5217
marital-status=Never-married (n=2125, N=6513)- Precision: 0.9744, Recall: 0.4000, F-beta: 0.5672
marital-status=Divorced (n=886, N=6513)- Precision: 0.7556, Recall: 0.4474, F-beta: 0.5620
marital-status=Widowed (n=217, N=6513)- Precision: 1.0000, Recall: 0.1905, F-beta: 0.3200
marital-status=Married-spouse-absent (n=79, N=6513)- Precision: 0.7500, Recall: 0.6000, F-beta: 0.6667
marital-status=Married-AF-spouse (n=3, N=6513)- Precision: 1.0000, Recall: 1.0000, F-beta: 1.0000
occupation=Exec-managerial (n=850, N=6513)- Precision: 0.7653, Recall: 0.7985, F-beta: 0.7815
occupation=Prof-specialty (n=818, N=6513)- Precision: 0.8231, Recall: 0.8016, F-beta: 0.8122
occupation=Protective-serv (n=107, N=6513)- Precision: 0.7200, Recall: 0.5455, F-beta: 0.6207
occupation=Other-service (n=642, N=6513)- Precision: 0.5000, Recall: 0.0800, F-beta: 0.1379
occupation=Adm-clerical (n=782, N=6513)- Precision: 0.5529, Recall: 0.5000, F-beta: 0.5251
occupation=Handlers-cleaners (n=279, N=6513)- Precision: 0.6000, Recall: 0.1429, F-beta: 0.2308
occupation=Transport-moving (n=317, N=6513)- Precision: 0.6154, Recall: 0.3019, F-beta: 0.4051
occupation=Craft-repair (n=804, N=6513)- Precision: 0.5950, Recall: 0.4162, F-beta: 0.4898
occupation=? (n=368, N=6513)- Precision: 0.5484, Recall: 0.4474, F-beta: 0.4928
occupation=Machine-op-inspct (n=402, N=6513)- Precision: 0.5625, Recall: 0.3103, F-beta: 0.4000
occupation=Sales (n=714, N=6513)- Precision: 0.6914, Recall: 0.6154, F-beta: 0.6512
occupation=Armed-Forces (n=2, N=6513)- Precision: 1.0000, Recall: 1.0000, F-beta: 1.0000
occupation=Tech-support (n=190, N=6513)- Precision: 0.6613, Recall: 0.7069, F-beta: 0.6833
occupation=Farming-fishing (n=213, N=6513)- Precision: 0.6364, Recall: 0.3043, F-beta: 0.4118
occupation=Priv-house-serv (n=25, N=6513)- Precision: 1.0000, Recall: 1.0000, F-beta: 1.0000
relationship=Husband (n=2622, N=6513)- Precision: 0.7193, Recall: 0.6726, F-beta: 0.6952
relationship=Unmarried (n=715, N=6513)- Precision: 0.8235, Recall: 0.2917, F-beta: 0.4308
relationship=Wife (n=300, N=6513)- Precision: 0.6496, Recall: 0.7063, F-beta: 0.6768
relationship=Not-in-family (n=1681, N=6513)- Precision: 0.8734, Recall: 0.4481, F-beta: 0.5923
relationship=Own-child (n=998, N=6513)- Precision: 1.0000, Recall: 0.2500, F-beta: 0.4000
relationship=Other-relative (n=197, N=6513)- Precision: 0.0000, Recall: 0.0000, F-beta: 0.0000
race=White (n=5575, N=6513)- Precision: 0.7280, Recall: 0.6455, F-beta: 0.6843
race=Black (n=632, N=6513)- Precision: 0.7037, Recall: 0.4691, F-beta: 0.5630
race=Asian-Pac-Islander (n=200, N=6513)- Precision: 0.6607, Recall: 0.6607, F-beta: 0.6607
race=Amer-Indian-Eskimo (n=59, N=6513)- Precision: 0.4000, Recall: 0.2857, F-beta: 0.3333
race=Other (n=47, N=6513)- Precision: 0.6667, Recall: 0.5000, F-beta: 0.5714
sex=Male (n=4372, N=6513)- Precision: 0.7277, Recall: 0.6413, F-beta: 0.6818
sex=Female (n=2141, N=6513)- Precision: 0.6886, Recall: 0.5897, F-beta: 0.6354
native-country=United-States (n=5825, N=6513)- Precision: 0.7283, Recall: 0.6396, F-beta: 0.6811
native-country=Philippines (n=32, N=6513)- Precision: 0.6000, Recall: 0.8571, F-beta: 0.7059
native-country=Mexico (n=121, N=6513)- Precision: 0.6667, Recall: 0.4000, F-beta: 0.5000
native-country=South (n=18, N=6513)- Precision: 0.4286, Recall: 0.5000, F-beta: 0.4615
native-country=? (n=124, N=6513)- Precision: 0.7812, Recall: 0.6250, F-beta: 0.6944
native-country=India (n=22, N=6513)- Precision: 0.6667, Recall: 0.7500, F-beta: 0.7059
native-country=Nicaragua (n=10, N=6513)- Precision: 1.0000, Recall: 0.0000, F-beta: 0.0000
native-country=Puerto-Rico (n=28, N=6513)- Precision: 1.0000, Recall: 0.6667, F-beta: 0.8000
native-country=Guatemala (n=16, N=6513)- Precision: 1.0000, Recall: 0.0000, F-beta: 0.0000
native-country=Dominican-Republic (n=11, N=6513)- Precision: 1.0000, Recall: 1.0000, F-beta: 1.0000
native-country=Taiwan (n=14, N=6513)- Precision: 0.8333, Recall: 0.8333, F-beta: 0.8333
native-country=Columbia (n=17, N=6513)- Precision: 1.0000, Recall: 0.0000, F-beta: 0.0000
native-country=Germany (n=32, N=6513)- Precision: 0.3333, Recall: 0.2500, F-beta: 0.2857
native-country=Vietnam (n=11, N=6513)- Precision: 0.0000, Recall: 0.0000, F-beta: 0.0000
native-country=Iran (n=7, N=6513)- Precision: 0.3333, Recall: 1.0000, F-beta: 0.5000
native-country=El-Salvador (n=22, N=6513)- Precision: 1.0000, Recall: 1.0000, F-beta: 1.0000
native-country=Jamaica (n=18, N=6513)- Precision: 0.0000, Recall: 1.0000, F-beta: 0.0000
native-country=Trinadad&Tobago (n=6, N=6513)- Precision: 1.0000, Recall: 1.0000, F-beta: 1.0000
native-country=England (n=18, N=6513)- Precision: 0.7143, Recall: 0.7143, F-beta: 0.7143
native-country=Outlying-US(Guam-USVI-etc) (n=4, N=6513)- Precision: 1.0000, Recall: 1.0000, F-beta: 1.0000
native-country=Canada (n=19, N=6513)- Precision: 0.8333, Recall: 0.6250, F-beta: 0.7143
native-country=Hungary (n=5, N=6513)- Precision: 1.0000, Recall: 1.0000, F-beta: 1.0000
native-country=Italy (n=16, N=6513)- Precision: 1.0000, Recall: 0.2500, F-beta: 0.4000
native-country=Scotland (n=3, N=6513)- Precision: 1.0000, Recall: 1.0000, F-beta: 1.0000
native-country=Poland (n=9, N=6513)- Precision: 0.0000, Recall: 0.0000, F-beta: 0.0000
native-country=Portugal (n=8, N=6513)- Precision: 1.0000, Recall: 0.0000, F-beta: 0.0000
native-country=Ireland (n=5, N=6513)- Precision: 1.0000, Recall: 1.0000, F-beta: 1.0000
native-country=Ecuador (n=6, N=6513)- Precision: 0.0000, Recall: 1.0000, F-beta: 0.0000
native-country=China (n=16, N=6513)- Precision: 1.0000, Recall: 1.0000, F-beta: 1.0000
native-country=Thailand (n=5, N=6513)- Precision: 1.0000, Recall: 0.0000, F-beta: 0.0000
native-country=Yugoslavia (n=4, N=6513)- Precision: 0.0000, Recall: 1.0000, F-beta: 0.0000
native-country=Hong (n=2, N=6513)- Precision: 1.0000, Recall: 1.0000, F-beta: 1.0000
native-country=Haiti (n=10, N=6513)- Precision: 1.0000, Recall: 1.0000, F-beta: 1.0000
native-country=Japan (n=11, N=6513)- Precision: 0.3333, Recall: 1.0000, F-beta: 0.5000
native-country=Cuba (n=18, N=6513)- Precision: 1.0000, Recall: 0.0000, F-beta: 0.0000
native-country=Greece (n=5, N=6513)- Precision: 0.5000, Recall: 0.6667, F-beta: 0.5714
native-country=Holand-Netherlands (n=1, N=6513)- Precision: 1.0000, Recall: 1.0000, F-beta: 1.0000
native-country=France (n=2, N=6513)- Precision: 1.0000, Recall: 1.0000, F-beta: 1.0000
native-country=Laos (n=4, N=6513)- Precision: 1.0000, Recall: 0.0000, F-beta: 0.0000
native-country=Peru (n=4, N=6513)- Precision: 1.0000, Recall: 1.0000, F-beta: 1.0000
native-country=Cambodia (n=3, N=6513)- Precision: 1.0000, Recall: 0.0000, F-beta: 0.0000
native-country=Honduras (n=1, N=6513)- Precision: 1.0000, Recall: 1.0000, F-beta: 1.0000

## Ethical Considerations

The inclusion of sensitive features such as race and sex necessitates careful consideration to avoid perpetuating biases. Users must critically assess and mitigate potential biases.

## Caveats and Recommendations

Utilize this model as part of a broader decision-making framework, supplemented by expert judgment.
