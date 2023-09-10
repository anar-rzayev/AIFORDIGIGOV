# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# dataset files
sample_file = "../data/queue_dataset_train_small_sample.csv"
train_file = "../data/queue_dataset_train.csv"
test_file = "../data/queue_dataset_test.csv"
baseline_res = "../data/baseline_submission.csv"


# read train & test samples
sample_data = pd.read_csv(sample_file)
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)


# utility functions
branches = ['Masallı ASAN', '1 saylı ASAN', '4 saylı ASAN', 'Qəbələ ASAN', '2 saylı ASAN', 'Sumqayıt ASAN']
service_org = ['Daxili İşlər Nazirliyi', 'Vergilər Nazirliyi',
               'Ədliyyə Nazirliyi', 'Funksional Yardımçı xidmətlər',
               'Azərbaycan Respublikasının Dövlət Sosial Müdafiə Fondu',
               'Əmlak Məsələləri Dövlət Komitəsi', 
               'Dövlət Miqrasiya Xidməti']
service_names = ['Ümumvətəndaş pasportlarının verilməsi və dəyişdirilməsi',
       'Kommersiya hüquqi şəxslərin və vergi ödəyicilərinin qeydiyyatı',
       'Etibarnamə', 'İmzanın təsdiqi', 'Digər xidmətlər',
       'Yaşayış yeri üzrə qeydiyyatda olan şəxslər haqqında arayışın verilməsi',
       'Azərcell', 'Vərəsəlik', 'Foto',
       'Əmək pensiyalarinin təyin edilməsi', 'Təkrar çıxarış - mənzil',
       'Asan İmza sertifikatının verilməsi',
       'Şəxsiyyət vəsiqələrinin verilməsi və dəyişdirilməsi',
       'Tibbi arayışların verilməsi (qan qrupu, boyun ölçülməsi, göz rəngi)',
       'Ölümün qeydə alınması, övladlığa götürmənin qeydə alınması, atalığın müəyyən edilməsinin qeydə alınması, nikahın pozulmasının qeydə alınması, adın, ata adının və soyadın dəyişdirilməsinin qeydə alınması, vətəndaşlıq vəziyyəti aktlarının dövlət qeydiyyatı haqqında şəhadətnamələrin (təkrar şəhadətnamələrin) verilməsi',
       'Surətin təsdiqi',
       'Əcnəbinin və ya vətəndaşlığı olmayan şəxsin olduğu yer üzrə qeydiyyata alınması',
       'İlkin qeydiyyat', 'Sürücülük vəsiqələrinin dəyişdirilməsi',
       'Məhkumluq barədə arayışların verilməsi', 'Miqrasiya xidməti',
       'İlkin çıxarış - mənzil', 'Tərcümə', 'Kapital Bank']


def encode_and_bind(original_dataframe, feature_to_encode):
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return(res) 


def restore_time(sample_data):
    # restore nan times based on age, gender and service type
    # first we need to change all ages to numerical values (avg of interval) to work 
    def process(q_time):
        q_time = list(map(lambda x: int(x[:2]), q_time.split(":")))
        return 60*q_time[0] + q_time[1]

    sample_copy = sample_data.copy()
    sample_copy['time_start_process'] = sample_copy['time_start_process'].map(lambda q_time : q_time if pd.isnull(q_time) else process(q_time))

    grouped_data = sample_copy.groupby(['customer_gender','service_name', 'customer_age_appl'])
    grouped_data_mean = grouped_data.mean()
    grouped_data_mean = grouped_data_mean.reset_index()[['customer_gender', 'service_name', 'customer_age_appl', 'time_start_process']]
    
    def missing_time(entry):
        condition = (
                (grouped_data_mean['customer_gender'] == entry['customer_gender']) & 
                (grouped_data_mean['service_name'] == entry['service_name']) &
                (grouped_data_mean['customer_age_appl'] == entry['customer_age_appl'])
                ) 
        q_time = int(grouped_data_mean[condition]['time_start_process'].values[0])
        return str(q_time // 60) + ":" + str(q_time % 60) + ":00"
        
    sample_data['time_start_process'] = sample_data.apply(lambda entry: missing_time(entry) if pd.isnull(entry['time_start_process']) else entry['time_start_process'],axis=1)
    return sample_data


def restore_age(sample_data):
    # restore time based on age, gender, service type
    sample_copy = sample_data.copy()
    sample_copy['customer_age_appl'] = sample_copy['customer_age_appl'].map(lambda age : age if pd.isnull(age) else list(map(int, age.split("-")))[0] + 2)

    grouped_data = sample_copy.groupby(['customer_gender','service_name'])
    grouped_data_mean = grouped_data.mean()
    grouped_data_mean = grouped_data_mean.reset_index()[['customer_gender', 'service_name', 'customer_age_appl']]
    
    def missing_age(entry):
        condition = (
                (grouped_data_mean['customer_gender'] == entry['customer_gender']) & 
                (grouped_data_mean['service_name'] == entry['service_name'])
                ) 
        age = int(grouped_data_mean[condition]['customer_age_appl'].values[0])
        rem = age % 5
        return str(age - rem + 1) + "-" + str(age - rem + 5)
        
    sample_data['customer_age_appl'] = sample_data.apply(lambda entry: missing_age(entry) if pd.isnull(entry['customer_age_appl']) else entry['customer_age_appl'],axis=1)
    return sample_data


def categorize_dates(sample_data):
    def weekdays(inp_date):
        inp_date = list(map(int, inp_date.split("-")))
        return date(inp_date[0], inp_date[1], inp_date[2]).weekday()

    sample_data['date'] = sample_data['date'].map(lambda inp_date : weekdays(inp_date))
    return sample_data
    

def categorize_ages(sample_data):
    def process(age):
        age = list(map(int, age.split("-")))
        if age[1] <= 15:
            return 0
        elif age[1] <= 30:
            return 1
        elif age[1] <= 55:
            return 2
        else:
            return 3
    
    sample_data['customer_age_appl'] = sample_data['customer_age_appl'].map(lambda age : process(age))
    return sample_data
    

def categorize_time(sample_data):
    def process(time):
        time = list(map(lambda x: int(x[:2]), time.split(":")))
        if time[1] < 12:
             return 0
        elif time[1] < 14:
            return 1
        else:
            return 2
    
    sample_data['time_start_process'] = sample_data['time_start_process'].map(lambda time : process(time))
    return sample_data
    
                                                               
def categorize_queue(sample_data):
    def process(queue):
        if queue<=5:
            return 0
        elif queue<=20:
            return 1
        else:
            return 2
    sample_data['previous_customer_count'] = sample_data['previous_customer_count'].map(lambda queue : process(queue))
    return sample_data


def remove_unused_cols(sample_data):
    # sample_data.drop(['customer_age_appl'], axis=1, inplace=True)
    sample_data.drop(['customer_gender'], axis=1, inplace=True)
    # sample_data.drop(['time_start_process'], axis=1, inplace=True)
    sample_data.drop(['operator_count'], axis=1, inplace=True)
    sample_data.drop(['date'], axis=1, inplace=True)
    sample_data.drop(['customer_city'], axis=1, inplace=True)
    sample_data.drop(['service_name_2'], axis=1, inplace=True)
    # sample_data.drop(['previous_customer_count'], axis=1, inplace=True)
    sample_data.drop(['service_name_organization'], axis=1, inplace=True)
    return sample_data
          
                                                 
print("Starting Preprocessing")
train_data = train_data.dropna()
# train_data = restore_age(train_data)
# train_data = restore_time(train_data)
# train_data = encode_and_bind(train_data, "service_name_organization")
train_data = encode_and_bind(train_data, "branch_name")
# train_data = encode_and_bind(train_data, "service_name")
train_data['service_name'].replace(service_names, list(range(len(service_names))), inplace=True)
train_data = categorize_time(train_data)
train_data = categorize_dates(train_data)
train_data = categorize_ages(train_data)
train_data = categorize_queue(train_data)
train_data = remove_unused_cols(train_data)

test_data = test_data.dropna()
# test_data = restore_age(test_data)
# test_data = restore_time(test_data)
# test_data = encode_and_bind(test_data, "service_name_organization")
test_data = encode_and_bind(test_data, "branch_name")
# test_data = encode_and_bind(test_data, "service_name")
test_data['service_name'].replace(service_names, list(range(len(service_names))), inplace=True)
test_data = categorize_time(test_data)
test_data = categorize_dates(test_data)
test_data = categorize_ages(test_data)
test_data = categorize_queue(test_data)
test_data = remove_unused_cols(test_data)

print("Finished Preprocessing\n")
print("-------------------------")
print("Starting Training")
# Creating data for training
X_train = train_data.drop(['service_canceled', "id"], axis=1)
Y_train = train_data["service_canceled"]
X_test = test_data.drop(["id"], axis=1)

# Select algorithms, Use the algorithm you want to use as classifier
classifier = RandomForestClassifier(n_estimators = 100, n_jobs=-1,
                                    verbose=1, random_state = 42)

# Check the accuracy, AUC, and ROC curve of the classifier set above
classifier.fit(X_train, Y_train)
accuracy = classifier.score(X_train, Y_train) * 100
Y_train_pred = classifier.predict_proba(X_train)[:, 1]

FPR, TPR, thresholds = roc_curve(Y_train, Y_train_pred)
AUC = roc_auc_score(Y_train, Y_train_pred)

# plt.plot(FPR, TPR)
print("Accuracy: ", "{0:.2f}".format(accuracy))
print("Area Under the Curve: ", "{0:.2f}".format(AUC))

print("\n-------------------------")
print("Starting Predictions")

# Print the test data prediction and generation of kaggle submission file
predict = classifier.predict(X_test)

# Create kaggle submission file
submission = pd.DataFrame({'id': test_data['id'], 'service_canceled': predict})
submission.to_csv('submission.csv', index=False)