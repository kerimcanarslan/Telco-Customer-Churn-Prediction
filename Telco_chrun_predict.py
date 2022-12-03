##################
# Kütüphaneler
##################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)

###########################
# FONKSİYONLAR
###########################

# 1-) Aykırı değerler için limit belirledik.
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quantile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quantile3 - quartile1
    up_limit = quantile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

# 2-) Kolonlar içinde bu limitleri aşan aykırı değerler var mı diye sormak.
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

# 3-) Sinsirellaları tespit etme.
def grab_col_name(dataframe, cat_th= 10, car_th=20):
    # 1-) Kategorik Değişkenleri seçelim
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    # Sayısal gibi görünen ama Kategorik olan değişkenleri seçelim
    num_but_cat =[col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                  dataframe[col].dtypes != "O"]
    # Kategorik gibi görünen ama Sayısal(Kardinal) olan değişkenleri seçelim
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    # Şimdi Kategorik Kolonları son haline getirelim
    cat_cols = cat_cols + num_but_cat   # ikisini birleştirelim
    cat_cols = [col for col in cat_cols if col not in cat_but_car]  # Kategorik olupta sayısal olanları  çıkaralım

    # 2-) Sayısal(numerik) Değişkenleri Seçelim
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]  # Sayısal olanalrı seçelim
    num_cols = [col for col in num_cols if col not in num_but_cat]  # Sayısal görünüp kat. olanları çıkaralım

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    # Biz genelde sayısal değişkenler üzerinde çalışacağımız için num_cols ve cat_but_car bize lazım olur.
    # Ama cat_cols'u da return edelim. num_but_cat'i zaten cat_cols'un içine attık. Ayrıca onu return etmeye gerek yok.
    # cat_cols + num_cols + cat_but_car toplamı toplam değişken sayısını verir.
    return cat_cols, num_cols, cat_but_car

# 4-) Aykırı değerlerin index bilgilerine ulaşmai
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:    # eğer aykırı değer sayısı 10'dan büyükse
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())    # Bu aykırı değerlere head at ve görmem için yazdır.
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])           # eğer aykırı değer sayısı 10'dan az ise hepsini yazdır

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        # eğer aykırı değerlerin index bilgisini istiyorsan ön tanımlı index=False bilgisini index=True yap.
        # Böylece aykırı değerlerin index bilgisini return edeceksin.
        return outlier_index

# 5-) Aykırı değerleri silmek gerekiyorsa.
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)

    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

# 6-) Aykırı değerleri baskılamamız gerekiyorsa.
def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit    # alt limitten daha aşağıda olanları alt limite eşitle.
    dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit      # üst limitten daha yukarda olanaları üst limite eşitle

# 7-) Eksik değerler ve oranlar tablosu
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    # eksik değer içeren değişkenleri tuttuk.
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    # içindeki eksik değer sayılarına göre büyükten küçüğe sıraladık ve n_mis adında yeni bir df oluşturduk.
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    # yüzdelik olarak oranlarını da ratio adında bir df'e attık.

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

# 8-) Eksik değerlerin ve eksik olmayanların hedef değişkene etkisi
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)   # eksik değer olanalara 1 olmayanlara 0 ver.
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("NA")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end= "\n\n")

# 9-) Binary Encoding(büyüklük anlamı taşır)
def label_encoding(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

# 10-) One-Hot Encoder (get-dummies)
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

# 11-) Kategorik değişken içinde kaç farklı sınıf var, sınıfların içinde kaç adet değer var?
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

# 12-) Rare analizi, hangi kolonlarda rare olabilecek sınıf var?
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

# 13-) Rare olabilecekleri rare encode etmek.
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

# 14-) Sayısal değişkenlerin kaç sınıfı var, sınıflarda kaç değer var? (bunu standartlaştırmadan sonra yapmak gerek)
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


df = pd.read_csv("TelcoChurn-221116-114752/Telco-Customer-Churn.csv")

df.columns = [col.upper() for col in df.columns]

df.head()


cat_cols, num_cols, cat_but_car = grab_col_name(df)

num_cols #['TENURE', 'MONTHLYCHARGES']
cat_but_car # ['CUSTOMERID', 'TOTALCHARGES']

# TotalCharges sayısal bir değişken olmalı
df["TOTALCHARGES"] = pd.to_numeric(df["TOTALCHARGES"], errors='coerce')

# Hedef değişken encode'u

df["CHURN"] = df["CHURN"].apply(lambda x: 1 if x == "Yes" else 0)

df["CHURN"] = df["CHURN"].astype("int64")

#  df["CHURN"]= df["CHURN"].map({"Yes":1,"No":0}) ALTERNATİF


##################################
# GÖREV 1: KEŞİFÇİ VERİ ANALİZİ
##################################

##################################
# GENEL RESİM
##################################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

cat_cols, num_cols, cat_but_car = grab_col_name(df)

num_cols # ['TENURE', 'MONTHLYCHARGES', 'TOTALCHARGES']


##################################
# KATEGORİK DEĞİŞKENLERİN ANALİZİ
##################################

for col in cat_cols:
    cat_summary(df, col)

##################################
# NUMERİK DEĞİŞKENLERİN ANALİZİ
##################################

for col in num_cols:
    num_summary(df, col, plot=True)

df[df["CONTRACT"] == "Month-to-month"]["TENURE"].hist(bins=20)
plt.xlabel("TENURE")
plt.title("Month-to-month")
plt.show()

df[df["CONTRACT"] == "Two year"]["TENURE"].hist(bins=20)
plt.xlabel("TENURE")
plt.title("Two year")
plt.show()


# MonthyChargers'a bakıldığında aylık sözleşmesi olan müşterilerin aylık ortalama ödemeleri daha fazla olabilir.

df[df["CONTRACT"] == "Month-to-month"]["MONTHLYCHARGES"].hist(bins=20)
plt.xlabel("MONTHLYCHARGES")
plt.title("Month-to-month")
plt.show()

df[df["CONTRACT"] == "Two year"]["MONTHLYCHARGES"].hist(bins=20)
plt.xlabel("MONTHLYCHARGES")
plt.title("Two year")
plt.show()


##################################
# NUMERİK DEĞİŞKENLERİN TARGETA GÖRE ANALİZİ
##################################

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "CHURN", col)


##################################
# KATEGORİK DEĞİŞKENLERİN TARGETA GÖRE ANALİZİ
##################################

def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "CHURN", col)

##################################
# KORELASYON
##################################

df[num_cols].corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

##################################
# GÖREV 2: FEATURE ENGINEERING
##################################

##################################
# EKSİK DEĞER ANALİZİ
##################################

df.isnull().sum()

na_columns = missing_values_table(df, na_name=True)


missing_vs_target(df, "CHURN", na_columns)

df["TOTALCHARGES"].fillna(0, inplace=True)
df[df["TOTALCHARGES"].isnull()]["TENURE"]
df.isnull().sum()

#df = df.dropna(subset=["TotalCharges"], axis=0)  # default axis=0 and default how= any

#df = df.dropna(subset=["TotalCharges"], axis=0)  # default axis=0 and default how= any

##################################
# BASE MODEL KURULUMU
##################################
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

dff = df.copy()
cat_cols = [col for col in cat_cols if col not in ["CHURN"]]
cat_cols



dff = one_hot_encoder(dff, cat_cols, drop_first=True)

y = dff["CHURN"]
X = dff.drop(["CHURN", "CUSTOMERID"], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_test,y_pred ), 4)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 4)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 4)}")
print(f"F1: {round(f1_score(y_pred, y_test), 4)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 4)}")

# Accuracy: 0.7922
# Recall: 0.6517
# Precision: 0.5052
# F1: 0.5692
# Auc: 0.7407


# Aykırı Değer Analizi ve Baskılama İşlemi
for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df[num_cols])

df_scores = clf.negative_outlier_factor_
df_scores[0:5]
# df_scores = -df_scores
np.sort(df_scores)[0:100]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()

th = np.sort(df_scores)[11]

df = df.drop(axis=0, labels=df[df_scores < th].index)

##################################
# ÖZELLİK ÇIKARIMI
##################################

# Tenure  değişkeninden yıllık kategorik değişken oluşturma
df.loc[(df["TENURE"] >= 0) & (df["TENURE"] <= 12), "NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["TENURE"] > 12) & (df["TENURE"] <= 24), "NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["TENURE"] > 24) & (df["TENURE"] <= 36), "NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["TENURE"] > 36) & (df["TENURE"] <= 48), "NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["TENURE"] > 48) & (df["TENURE"] <= 60), "NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["TENURE"] > 60) & (df["TENURE"] <= 72), "NEW_TENURE_YEAR"] = "5-6 Year"

df["NEW_TENURE_YEAR"].value_counts()

# Kontratı 1 veya 2 yıllık müşterileri Engaged olarak belirtme
df["NEW_Engaged"] = df["CONTRACT"].apply(lambda x: 1 if x in ["One year", "Two year"] else 0)

# En az bir tane online destekten yararlanan
df["NEW_noProt"] = df.apply(
    lambda x: 1 if (x["ONLINEBACKUP"] != "No") or (x["DEVICEPROTECTION"] != "No") or (x["TECHSUPPORT"] != "No") else 0,axis=1)

# Aylık sözleşmesi bulunan ve genç olan müşteriler
df["NEW_Young_Not_Engaged"] = df.apply(lambda x: 1 if (x["NEW_Engaged"] == 0) and (x["SENIORCITIZEN"] == 0) else 0,
                                       axis=1)

# Kişinin toplam aldığı servis sayısı
df['NEW_TotalServices'] = (df[['PHONESERVICE', 'INTERNETSERVICE', 'ONLINESECURITY',
                               'ONLINEBACKUP', 'DEVICEPROTECTION', 'TECHSUPPORT',
                               'STREAMINGTV', 'STREAMINGMOVIES']] == 'Yes').sum(axis=1)

# Herhangi bir streaming hizmeti alan kişiler
df["NEW_FLAG_ANY_STREAMING"] = df.apply(
    lambda x: 1 if (x["STREAMINGTV"] == "Yes") or (x["STREAMINGMOVIES"] == "Yes") else 0, axis=1)

# Kişi otomatik ödeme yapıyor mu?
df["NEW_FLAG_AutoPayment"] = df["PAYMENTMETHOD"].apply(
    lambda x: 1 if x in ["Bank transfer (automatic)", "Credit card (automatic)"] else 0)

# ortalama aylık ödeme
df["NEW_AVG_Charges"] = df["TOTALCHARGES"] / (df["TENURE"] + 0.1)

# Güncel Fiyatın ortalama fiyata göre artışı
df["NEW_Increase"] = df["NEW_AVG_Charges"] / (df["MONTHLYCHARGES"] + 1)

# Servis başına ücret
df["NEW_AVG_Service_Fee"] = df["MONTHLYCHARGES"] / (df['NEW_TotalServices'] + 1)



df.loc[(df['GENDER'] == 'Male') & (df['SENIORCITIZEN'] == 0), 'new_sex_cat'] = 'youngmale'
df.loc[(df['GENDER'] == 'Male') & (df['SENIORCITIZEN'] == 1), 'new_sex_cat'] = 'oldmale'
df.loc[(df['GENDER'] == 'Female') & (df['SENIORCITIZEN'] == 0), 'new_sex_cat'] = 'youngfemale'
df.loc[(df['GENDER'] == 'Female') & (df['SENIORCITIZEN'] == 1), 'new_sex_cat'] = 'oldfemale'



def num_value_dispersion(dataframe, col, target_col, cut_number = 4):       #hedef değişkene göre dağılım (sayısal değişkenler)
    print("-------------Tüm değerlere göre------------")
    print(dataframe.groupby(col).agg({target_col: ["mean", "count"]}))
    print("-------------Qcut ile sınıflara ayrılarak------------")
    temp_df = dataframe
    temp_df[f"{col}_cut"] = pd.qcut(temp_df[col], cut_number)
    print(temp_df.groupby(f"{col}_cut").agg({target_col: ["mean", "count"]}))

num_value_dispersion(df,"TOTALCHARGES","CHURN")


##################################
# ENCODING
##################################

# Değişkenlerin tiplerine göre ayrılması işlemi
cat_cols, num_cols, cat_but_car = grab_col_name(df)


rare_analyser(df, "CHURN", cat_cols)


# LABEL ENCODING

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoding(df, col)


# One-Hot Encoding İşlemi
# cat_cols listesinin güncelleme işlemi
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["CHURN", "NEW_TotalServices"]]
cat_cols

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()
df.shape

# SCALING

rs = RobustScaler()

for col in num_cols:
    df[col] = rs.fit_transform(df[[col]])

df.describe().T

##################################
# MODELLEME
##################################


y = df["CHURN"]
X = df.drop(["CHURN", "CUSTOMERID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 3)}")
print(f"Recall: {round(recall_score(y_pred, y_test), 2)}")
print(f"Precision: {round(precision_score(y_pred, y_test), 2)}")
print(f"F1: {round(f1_score(y_pred, y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred, y_test), 2)}")

# Accuracy: 0.777
# Recall: 0.59
# Precision: 0.5
# F1: 0.55
# Auc: 0.71

def plot_feature_importance(importance, names, model_type):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(15, 10))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show()

plot_feature_importance(catboost_model.get_feature_importance(), X.columns, 'CATBOOST')



## LOGISTIC REGRESSION

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show(block=True)
    return drop_list


drop_list = high_correlated_cols(df)

df.drop(drop_list, axis=1, inplace=True)

y = df["CHURN"]
X = df.drop(["CHURN", "CUSTOMERID"], axis=1)

LR = LogisticRegression().fit(X, y)

cvr = cross_validate(LR, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])

for k, v in cvr.items():
    print(f"{k} : {v.mean()}")

# fit_time : 0.05126333236694336
# score_time : 0.005346616109212239
# test_accuracy : 0.8054607508532423
# test_f1 : 0.5912745584284895
# test_roc_auc : 0.8466909728569738
# test_precision : 0.6689738820494
# test_recall : 0.52997028863209