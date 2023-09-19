
# %% import pakietów i załadowanie zbioru danych

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib.pyplot import figure

df_titanic = pd.read_csv('C:/Data Science/Dane/Pliki csv/Datasets Data Mining/Titanic.csv', sep=";")
df_titanic.head

# %% pd.dataframe do tabeli w LateX'u
df_table1 = df_titanic.iloc[0:10,0:7]
print(df_table1.to_latex(index=False,
                formatters={"name": str.upper},
                float_format="{:.1f}".format,
))  

df_table2 = df_titanic.iloc[0:10,7:14]
print(df_table2.to_latex(index=False,
                formatters={"name": str.upper},
                float_format="{:.1f}".format,
))  


# %% EDA
# %%% liczba unikalnych wartosci dla zmiennych
df_titanic['pclass'].nunique()
df_titanic['survived'].nunique()
df_titanic['name'].nunique()
df_titanic['sex'].nunique()
df_titanic['age'].nunique()
df_titanic['sibsp'].nunique()
df_titanic['parch'].nunique()
df_titanic['ticket'].nunique()
df_titanic['fare'].nunique()
df_titanic['cabin'].nunique()
df_titanic['embarked'].nunique()
df_titanic['boat'].nunique()
df_titanic['body'].nunique()
df_titanic['home.dest'].nunique()

df_titanic.min()
x = df_titanic['age'].dropna()
x = list(x)
x = [i.replace(",", ".") for i in x]
x = [eval(i) for i in x]
min(x)
max(x)

x = df_titanic['fare'].dropna()
x = list(x)
x = [i.replace(",", ".") for i in x]
x = [eval(i) for i in x]
min(x)
max(x)


# %%% wartosci brakujące
df_titanic.isnull().sum()

100*(round(df_titanic.isnull().sum()/len(df_titanic), 3))


# %%% wytypowanie i usuwanie nieotrzebnych zmiennych
df_titanic['initial']=0
df_titanic["initial"] = df_titanic["name"].apply(lambda st: st[st.find(",")+2:st.find(".")])
df_titanic['initial'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess',
                               'Jonkheer','Col','Rev','Capt','Sir','Don', 'Dona', 'the Countess']
                              ,['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other',
                                'Other','Mr','Mr','Mr', 'Mrs', 'Mrs'],inplace=True)

df_titanic = df_titanic.drop(["cabin", "boat", "body", "home.dest", "name", "ticket"], axis=1)

df_titanic = df_titanic.reindex(columns=["survived", "pclass", "initial","sex", "age", "sibsp", "parch", 
                                       "fare", "embarked"])

print(df_titanic.head().to_latex(index=False,
                formatters={"name": str.upper},
                float_format="{:.1f}".format))

# %%% Zmiana typu zmiennych
df_titanic['age'] = df_titanic['age'].dropna().str.replace(",", ".")
df_titanic['fare'] = df_titanic['fare'].dropna().str.replace(",", ".")
df_titanic[['age', 'fare']] = df_titanic[['age', 'fare']].apply(pd.to_numeric)

# %%% uzupełnienie wartosci brakujacych

#uzupełniamy brakujące wartoci fare srednimi wartosciami
df_titanic["embarked"] = df_titanic["embarked"].fillna(df_titanic['embarked'].mode())
#uzupełniamy brakujące wartoci fare srednimi wartosciami
df_titanic["fare"] = df_titanic["fare"].fillna(df_titanic['fare'].mean())
#uzupełniamy wartosci brakujace wg. srednich zmiennej age pogrupowanych po zmiennej initial
df_titanic['age'] = df_titanic['age'].fillna(df_titanic.groupby('initial')['age'].transform('mean'))

df_titanic = df_titanic.drop("initial", axis=1)

# %%% Wizualizacja danych

#wykresy słupkowe
fig, axis = plt.subplots(2,3, figsize=(6.8,4.5))
sn.countplot(x="survived",data=df_titanic, ax=axis[0,0])
sn.countplot(x="pclass",data=df_titanic, ax=axis[0,1])
sn.countplot(x="sex",data=df_titanic, ax=axis[0,2])
sn.countplot(x="sibsp",data=df_titanic, ax=axis[1,0])
sn.countplot(x="parch",data=df_titanic, ax=axis[1,1])
sn.countplot(x="embarked",data=df_titanic, ax=axis[1,2])

fig.tight_layout()
fig
fig.savefig('C:/Data Science/Python/Raporty/Titanic Ensemble/Barplots.jpg') 

#histogramy
fig, axis = plt.subplots(1,2, figsize = (6,2.5))
axis[0].hist(df_titanic['age'], color="green" , density=False, edgecolor='black' )
axis[0].set_title('age')
axis[1].hist(df_titanic['fare'],  density=False, edgecolor='black' )
axis[1].set_title('fare')
fig.tight_layout()
fig.savefig('C:/Data Science/Python/Raporty/Titanic Ensemble/Histograms.jpg') 
fig

#fare barplot
df_titanic['fare_przedzialy'] = pd.qcut(df_titanic['fare'], 4)
fig, axis = plt.subplots(figsize=(5,3.3))
axis = sn.barplot(x ='fare_przedzialy', y ='survived', data = df_titanic, errorbar=None)
axis.set_ylabel("survived %")
axis.set_xlabel("fare")
axis.set_xticklabels(axis.get_xticklabels(), fontsize=8)
#axis.set_xticklabels(axis.get_xticklabels(), rotation=40, ha='right')
fig.tight_layout()
fig.savefig('C:/Data Science/Python/Raporty/Titanic Ensemble/Fare_Barplot.jpg')
fig

#embarked barplot
fig, axis = plt.subplots(figsize=(5,3.3))
axis = sn.barplot(x ='embarked', y ='survived', data = df_titanic, errorbar=None)
axis.set_ylabel("survived %")
axis.set_xlabel("embarked")
fig.tight_layout()
fig.savefig('C:/Data Science/Python/Raporty/Titanic Ensemble/Embarked_Barplot.jpg')

#Age przedziały Barplot
df_titanic['age_przedzialy'] = pd.qcut(df_titanic['age'], 4)
fig, axis =  plt.subplots(figsize=(6,4))
axis = sn.barplot(x ='age_przedzialy', y ='survived', data = df_titanic, errorbar=None)
axis.set_ylabel("survived %")
axis.set_xlabel("age")
axis.set_yticks([0.0,0.1,0.2,0.3,0.4,0.5])
fig.tight_layout()
fig.savefig('C:/Data Science/Python/Raporty/Titanic Ensemble/Age_Barplot.jpg')
fig

#Age+Sex
x = df_titanic.groupby(['age_przedziały', 'sex'], as_index = False)['survived'].mean()
fig, axis = plt.subplots(figsize=(7,5.7))
axis = x.pivot("age_przedziały", "sex", "survived").plot(kind='bar', ax=axis)
axis.set_ylabel("survived %")
axis.set_xlabel("age")
axis.set_xticklabels(axis.get_xticklabels(), rotation=40, ha='right')
fig.tight_layout()
fig.savefig('C:/Data Science/Python/Raporty/Titanic Ensemble/Age&Sex_Barplots.jpg') 
fig

#violin plot pclass, sex
fig, axis=plt.subplots(1,1)
axis = sn.violinplot(data=df_titanic, x='pclass', y='age',hue='survived',split=True)
axis.set_yticks(range(0,100,10))
axis.set_axisbelow(True)
axis.grid(which="major",axis="y")
fig.tight_layout()
fig
fig.savefig('C:/Data Science/Python/Raporty/Titanic Ensemble/Age&Pclass_Violin.jpg')


fig, axis=plt.subplots(1,1)
axis = sn.violinplot(data=df_titanic, x='pclass', y='age',hue='survived',split=True)
axis.set_yticks(range(0,100,10))
axis.set_axisbelow(True)
axis.grid(which="major",axis="y")
fig.tight_layout()
fig

#tabelki sibsp, parch
surv_sibsp_tab = pd.crosstab(df_titanic["survived"], df_titanic['sibsp'])
surv_parch_tab = pd.crosstab(df_titanic["survived"], df_titanic['parch'])

print(surv_parch_tab.to_latex())
print(surv_sibsp_tab.to_latex())

#usunięcie kolumn ilosciowych podzielonych na przedzialy, przed eksportem 
df_titanic = df_titanic.drop(["age_przedzialy", "fare_przedzialy"], axis=1)

#eksport csv zmodyfikowanego zbioru danych
df_titanic.to_csv('C:/Data Science/Python/Raporty/Titanic Ensemble/df_titanic_R.csv', index=False)  

