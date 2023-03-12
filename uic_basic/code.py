
import pandas as pd
import sys
import sklearn.impute
from pandas import CategoricalDtype

from sklearn.impute import SimpleImputer

from cleverminer import cleverminer

df = pd.read_csv ('w:\\accidents.txt ', encoding='cp1250', sep='\t')
df=df[['Driver_Age_Band','Driver_IMD','Sex','Journey','Casualties','Severity','Area','Vehicle_Age','Road_Type','Speed_limit','Light','Vehicle_Location','Vehicle_Type']]

imputer = SimpleImputer(strategy="most_frequent")
df = pd.DataFrame(imputer.fit_transform(df),columns = df.columns)


Vehicle_Age_cat = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16-20','>20']
Vehicle_Age_cat_type =CategoricalDtype(categories=Vehicle_Age_cat, ordered=True)
df['Vehicle_Age'] = df['Vehicle_Age'].astype('category').cat.reorder_categories(Vehicle_Age_cat,ordered=True) # ( Vehicle_Age_cat,ordered=True)

Driver_IMD_cat = [1.0, 2.0,3.0,4.0, 5.0, 6.0,7.0,8.0,9.0,10.0]
Driver_IMD_cat_type =CategoricalDtype(categories=Driver_IMD_cat, ordered=True)
df['Driver_IMD'] = df['Driver_IMD'].astype('category').cat.reorder_categories(Driver_IMD_cat,ordered=True)

Speed_limit_cat = [10, 15, 20, 30, 40, 50, 60, 70]
Speed_limit_cat_type =CategoricalDtype(categories=Speed_limit_cat, ordered=True)
df['Speed_limit'] = df['Speed_limit'].astype('category').cat.reorder_categories(Speed_limit_cat,ordered=True)



clm = cleverminer(df=df,target='Severity',proc='UICMiner',
               quantifiers= {'aad_score':20,'aad_weights':[5,1,0],'base':200, 'relevant_base_lift':0.8},
               ante ={
                    'attributes':[
                        {'name': 'Driver_Age_Band', 'type': 'seq', 'minlen': 1, 'maxlen': 3},
                        {'name': 'Driver_IMD', 'type': 'seq', 'minlen': 1, 'maxlen': 3},
                        {'name': 'Sex', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'Area', 'type': 'subset', 'minlen': 1, 'maxlen': 2},
                        {'name': 'Journey', 'type': 'subset', 'minlen': 1, 'maxlen': 2},
                        {'name': 'Road_Type', 'type': 'subset', 'minlen': 1, 'maxlen': 2},
                        {'name': 'Speed_limit', 'type': 'seq', 'minlen': 1, 'maxlen': 2},
                        {'name': 'Light', 'type': 'subset', 'minlen': 1, 'maxlen': 2},
                        {'name': 'Vehicle_Location', 'type': 'subset', 'minlen': 1, 'maxlen': 2},
                        {'name': 'Vehicle_Type', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'Vehicle_Age', 'type': 'seq', 'minlen': 1, 'maxlen': 11}
                    ], 'minlen':1, 'maxlen':2, 'type':'con'}
                  , opts = {'no_automatic_data_conversions':True}
                  )

#26 rules


clm.print_data_definition()
clm.print_rulelist()
clm.print_summary()
clm.print_rule(1)
clm.print_rule(2)
clm.print_rule(17)
