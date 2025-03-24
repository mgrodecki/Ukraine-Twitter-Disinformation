import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori
from IPython.display import display


#store_data = pd.read_csv("store_data.csv", header=None)
store_data = pd.read_csv("Cleaned_Transactions_0819_UkraineCombinedTweetsDeduped.csv")
store_data.head()
display(store_data.head())
print(store_data.shape)

items = set()
for col in store_data:
    items.update(store_data[col].unique())
print(items)

records = []
for i in range(1, 50):
    records.append([str(store_data.values[i, j]) for j in range(0, 20)])

association_rules = apriori(records, min_support=0.03, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)


print("There are {} Relation derived.".format(len(association_results)))

for i in range(0, len(association_results)):
    print(association_results[i][0])

for item in association_results:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")