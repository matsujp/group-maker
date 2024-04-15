# %%
import pandas as pd
import pulp
import itertools as it
import math

df = pd.read_csv(
    "src/data.csv",
)

name_list = df["name"].to_list()
score_dict = dict(zip(df["name"], df["score"]))
priority_dict = dict(zip(df["name"], df["priority"].fillna(0)))
separate_dict = dict(zip(df["name"], df["separate"].fillna(0)))
inout_list = ["in", "out"]
group_num = 8
one_side_group_num_list = range(math.ceil(group_num / 2))


model = pulp.LpProblem(sense=pulp.LpMinimize)
x = pulp.LpVariable.dicts(
    name="x",
    indices=it.product(name_list, inout_list, one_side_group_num_list),
    cat=pulp.LpBinary,
)

#
y = pulp.LpVariable.dicts(
    name="y",
    indices=it.combinations(it.product(inout_list, one_side_group_num_list), r=2),
    cat=pulp.LpContinuous,
)
group_score = {}
for inout, num in it.product(inout_list, one_side_group_num_list):
    group_score[inout, num] = pulp.lpSum(
        [x[name, inout, num] * score_dict[name] for name in name_list]
    )
obj1 = pulp.lpSum(
    y[a, b]
    for a, b in it.combinations(it.product(inout_list, one_side_group_num_list), r=2)
)
for a, b in it.combinations(it.product(inout_list, one_side_group_num_list), r=2):
    model += y[a, b] >= group_score[a] - group_score[b]
    model += y[a, b] >= -(group_score[a] - group_score[b])
#
z = pulp.LpVariable.dicts(
    name="z",
    indices=it.combinations(it.product(inout_list, one_side_group_num_list), r=2),
    cat=pulp.LpContinuous,
)
group_num = {}
for inout, num in it.product(inout_list, one_side_group_num_list):
    group_num[inout, num] = pulp.lpSum([x[name, inout, num] for name in name_list])
obj2 = pulp.lpSum(
    z[a, b]
    for a, b in it.combinations(it.product(inout_list, one_side_group_num_list), r=2)
)
for a, b in it.combinations(it.product(inout_list, one_side_group_num_list), r=2):
    model += z[a, b] >= group_score[a] - group_score[b]
    model += z[a, b] >= -(group_score[a] - group_score[b])
#
for name in name_list:
    if priority_dict[name] == 1:
        obj3 = pulp.lpSum(
            999 * x[name, inout, num]
            for inout, num in it.product(inout_list, one_side_group_num_list)
            if num != 0
        )
#
for inout, num in it.product(inout_list, one_side_group_num_list):
    obj4 = 999 * (
        pulp.lpSum(
            x[name, inout, num] for name in name_list if separate_dict[name] == 1
        )
        - 1
    )
#
for name in name_list:
    model += (
        pulp.lpSum(
            x[name, inout, num]
            for inout, num in it.product(inout_list, one_side_group_num_list)
        )
        == 1
    )

model += obj1 + obj2 + obj3 + obj4
# %%
solver = pulp.FSCIP_CMD(path="C:/Program Files/SCIPOptSuite 9.0.0/bin/fscip.exe")
status = model.solve(solver=solver)
print(pulp.LpStatus[status])
# %%
print(model.objective)
# %%
result = []
for name in name_list:
    for inout, num in it.product(inout_list, one_side_group_num_list):
        if x[name, inout, num] == 1:
            result.append([name, inout, num])
            break

df = pd.DataFrame(data=result, columns=["name", "in-out", "no"])
df.to_csv("result.csv")
print(df)
