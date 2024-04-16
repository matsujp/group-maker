import pandas as pd
import pulp
import itertools as it


df_member = pd.read_excel("src/data.xlsx", sheet_name="member")
df_group = pd.read_excel("src/data.xlsx", sheet_name="group")

member_name_list = df_member["name"].to_list()
member_hc_dict = dict(zip(df_member["name"], df_member["hc"]))
member_separate_dict = dict(zip(df_member["name"], df_member["separate"].fillna(0)))
member_fix_dict = dict(zip(df_member["name"], df_member["fix"].fillna(0)))

group_name_list = df_group["name"].to_list()
group_num_dict = dict(zip(df_group["name"], df_group["num"]))
group_priority_dict = dict(zip(df_group["name"], df_group["priority"]))


model = pulp.LpProblem(sense=pulp.LpMinimize)
x = pulp.LpVariable.dicts(
    name="x",
    indices=it.product(member_name_list, group_name_list),
    cat=pulp.LpBinary,
)
#
for member_name, group_name in it.product(member_name_list, group_name_list):
    if member_fix_dict[member_name] == group_name:
        model += x[member_name, group_name] == 1
#
for member_name in member_name_list:
    model += (
        pulp.lpSum(x[member_name, group_name] for group_name in group_name_list) == 1
    )
#
for group_name in group_name_list:
    model += (
        pulp.lpSum(x[member_name, group_name] for member_name in member_name_list)
        <= group_num_dict[group_name]
    )
#
y = pulp.LpVariable.dicts(
    name="y",
    indices=it.product(member_name_list, group_name_list),
    cat=pulp.LpContinuous,
)
group_ahc = {}
for group_name in group_name_list:
    group_ahc[group_name] = (
        pulp.lpSum(
            [x[name, group_name] * member_hc_dict[name] for name in member_name_list]
        )
        / group_num_dict[group_name]
    )

obj1 = pulp.lpSum(
    y[member_name, group_name]
    for member_name, group_name in it.product(member_name_list, group_name_list)
)
for member_name, group_name in it.product(member_name_list, group_name_list):
    model += (
        y[member_name, group_name]
        <= member_hc_dict[member_name] * x[member_name, group_name]
        - group_ahc[group_name]
    )
    model += y[member_name, group_name] <= -(
        member_hc_dict[member_name] * x[member_name, group_name] - group_ahc[group_name]
    )
#
z = pulp.LpVariable.dicts(
    name="z",
    indices=group_name_list,
    cat=pulp.LpContinuous,
)
ahc = pulp.lpSum(group_ahc[group_name] for group_name in group_name_list) / len(
    group_name_list
)
obj2 = pulp.lpSum(z[group_name] for group_name in group_name_list)
for group_name in group_name_list:
    model += (
        z[group_name]
        >= pulp.lpSum(
            member_hc_dict[member_name] * x[member_name, group_name]
            for member_name in member_name_list
        )
        - ahc
    )
    model += z[group_name] >= -(
        pulp.lpSum(
            member_hc_dict[member_name] * x[member_name, group_name]
            for member_name in member_name_list
        )
        - ahc
    )
#
# for name in member_name_list:
#     if member_priority_dict[name] == 1:
#         obj3 = -pulp.lpSum(
#             x[name, group_name]
#             for group_name in group_name_list
#             if group_priority_dict[group_name] == 1
#         )
#

h = pulp.LpVariable(
    name="h",
    cat=pulp.LpContinuous,
)
m = pulp.LpVariable(
    name="l",
    cat=pulp.LpContinuous,
)
sep_point = {}
for group_name in group_name_list:
    sep_point[group_name] = pulp.lpSum(
        x[name, group_name]
        for name in member_name_list
        if member_separate_dict[name] == 1
    )
obj4 = h - m
for group_name in group_name_list:
    model += sep_point[group_name] <= h
    model += sep_point[group_name] >= m

model += -obj1 + obj2 + 99 * obj4
# %%
solver = pulp.FSCIP_CMD(path="C:/Program Files/SCIPOptSuite 9.0.0/bin/fscip.exe")
status = model.solve(solver=solver)
print(pulp.LpStatus[status])
# %%
print(model.objective.value())
# %%
result = []
for member_name in member_name_list:
    for group_name in group_name_list:
        if round(x[member_name, group_name].value()) == 1:
            result.append(
                [
                    member_name,
                    member_hc_dict[member_name],
                    member_separate_dict[member_name],
                    group_name,
                ]
            )
            break

df_member = pd.DataFrame(data=result, columns=["name", "hc", "separate", "group"])
df_member.to_csv("result_member.csv", index=False)
print(df_member)
