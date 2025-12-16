import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("./Customer Data/merged_customer_data.csv", encoding='big5')

# 選擇分析的城市
goal_region = 'Los Angeles'
region_df = df[df['城市'] == goal_region].copy()

print(f"\n{'='*60}")
print(f"分析城市: {goal_region}")
print(f"總客戶數: {len(region_df)}")
print(f"{'='*60}\n")

# (a) 營收分析
print("【營收分析】")
print("-" * 60)

# 1. 計算每位客戶的總營收
region_df['總營收'] = region_df['總收入']

# 2. 各服務類別營收分析
service_columns = ['電話服務', '多線路服務', '網際網路服務', '線上安全服務', 
                   '線上備份服務', '裝置保護方案', '技術支援方案']

service_revenue = {}
for service in service_columns:
    if service in region_df.columns:
        service_users = region_df[region_df[service].isin(['Yes', 'Cable', 'DSL', 'Fiber Optic'])]
        total_revenue = service_users['總營收'].sum()
        avg_revenue = service_users['總營收'].mean()
        count = len(service_users)
        service_revenue[service] = {
            '營收項目': service,
            '總營收': total_revenue,
            '平均營收': avg_revenue,
            '客戶數': count
        }

service_df = pd.DataFrame(service_revenue).T
service_df = service_df.sort_values('總營收', ascending=False).reset_index(drop=True)
print(service_df)

print(f"\n最高總營收服務: {service_df.iloc[0]['營收項目']} (${service_df.iloc[0]['總營收']:,.2f})")
print(f"最高平均營收服務: {service_df.loc[service_df['平均營收'].idxmax(), '營收項目']} (${service_df['平均營收'].max():,.2f})")
print(f"最多客戶使用服務: {service_df.loc[service_df['客戶數'].idxmax(), '營收項目']} ({int(service_df['客戶數'].max()):,}人)")

print("==" * 60)

# (b) 特徵分析
print("婚姻狀態與營收分析")
merry_group = region_df.groupby('婚姻', observed=True)
merry_revenue = {
    '婚姻狀態': merry_group['婚姻'].first(),
    '客戶數': merry_group.size(),
    '平均營收': merry_group['總營收'].mean(),
    '總營收': merry_group['總營收'].sum()
}
merry_df = pd.DataFrame(merry_revenue).reset_index(drop=True)
print(merry_df)

print("--" * 60)
print("年齡與營收分析")
group_of_age = pd.cut(region_df['年齡'], bins=[0, 19.5, 29.5, 39.5, 49.5, 59.5, 69.5, 79.5, 89.5, 99.5], 
                      labels=['0-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99'])
region_df['年齡區間'] = group_of_age
age_group = region_df.groupby('年齡區間', observed=True)
age_revenue = {
    '年齡區間': age_group['年齡區間'].first(),
    '客戶數': age_group.size(),
    '平均營收': age_group['總營收'].mean(),
    '總營收': age_group['總營收'].sum()
}
age_df = pd.DataFrame(age_revenue).reset_index(drop=True)
age_df = age_df.dropna().reset_index(drop=True)
print(age_df)


print(f"\n最高總營收年齡區間: {age_df.loc[age_df['總營收'].idxmax(), '年齡區間']} (${age_df['總營收'].max():,.2f})")
print(f"最高平均營收年齡區間: {age_df.loc[age_df['平均營收'].idxmax(), '年齡區間']} (${age_df['平均營收'].max():,.2f})")