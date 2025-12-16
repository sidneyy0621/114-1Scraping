# -*- coding: utf-8 -*-
"""
題目一: 探索性分析 customer_data.csv 檔案的特徵(視覺化呈現)
並處理資料的格式與問題，最後生成新的、整理好的CSV檔案
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# 設定中文字型以正確顯示中文
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 設定圖表樣式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """載入和初步探索資料"""
    print("="*50)
    print("載入客戶資料...")
    print("="*50)
    
    # 取得腳本所在目錄，確保路徑正確
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'Customer Data', 'customer_data.csv')
    
    # 載入資料 - 使用不同編碼嘗試
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except:
        try:
            df = pd.read_csv(file_path, encoding='big5')
        except:
            df = pd.read_csv(file_path, encoding='cp950')
    
    # 移除欄位名稱的前後空白
    df.columns = df.columns.str.strip()
    
    print(f"資料形狀: {df.shape}")
    print(f"欄位數量: {df.shape[1]}")
    print(f"資料筆數: {df.shape[0]}")
    
    return df

def basic_data_info(df):
    """基本資料資訊分析"""
    print("\n" + "="*50)
    print("基本資料資訊")
    print("="*50)
    
    print("\n欄位名稱:")
    for i, col in enumerate(df.columns):
        print(f"{i+1}. {col}")
    
    print(f"\n資料類型:")
    print(df.dtypes)
    
    print(f"\n資料概覽:")
    print(df.head())
    
    print(f"\n統計摘要:")
    print(df.describe(include='all'))

def analyze_missing_values(df):
    """遺失值分析"""
    print("\n" + "="*50)
    print("遺失值分析")
    print("="*50)
    
    # 計算遺失值
    missing_count = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100
    
    missing_df = pd.DataFrame({
        '遺失數量': missing_count,
        '遺失百分比': missing_percent
    })
    
    # 只顯示有遺失值的欄位
    missing_with_values = missing_df[missing_df['遺失數量'] > 0]
    
    if len(missing_with_values) > 0:
        print("\n有遺失值的欄位:")
        print(missing_with_values.sort_values('遺失百分比', ascending=False))
        
        # 視覺化遺失值
        plt.figure(figsize=(12, 6))
        missing_with_values['遺失百分比'].plot(kind='bar')
        plt.title('各欄位遺失值百分比')
        plt.xlabel('欄位')
        plt.ylabel('遺失百分比 (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plots_dir = os.path.join(script_dir, 'plots')
        plt.savefig(os.path.join(plots_dir, 'missing_values_analysis.png'), dpi=300, bbox_inches='tight')
        # plt.show()
        
    else:
        print("沒有遺失值！")
    
    return missing_df

def analyze_categorical_features(df):
    """類別型特徵分析"""
    print("\n" + "="*50)
    print("類別型特徵分析")
    print("="*50)
    
    # 識別類別型欄位
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    print(f"類別型欄位 ({len(categorical_cols)}個):")
    for col in categorical_cols:
        unique_count = df[col].nunique()
        print(f"- {col}: {unique_count} 個不同值")
        if unique_count <= 10:
            print(f"  值: {list(df[col].unique())}")
        else:
            print(f"  前5個值: {list(df[col].unique()[:5])}")
    
    # 視覺化所有類別型特徵
    # 分批繪圖，每頁 9 張圖 (3x3)
    chunk_size = 9
    categorical_cols_list = list(categorical_cols)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, 'plots')
    
    for i in range(0, len(categorical_cols_list), chunk_size):
        chunk = categorical_cols_list[i:i + chunk_size]
        
        rows = 3
        cols = 3
        fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
        axes = axes.flatten()
        
        for j, col in enumerate(chunk):
            if col in df.columns:
                value_counts = df[col].value_counts().nlargest(10) # 只顯示前10個類別
                sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[j])
                axes[j].set_title(f'{col} 分布')
                axes[j].tick_params(axis='x', rotation=45)
                axes[j].set_ylabel('計數')
        
        # 隱藏多餘的子圖
        for j in range(len(chunk), rows * cols):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'categorical_features_part_{i//chunk_size + 1}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        # plt.show()

def analyze_numerical_features(df):
    """數值型特徵分析"""
    print("\n" + "="*50)
    print("數值型特徵分析")
    print("="*50)
    
    # 識別數值型欄位
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    print(f"數值型欄位 ({len(numerical_cols)}個):")
    for col in numerical_cols:
        print(f"- {col}")
        print(f"  範圍: {df[col].min():.2f} ~ {df[col].max():.2f}")
        print(f"  平均: {df[col].mean():.2f}")
        print(f"  標準差: {df[col].std():.2f}")
        print()
    
    # 數值型特徵分布視覺化
    if len(numerical_cols) > 0:
        # 相關係數矩陣
        plt.figure(figsize=(12, 10))
        corr_matrix = df[numerical_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('數值型特徵相關係數矩陣')
        plt.tight_layout()
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plots_dir = os.path.join(script_dir, 'plots')
        plt.savefig(os.path.join(plots_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
        # plt.show()
        
        # 所有數值欄位的分布
        # 分批繪圖，每頁 12 張圖 (3x4)
        chunk_size = 12
        numerical_cols_list = list(numerical_cols)
        
        for i in range(0, len(numerical_cols_list), chunk_size):
            chunk = numerical_cols_list[i:i + chunk_size]
            
            rows = 3
            cols = 4
            fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
            axes = axes.flatten()
            
            for j, col in enumerate(chunk):
                if col in df.columns:
                    df[col].hist(bins=30, ax=axes[j], alpha=0.7)
                    axes[j].set_title(f'{col} 分布')
                    axes[j].set_xlabel(col)
                    axes[j].set_ylabel('頻率')
                    axes[j].grid(True, alpha=0.3)
            
            # 隱藏多餘的子圖
            for j in range(len(chunk), rows * cols):
                axes[j].axis('off')
                
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'numerical_features_distribution_part_{i//chunk_size + 1}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            # plt.show()

def analyze_customer_status(df):
    """客戶狀態分析"""
    print("\n" + "="*50)
    print("客戶狀態分析")
    print("="*50)
    
    status_col = None
    for col in df.columns:
        if '客戶狀態' in col or '狀態' in col or 'Status' in col:
            status_col = col
            break
    
    if status_col:
        print(f"客戶狀態欄位: {status_col}")
        status_counts = df[status_col].value_counts()
        print("\n客戶狀態分布:")
        print(status_counts)
        
        # 客戶狀態分布圓餅圖
        plt.figure(figsize=(10, 8))
        plt.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', startangle=90)
        plt.title('客戶狀態分布')
        plt.axis('equal')
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plots_dir = os.path.join(script_dir, 'plots')
        plt.savefig(os.path.join(plots_dir, 'customer_status_distribution.png'), dpi=300, bbox_inches='tight')
        # plt.show()
        
        # 分析不同狀態下的特徵
        if '年齡' in df.columns:
            plt.figure(figsize=(12, 6))
            for status in df[status_col].unique():
                if pd.notna(status):
                    subset = df[df[status_col] == status]['年齡']
                    plt.hist(subset, alpha=0.6, label=status, bins=20)
            plt.xlabel('年齡')
            plt.ylabel('頻率')
            plt.title('不同客戶狀態的年齡分布')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(plots_dir, 'age_by_customer_status.png'), dpi=300, bbox_inches='tight')
            # plt.show()

def detect_data_quality_issues(df):
    """檢測資料品質問題"""
    print("\n" + "="*50)
    print("資料品質檢測")
    print("="*50)
    
    issues = []
    
    # 檢測重複資料
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"發現 {duplicates} 筆重複資料")
    
    # 檢測異常值
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        if outliers > 0:
            issues.append(f"{col}: {outliers} 個異常值")
    
    # 檢測不一致的資料格式
    for col in df.select_dtypes(include=['object']).columns:
        unique_values = df[col].unique()
        # 檢測是否有空字串或空白字元
        empty_like = sum(1 for x in unique_values if pd.isna(x) or str(x).strip() == '')
        if empty_like > 0:
            issues.append(f"{col}: 有空值或空字串")
    
    if issues:
        print("發現的問題:")
        for issue in issues:
            print(f"- {issue}")
    else:
        print("沒有發現明顯的資料品質問題")
    
    return issues

def clean_data(df):
    """資料清理 - 使用合理的填補邏輯"""
    print("\n" + "="*50)
    print("開始資料清理")
    print("="*50)
    
    df_clean = df.copy()
    
    print(f"原始資料形狀: {df_clean.shape}")
    
    # 移除重複資料
    duplicates_before = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates()
    duplicates_after = duplicates_before - df_clean.duplicated().sum()
    if duplicates_after > 0:
        print(f"移除了 {duplicates_after} 筆重複資料")
    
    # --- 合理的遺失值填補邏輯 ---
    
    # 1. 處理客戶流失相關欄位 (Churn related)
    # 如果客戶狀態不是 'Churned'，流失類別和原因填補為 'Not Applicable'
    if '客戶狀態' in df_clean.columns:
        no_churn_mask = df_clean['客戶狀態'] != 'Churned'
        
        if '客戶流失類別' in df_clean.columns:
            print("處理 '客戶流失類別': 非流失客戶填補 'Not Applicable'")
            df_clean.loc[no_churn_mask, '客戶流失類別'] = df_clean.loc[no_churn_mask, '客戶流失類別'].fillna('Not Applicable')
            # 對於真正流失但資料缺失的，填補 'Unknown'
            df_clean['客戶流失類別'] = df_clean['客戶流失類別'].fillna('Unknown')
            
        if '客戶離開原因' in df_clean.columns:
            print("處理 '客戶離開原因': 非流失客戶填補 'Not Applicable'")
            df_clean.loc[no_churn_mask, '客戶離開原因'] = df_clean.loc[no_churn_mask, '客戶離開原因'].fillna('Not Applicable')
            # 對於真正流失但資料缺失的，填補 'Unknown'
            df_clean['客戶離開原因'] = df_clean['客戶離開原因'].fillna('Unknown')

    # 2. 處理電話服務相關欄位 (Phone related)
    if '電話服務' in df_clean.columns:
        no_phone_mask = df_clean['電話服務'] == 'No'
        
        if '平均長途話費' in df_clean.columns:
            print("處理 '平均長途話費': 無電話服務填補 0")
            df_clean.loc[no_phone_mask, '平均長途話費'] = df_clean.loc[no_phone_mask, '平均長途話費'].fillna(0)
            
        if '多線路服務' in df_clean.columns:
            print("處理 '多線路服務': 無電話服務填補 'No'")
            df_clean.loc[no_phone_mask, '多線路服務'] = df_clean.loc[no_phone_mask, '多線路服務'].fillna('No')

    # 3. 處理網路服務相關欄位 (Internet related)
    if '網路服務' in df_clean.columns:
        no_internet_mask = df_clean['網路服務'] == 'No'
        internet_cols = ['網路連線類型', '線上安全服務', '線上備份服務', '設備保護計劃', 
                         '技術支援計劃', '電視節目', '電影節目', '音樂節目', '無限資料下載']
        
        print("處理網路相關服務: 無網路服務填補 'No'")
        for col in internet_cols:
            if col in df_clean.columns:
                df_clean.loc[no_internet_mask, col] = df_clean.loc[no_internet_mask, col].fillna('No')
                
        if '平均下載量( GB)' in df_clean.columns:
            print("處理 '平均下載量( GB)': 無網路服務填補 0")
            df_clean.loc[no_internet_mask, '平均下載量( GB)'] = df_clean.loc[no_internet_mask, '平均下載量( GB)'].fillna(0)

    # 4. 優惠方式 (Offer)
    if '優惠方式' in df_clean.columns:
        print("處理 '優惠方式': 遺失值填補 'None'")
        df_clean['優惠方式'] = df_clean['優惠方式'].fillna('None')

    # 處理負數的每月費用
    if '每月費用' in df_clean.columns and '總費用' in df_clean.columns and '加入期間 (月)' in df_clean.columns:
        negative_mask = df_clean['每月費用'] < 0
        negative_count = negative_mask.sum()
        if negative_count > 0:
            print(f"處理 '每月費用': 發現 {negative_count} 筆負數，嘗試使用 '總費用' / '加入期間 (月)' 修正")
            
            # 計算推估的月費
            # 避免除以零，雖然加入期間看起來最小是1
            calculated_monthly = df_clean.loc[negative_mask, '總費用'] / df_clean.loc[negative_mask, '加入期間 (月)']
            
            # 填補
            df_clean.loc[negative_mask, '每月費用'] = calculated_monthly
            
            # 如果仍有無法計算的（例如加入期間為0），用中位數填補
            remaining_negative = df_clean['每月費用'] < 0
            if remaining_negative.sum() > 0:
                median_val = df_clean.loc[df_clean['每月費用'] > 0, '每月費用'].median()
                df_clean.loc[remaining_negative, '每月費用'] = median_val
                print(f"  - 仍有 {remaining_negative.sum()} 筆無法計算，使用中位數 {median_val:.2f} 填補")

    # 5. 其他剩餘的遺失值處理
    print("處理剩餘遺失值...")
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype in ['object']:
                # 類別型變數用眾數填補
                mode_val = df_clean[col].mode()
                if len(mode_val) > 0:
                    df_clean[col].fillna(mode_val[0], inplace=True)
                    print(f"{col}: 用眾數 '{mode_val[0]}' 填補剩餘遺失值")
            else:
                # 數值型變數用中位數填補
                median_val = df_clean[col].median()
                df_clean[col].fillna(median_val, inplace=True)
                print(f"{col}: 用中位數 {median_val:.2f} 填補剩餘遺失值")
    
    # 清理字串欄位（移除前後空白）
    string_cols = df_clean.select_dtypes(include=['object']).columns
    for col in string_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()
        # 將 'nan' 字串轉回 NaN (雖然理論上不應該有了)
        df_clean[col] = df_clean[col].replace('nan', np.nan)
    
    # 資料類型優化
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            try:
                # 嘗試轉換為數值型
                pd.to_numeric(df_clean[col])
                df_clean[col] = pd.to_numeric(df_clean[col], errors='ignore')
            except:
                pass
    
    print(f"清理後資料形狀: {df_clean.shape}")
    print("資料清理完成！")
    
    return df_clean

def generate_summary_report(df_original, df_clean):
    """生成摘要報告"""
    print("\n" + "="*50)
    print("資料分析摘要報告")
    print("="*50)
    
    print(f"原始資料筆數: {df_original.shape[0]}")
    print(f"清理後資料筆數: {df_clean.shape[0]}")
    print(f"移除資料筆數: {df_original.shape[0] - df_clean.shape[0]}")
    
    print(f"\n欄位總數: {df_clean.shape[1]}")
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    print(f"數值型欄位: {len(numerical_cols)}")
    print(f"類別型欄位: {len(categorical_cols)}")
    
    # 遺失值摘要
    total_missing_before = df_original.isnull().sum().sum()
    total_missing_after = df_clean.isnull().sum().sum()
    print(f"\n處理前總遺失值: {total_missing_before}")
    print(f"處理後總遺失值: {total_missing_after}")

def main():
    """主函數"""
    # 確保 plots 目錄存在
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print("客戶資料探索性分析開始...")
    
    # 載入資料
    df = load_and_explore_data()
    
    # 基本資料資訊
    basic_data_info(df)
    
    # 遺失值分析
    missing_analysis = analyze_missing_values(df)
    
    # 類別型特徵分析
    analyze_categorical_features(df)
    
    # 數值型特徵分析
    analyze_numerical_features(df)
    
    # 客戶狀態分析
    analyze_customer_status(df)
    
    # 資料品質檢測
    quality_issues = detect_data_quality_issues(df)
    
    # 資料清理
    df_clean = clean_data(df)
    
    # 儲存清理後的資料
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'Customer Data', 'new_customer_data.csv')
    df_clean.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n清理後的資料已儲存至: {output_path}")
    
    # 生成摘要報告
    generate_summary_report(df, df_clean)
    
    print("\n" + "="*50)
    print("分析完成！")
    print("="*50)

if __name__ == "__main__":
    main()
