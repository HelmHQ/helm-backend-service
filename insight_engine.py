import pandas as pd
import numpy as np
import json
from scipy import stats
from datetime import datetime

class InsightEngine:
    def __init__(self, raw_data_list):
        """
        raw_data_list: List of dicts (DailyMetric objects from Flutter)
        """
        self.df = pd.DataFrame(raw_data_list)
        
    def run_analysis(self):
        """Main pipeline execution."""
        if len(self.df) < 5:
            return {"error": "Not enough data. Need at least 5 days of history."}
            
        self._preprocess()
        self._feature_engineering()
        
        # Run different types of analysis
        correlations = self._analyze_numerical_correlations()
        group_diffs = self._analyze_categorical_differences()
        
        # Combine and sort by statistical significance
        all_insights = correlations + group_diffs
        all_insights.sort(key=lambda x: abs(x['strength']), reverse=True)
        
        # Return top 5 strongest insights
        return all_insights[:5]

    def _preprocess(self):
        # 1. Handle Dates
        self.df['date_obj'] = pd.to_datetime(self.df['date'])
        self.df['day_of_week'] = self.df['date_obj'].dt.day_name()
        self.df['is_weekend'] = self.df['date_obj'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
        
        # 2. Fill Missing Values
        # Numerical: Mean
        num_cols = ['totalScreenOnTime', 'appUsageFrequency', 'deviceUnlockCount', 
                   'notificationsReceived', 'dailyStepCount', 'sleepTime', 
                   'appSwitchingFrequency', 'entropyScore',
                   'sleepQuality', 'moodScore', 'stressScore', 
                   'productivityScore', 'energyLevel']
                   
        for col in num_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].mean())

        # Categorical/Binary: Mode (Most Frequent)
        cat_cols = ['fragmentation', 'exercise', 'dominantActivityTag']
        for col in cat_cols:
             if col in self.df.columns:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 0)

    def _feature_engineering(self):
        """Expands JSON columns into analyzable features."""
        
        # A. Screen Time Categories
        self._expand_json_column('screenTimeByCategory', prefix='cat')
        
        # B. App Details (Filter < 300s)
        self._expand_app_details()
        
        # C. Sentiment (Top 2)
        self._expand_sentiment()

    def _expand_json_column(self, col_name, prefix):
        """Parses simple JSON dicts like Screen Time Categories."""
        def parse_safe(x):
            if isinstance(x, str):
                try: return json.loads(x)
                except: return {}
            return {}
            
        expanded = self.df[col_name].apply(parse_safe).apply(pd.Series)
        expanded = expanded.fillna(0).add_prefix(prefix)
        self.df = pd.concat([self.df, expanded], axis=1)

    def _expand_app_details(self):
        """Parses nested app details and filters noise (< 5 mins)."""
        app_data = []
        
        for _, row in self.df.iterrows():
            daily_apps = {}
            raw = row.get('appDetailsByCategory', '{}')
            if isinstance(raw, str):
                try:
                    data = json.loads(raw)
                    # Iterate through categories (Social, Productivity...)
                    for cat, apps in data.items():
                        for app_name, duration in apps.items():
                            # CRITICAL: Ignore usage < 5 minutes (300s)
                            if duration >= 300:
                                daily_apps[f"app_{app_name}"] = duration
                except:
                    pass
            app_data.append(daily_apps)
            
        app_df = pd.DataFrame(app_data).fillna(0)
        self.df = pd.concat([self.df, app_df], axis=1)

    def _expand_sentiment(self):
        """Parses sentiment JSON and extracts top 2 emotions."""
        sentiments_list = []
        
        for _, row in self.df.iterrows():
            day_sentiments = {}
            raw = row.get('sentimentAnalysisRaw') # Using the correct new column name
            
            if isinstance(raw, str) and raw:
                try:
                    data = json.loads(raw)
                    labels = data.get('sentiments', [])
                    scores = data.get('scores', [])
                    
                    # Take top 2
                    for i in range(min(2, len(labels))):
                        label = labels[i]
                        score = scores[i]
                        day_sentiments[f"emotion_{label}"] = score
                except:
                    pass
            sentiments_list.append(day_sentiments)
            
        sent_df = pd.DataFrame(sentiments_list).fillna(0)
        self.df = pd.concat([self.df, sent_df], axis=1)

    # --- STATISTICAL TESTS ---

    def _analyze_numerical_correlations(self):
        """Pearson/Spearman Correlation for Metric vs Metric."""
        insights = []
        
        # Targets
        targets = ['sleepQuality', 'moodScore', 'stressScore', 'productivityScore', 'energyLevel']
        # Features (Auto-detected from expansion)
        feature_cols = [c for c in self.df.columns if c.startswith(('cat_', 'app_', 'emotion_', 'total', 'daily', 'entropy'))]
        
        for target in targets:
            if target not in self.df.columns: continue
            
            for feature in feature_cols:
                if feature not in self.df.columns: continue
                
                # Ensure numeric and drop NaNs for calculation
                valid_data = self.df[[feature, target]].dropna()
                if len(valid_data) < 3: continue # Need data points
                
                # Pearson Correlation
                corr, p_value = stats.pearsonr(valid_data[feature], valid_data[target])
                
                # Check Significance (p < 0.1 for 'trend', p < 0.05 for 'strong')
                # Check Strength (|r| > 0.4)
                if abs(corr) > 0.4 and p_value < 0.15: # Slightly lenient for academic demo
                    insights.append({
                        "type": "correlation",
                        "feature": feature,
                        "target": target,
                        "strength": corr, # -1 to 1
                        "p_value": p_value,
                        "significance": "high" if p_value < 0.05 else "moderate"
                    })
        return insights

    def _analyze_categorical_differences(self):
        """T-Test and ANOVA for Categorical Features."""
        insights = []
        targets = ['sleepQuality', 'moodScore', 'stressScore', 'productivityScore', 'energyLevel']
        
        # 1. Binary Features (T-Test)
        binary_cols = ['is_weekend', 'exercise', 'fragmentation']
        
        for target in targets:
            for col in binary_cols:
                if col not in self.df.columns: continue
                
                group0 = self.df[self.df[col] == 0][target]
                group1 = self.df[self.df[col] == 1][target]
                
                if len(group0) > 1 and len(group1) > 1:
                    t_stat, p_value = stats.ttest_ind(group0, group1)
                    
                    if p_value < 0.15: # Meaningful difference found
                        diff = group1.mean() - group0.mean()
                        insights.append({
                            "type": "group_diff",
                            "feature": col,
                            "target": target,
                            "strength": diff, # Positive means Group 1 is higher
                            "p_value": p_value
                        })

        # 2. Categorical Features (ANOVA) - e.g. Dominant Activity
        if 'dominantActivityTag' in self.df.columns:
            for target in targets:
                groups = [group[target].values for name, group in self.df.groupby('dominantActivityTag')]
                if len(groups) > 1:
                    f_stat, p_value = stats.f_oneway(*groups)
                    if p_value < 0.15:
                         # Find which group is the outlier
                         best_activity = self.df.groupby('dominantActivityTag')[target].mean().idxmax()
                         insights.append({
                            "type": "anova",
                            "feature": "dominantActivityTag",
                            "target": target,
                            "strength": 1.0, # Placeholder
                            "detail": best_activity, # e.g., "Sports"
                            "p_value": p_value
                        })
                        
        return insights