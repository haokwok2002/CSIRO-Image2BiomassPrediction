def add_hkf_folds(
        self,
        df: pd.DataFrame,
        n_folds: int = 5,
        stratify_col: str = 'State',
        group_col: str = 'Sampling_Date',
        seed: int = 42,
    ) -> pd.DataFrame:
        """
        Create folds with StratifiedGroupKFold: stratify by BOTH `stratify_col` and binned Dry_Total_g,
        and group by `group_col`.
        - Ensures groups (Sampling_Date) are not split across folds
        - Preserves label distribution of BOTH State and Dry_Total_g bins across folds
        """
        try:
            from sklearn.model_selection import StratifiedGroupKFold
        except Exception as e:
            raise ImportError("StratifiedGroupKFold requires scikit-learn >= 1.1. Please upgrade.") from e

        df = df.copy()
        num_bins = min(10, int(np.floor(1 + np.log2(len(df)))))
        print(f"Stratifying Dry_Total_g into {num_bins} bins")

        # Bin Dry_Total_g
        df['total_bin'] = pd.cut(
            df['Dry_Total_g'], 
            bins=num_bins, 
            labels=False,
            duplicates='drop'
        )

        # Create multi-stratify key: State + bin (as string, e.g., "VIC_0")
        df['stratify_key'] = df[stratify_col].astype(str) + "_" + df['total_bin'].astype(str)
        y = df['stratify_key'].values
        groups = df[group_col].astype(str).values

        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        df['fold'] = -1
        for fold_idx, (_, val_idx) in enumerate(sgkf.split(np.zeros(len(df)), y, groups)):
            df.loc[val_idx, 'fold'] = fold_idx

        self.df_wide = df    
        return df