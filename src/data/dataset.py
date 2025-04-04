from torch.utils.data import Dataset
import torch
import random
import pandas as pd


def compute_train_stats(x, num_features):
    x_num = x[num_features]
    x_mean = x_num.mean()
    x_std = x_num.std()

    return {'mean': x_mean, 'std': x_std}


def add_stratification_target(df):
    """
    Add stratification target to the dataframe based on TA_F_MDS and ai.
    
    Args:
        df (DataFrame): Input dataframe containing 'TA_F_MDS' and 'ai' columns.

    Returns:
        DataFrame: Dataframe with stratification target added.
    """
    grouped = df.groupby('sitename').agg({'TA_F_MDS': 'mean', 'ai': 'first'})
    grouped['TA_F_MDS_bins'] = pd.qcut(grouped['TA_F_MDS'], 2, labels=False)
    grouped['ai_bins'] = pd.qcut(grouped['ai'], 2, labels=False)
    grouped['stratify'] = grouped['TA_F_MDS_bins'].astype(str) + '_' + grouped['ai_bins'].astype(str)
    grouped = grouped.drop(columns=['TA_F_MDS_bins', 'ai_bins'])

    return grouped


class gpp_dataset(Dataset):
    def __init__(self, x, train_stats, num_features, cat_features=None, test=False, chunk_size=128, overlap=32, max_offset=96):
        """
        A PyTorch Dataset for GPP prediction.

        Args:
            x (DataFrame): Input data containing numerical features and target variable.
            train_stats (dict): Dictionary containing mean and standard deviation for centering features.
            num_features (list): List of numerical feature names.
            cat_features (list): List of categorical feature names.
            test (bool): Flag indicating whether the dataset is for testing.
            chunk_size (int): Size of each chunk/window.
            overlap (int): Overlap size between consecutive chunks/windows.
            max_offset (int): Maximum random offset for varying the starting point.
        """
        self.test = test
        self.data = x
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.max_offset = max_offset
        self.cat = len(cat_features) > 0
        
        # Select numeric variables only
        x_num = x[num_features]

        # Center data, according to training data stats
        x_centered = (x_num - train_stats['mean']) / train_stats['std']

        # Create tensor for the covariates
        self.x = torch.tensor(x_centered.values, dtype=torch.float32)
        if self.cat:
            cat_features = [col for col in x.columns if any([col.startswith(cat) for cat in cat_features]) and col not in cat_features]
            x_cat = x[cat_features]
            self.c = torch.tensor(x_cat.values, dtype=torch.float32)
        
        # Define target        
        self.y = torch.tensor(x['GPP_NT_VUT_REF'].values, dtype=torch.float32)

        # Get max number of samples in one site
        self.max_samples = x.index.value_counts().max()
        
        if not self.test:
            self.sites = x.index.unique()
            self.sitename = x.index

            # Prepare lists to store chunks
            x_chunks_list = []
            y_chunks_list = []
            if self.cat:
                c_chunks_list = []

            for site in self.sites:
                site_mask = self.sitename == site
                timestamp_site = x["TIMESTAMP"][site_mask]
                
                x_site = self.x[site_mask]
                y_site = self.y[site_mask]

                # Randomize the start offset
                start_offset = random.randint(0, self.max_offset)
                
                # Create chunks for the site
                valid_chunks, valid_targets, valid_cats = self.create_sliding_chunks(
                    x_site, y_site, timestamp_site, chunk_size=self.chunk_size, 
                    overlap=self.overlap, start_offset=start_offset, cat_features=self.cat
                )
                
                x_chunks_list.extend(valid_chunks)
                y_chunks_list.extend(valid_targets)
                if self.cat:
                    c_chunks_list.extend(valid_cats)

            # Convert lists to tensors
            self.x = torch.stack(x_chunks_list)
            self.y = torch.stack(y_chunks_list).squeeze(-1)
            if self.cat:
                self.c = torch.stack(c_chunks_list)
            self.len = len(self.x)
        else:
            # if validation or test dataset, take entire site time series as one chunk
            self.sitename = x.index
            self.sites = x.index.unique()
            self.len = len(self.sites)

    def create_sliding_chunks(self, x_site, y_site, timestamp_site, chunk_size, overlap, start_offset=0, cat_features=False):
        """
        Create sliding chunks of data for a single site.
        Args:
            x_site (Tensor): Covariates for the site.
            y_site (Tensor): Target variable for the site.
            timestamp_site (Series): Timestamps for the site.
            chunk_size (int): Size of each chunk/window.
            overlap (int): Overlap size between consecutive chunks/windows.
            start_offset (int): Random offset to start chunking.
            cat_features (bool): Flag indicating if categorical features are present.
        Returns: 
            Tuple of lists containing chunks of covariates, target variable, and categorical features.
        """
        x_chunks = []
        y_chunks = []
        c_chunks = []
        
        step = chunk_size - overlap
        start = start_offset
        for i in range(start, len(x_site), step):
            end = i + chunk_size
            if end <= len(x_site):
                chunk_timestamps = timestamp_site.iloc[i:end]
                years_in_chunk = chunk_timestamps.dt.year.unique()

                # Ensure the chunk does not span across non-consecutive years
                if len(years_in_chunk) > 1:
                    if not self.are_years_consecutive(years_in_chunk):
                        continue  # Skip chunks that span non-consecutive years

                x_chunks.append(x_site[i:end])
                y_chunks.append(y_site[i:end])
                if cat_features:
                    c_chunks.append(self.c[i:end])
        
        return x_chunks, y_chunks, c_chunks
    
    def are_years_consecutive(self, years):
        """
        Check if all years in the list are consecutive.
        
        Args:
            years (array-like): Array of unique years in a chunk.

        Returns:
            bool: True if all years are consecutive, False otherwise.
        """
        return all((years[i+1] - years[i]) == 1 for i in range(len(years) - 1))

    def __getitem__(self, idx):
        """
        Get the covariates and target variable for a specific chunk.

        Args:
            idx (int): Index of the chunk.

        Returns:
            Tuple of numerical covariates and target variable for the specified chunk.
        """
        if not self.test:
            if self.cat:
                return self.x[idx], self.y[idx], self.c[idx]
            else:
                return self.x[idx], self.y[idx]
        else:
            rows = [s == self.sites[idx] for s in self.sitename]
            if self.cat:
                return self.x[rows], self.y[rows], self.c[rows]
            else:
                return self.x[rows], self.y[rows]

    def __len__(self):
        """
        Get the total number of chunks in the dataset.

        Returns:
            int: The number of chunks in the dataset.
        """
        return self.len