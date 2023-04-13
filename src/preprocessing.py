import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor


def preprocess(df, outputs, datetime, classification):
    data = Preprocess()
    data.numeric(df, outputs, datetime, classification)
    data.impute()
    data.outliers(frac=0.05)

    return data.data


class Preprocess:
    def numeric(self, df, outputs, datetime, classification):
        self.data = df[outputs].copy()
        df = df.drop(columns=outputs)
        self.outputs = outputs
        self.datetime = datetime
        self.time_series = datetime is not None
        self.classification = classification

        if self.time_series:
            self.data[datetime] = df[datetime]
            df = df.drop(columns=datetime)

        # determine which columns are strings
        strings = np.where(df.dtypes == "object")[0]
        text = df.iloc[:, strings]
        df = df.drop(columns=df.columns[strings])

        # make sure non-string columns are numeric
        for c in df.columns:
            df[c] = df[c].astype("float")
        if not self.classification:
            for out in self.outputs:
                self.data[out] = self.data[out].astype("float")

        # collect the words (and their inverse frequencies) from each document
        # 'matrix' is a term (columns) document (rows) matrix
        matrix = pd.DataFrame()
        for c in text.columns:
            vector = TfidfVectorizer()
            matrix2 = vector.fit_transform(text[c].tolist())
            names = vector.get_feature_names_out()
            names = [str(c) + " " + str(n) for n in names]
            matrix2 = pd.DataFrame(matrix2.toarray(), columns=names)
            matrix = pd.concat([matrix, matrix2], axis=1)

        self.data = pd.concat([self.data, df, matrix], axis="columns")

    def impute(self):
        if self.data.isna().any().any():
            if self.classification:
                Y = self.data[self.outputs].copy()
                X = self.data.drop(columns=self.outputs).copy()
                if self.time_series:
                    X = X.drop(columns=self.datetime)

                # fill in missing values with the most frequent column value (for Y)
                impute = SimpleImputer(strategy="most_frequent")
                y_columns = Y.columns
                Y = pd.DataFrame(data=impute.fit_transform(Y), columns=y_columns)

                # fill in missing values with multivariate imputation (for X)
                impute = IterativeImputer(random_state=42)
                x_columns = X.columns
                X = pd.DataFrame(data=impute.fit_transform(X), columns=x_columns)

                if self.time_series:
                    self.data = pd.concat([self.data[[self.datetime]], Y, X], axis="columns")
                else:
                    self.data = pd.concat([Y, X], axis="columns")
            
            else:
                if self.time_series:
                    df = self.data.drop(columns=self.datetime).copy()
                else:
                    df = self.data.copy()
                
                # fill in missing values with multivariate imputation
                impute = IterativeImputer(random_state=42)
                df_columns = df.columns
                df = pd.DataFrame(data=impute.fit_transform(df), columns=df_columns)

                if self.time_series:
                    self.data = pd.concat([self.data[[self.datetime]], df], axis="columns")
                else:
                    self.data = df.copy()

    def outliers(self, frac):
        if not self.time_series:
            if self.classification:
                df = self.data.drop(columns=self.outputs).copy()
            else:
                df = self.data.copy()

            # train a model to detect outliers
            model = LocalOutlierFactor(n_neighbors=20, leaf_size=30, novelty=False)
            model.fit(df)

            # remove the outliers
            cutoff = np.quantile(model.negative_outlier_factor_, frac)
            good_idx = np.where(model.negative_outlier_factor_ > cutoff)[0]
            df = df.iloc[good_idx, :].reset_index(drop=True)

            if self.classification:
                self.data = self.data.iloc[good_idx, :].reset_index(drop=True)
                self.data = pd.concat([self.data[self.outputs], df], axis="columns")
            else:
                self.data = df.copy()
