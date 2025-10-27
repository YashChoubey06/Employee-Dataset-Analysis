# Employee-Dataset-Analysis



A project on Data Extraction and Processing where an employee database having inconsistensies like missing values, outliers, inconsistent data etc is cleaned, transformed, and an EDA (Exploratory Data Analysis) is done. The data is visualised using different graphs, summary statistics, relationships between features is obtained.

The script is a practical example of preparing real-world data for analysis and visualization.

## Key Features & Abilities Showcased

* **Data Cleaning:**
    * **Missing Value Imputation:** Fills `NaN` values using statistical methods (median for numerical, mode for categorical).
    * **Text Standardization:** Corrects inconsistent text entries (e.g., "Male", "male ", "MALE") using string manipulation.
    * **Data Type Correction:** Converts columns to their proper data types (e.g., from `object` to `numeric`), handling conversion errors.
    * **Duplicate Removal:** Identifies and drops duplicate records from the dataset.
    * **Outlier Handling:** Uses the Interquartile Range (IQR) method to identify and clip outliers in numerical columns.

* **Feature Engineering:**
    * **Feature Creation:** Generates a new `YearsInCompany` column from existing data.
    * **Data Transformation:** Encodes categorical features (like `City`, `Gender`) into a machine-readable format using `scikit-learn`'s `LabelEncoder`.

* **Exploratory Data Analysis (EDA):**
    * **Statistical Summary:** Generates descriptive statistics (`.describe()`) for key features.
    * **Data Visualization:** Creates several plots to understand the data:
        * **Histograms** and **Boxplots** (via `seaborn`) to view the distribution of `Age`, `PaymentTier`, etc.
        * A **Correlation Heatmap** to visualize the relationships between variables, especially their impact on `LeaveOrNot`.
    * **Insight Generation:** Calculates key metrics like the overall attrition rate and breaks down attrition by `PaymentTier`, `City`, and `Gender` using `groupby`.

* **Output:**
    * Saves the fully processed data to a new `Employee_Cleaned.csv` file.

## Technologies Used

* **Python**
* **pandas:** For data manipulation and analysis.
* **numpy:** For numerical operations.
* **matplotlib** & **seaborn:** For data visualization.
* **scikit-learn:** For data preprocessing (`LabelEncoder`).
