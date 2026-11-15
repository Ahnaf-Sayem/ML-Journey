# Decision Tree From Scratch

A decision tree classifier built from scratch with NumPy (no sklearn for the model itself),
used to predict whether a person will default on a loan.

## How it works
- Splits are chosen by **Gini impurity**: for every feature and every possible threshold,
  the tree picks the split that reduces impurity the most.
- It grows recursively and stops at a maximum depth, when a node has too few samples
  (`min_sample_split`), or when a node is pure.
- Each leaf returns the **fraction of defaulters** in that leaf, so predictions are
  probabilities. This gives better AUC than hard 0/1 labels.

## Files
- `Decision_Trees_Scratch.py` - the `Decision_Tree` class
- `Data_cleaning.py` - loads the CSV and makes train/val/test splits (60/20/20)
- `Testing.py` - trains the tree and prints Train, Val and Test AUC
- `loan_2000_people.csv` - synthetic dataset, 2000 people
  (age, seniority, income, debt_pct, job_fixed, bad_record, owns_home, default)
- `Pipfile` / `Pipfile.lock` - dependencies

## How to run
1. Clone the repo and go into the project folder:
   ```
   git clone https://github.com/Ahnaf-Sayem/ML-Journey.git
   cd "ML-Journey/Decision Trees"
   ```
2. Install the dependencies (needs pipenv: `pip install pipenv`):
   ```
   pipenv install
   ```
3. Run it:
   ```
   pipenv run python Testing.py
   ```

## Results
| Set | AUC |
|-----|-----|
| Train | 0.855 |
| Val | 0.757 |
| Test | 0.717 |

For comparison, sklearn's `DecisionTreeClassifier` scores about 0.75 on validation
with the same split, so this implementation performs on par with it.

## What I learned
- Why overfitting happens and how `depth` and `min_sample_split` control it
- Why AUC needs probabilities instead of hard 0/1 predictions
- Why a small validation set gives noisy scores
