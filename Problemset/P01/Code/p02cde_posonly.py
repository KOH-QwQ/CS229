import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')
    x_train,t_train=util.load_dataset(train_path,'t',add_intercept=True)
    x_train,y_train=util.load_dataset(train_path,'y',add_intercept=True)
    x_eval,t_eval=util.load_dataset(test_path,'t',add_intercept=True)
    x_eval,y_eval=util.load_dataset(test_path,'y',add_intercept=True)

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    model1=LogisticRegression()
    model1.fit(x_train,t_train)
    util.plot(x_eval,t_eval,model1.theta,r"C:\Users\17270\Desktop\cs229-2018-autumn-main\problem-sets\PS1\P02(c).png")
    
    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    model2=LogisticRegression()
    model2.fit(x_train,y_train)
    util.plot(x_eval,y_eval,model2.theta,r"C:\Users\17270\Desktop\cs229-2018-autumn-main\problem-sets\PS1\P02(d).png")

    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    x_valid,t_valid=util.load_dataset(valid_path,'t',add_intercept=True)
    x_valid,y_valid=util.load_dataset(valid_path,'y',add_intercept=True)
    model3=LogisticRegression()
    model3.fit(x_train,y_train)
    h_valid=model3.predict(x_valid)
    # print(h_valid)
    h_true=h_valid[y_valid==1]
    alpha=np.mean(h_true)

    h_eval=model3.predict(x_eval)
    h_eval/=alpha
    # print(alpha)
    correctness=1+np.log((2-alpha)/(alpha))/model3.theta[0]
    # print(correctness)
    util.plot(x_eval,(h_eval>=0.5),model3.theta,r"C:\Users\17270\Desktop\cs229-2018-autumn-main\problem-sets\PS1\P02(e)1.png",correctness)
    

    
    # *** END CODER HERE

train_path=r"C:\Users\17270\Desktop\cs229-2018-autumn-main\problem-sets\PS1\data\ds3_train.csv"
valid_path=r"C:\Users\17270\Desktop\cs229-2018-autumn-main\problem-sets\PS1\data\ds3_valid.csv"
test_path=r"C:\Users\17270\Desktop\cs229-2018-autumn-main\problem-sets\PS1\data\ds3_test.csv"
pred_path=r"C:\Users\17270\Desktop\cs229-2018-autumn-main\problem-sets\PS1\data\X.txt"

main(train_path, valid_path, test_path, pred_path)