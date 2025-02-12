import numpy as np
import util
import sys
from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    model=GDA()
    model.fit(x_train,y_train)
    x_eval,y_eval=util.load_dataset(eval_path, add_intercept=False)
    prediction=model.predict(x_eval)
    np.savetxt(pred_path,prediction,fmt="%d")

    accuracy=np.mean(prediction==y_eval)
    print(f"{accuracy:.3f}")
    # *** END CODE HERE ***



class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        phi=np.sum(y==1)/m
        mu0=np.zeros(n)
        mu1=np.zeros(n)
        for i in range(m):
            if(y[i]==1): 
                mu1=mu1+x[i]
            else:
                mu0=mu0+x[i]
        mu1/=np.sum(y==1)
        mu0/=np.sum(y==0)
        
        sigma=np.zeros((n,n))
        for i in range(m):
            if y[i]==1 : 
                sigma += np.outer(x[i]-mu1,x[i]-mu1)
            else:
                sigma += np.outer(x[i]-mu0,x[i]-mu0)
        
        sigma/=m
        
        sigma_inv=np.linalg.inv(sigma)

        self.theta_0=(mu0.T @ sigma_inv @ mu0 - mu1.T @ sigma_inv @ mu1)/2 + np.log(phi/(1-phi))
        self.theta=sigma_inv @ (mu1-mu0)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return (1/(1+np.exp(-1*(self.theta_0+x @ self.theta)))>=0.5).astype(int)
        # *** END CODE HERE
        
if __name__ == "__main__":
    train_path = r"C:\\Users\17270\Desktop\cs229-2018-autumn-main\problem-sets\PS1\data\ds1_train.csv"  
    eval_path = r"C:\\Users\17270\Desktop\cs229-2018-autumn-main\problem-sets\PS1\data\ds1_valid.csv"  
    pred_path = r"C:\\Users\17270\Desktop\cs229-2018-autumn-main\problem-sets\PS1\data\p01prediction_gda.txt"  

    main(train_path, eval_path, pred_path)
