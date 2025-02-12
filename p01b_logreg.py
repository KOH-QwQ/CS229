import numpy as np
import util
import sys
from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    print(train_path)
      
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_eval,y_eval=util.load_dataset(eval_path, add_intercept=True)
    model=LogisticRegression()
    model.fit(x_train,y_train)
    prediction = model.predict(x_eval)
    np.savetxt(pred_path,prediction,fmt="%d")
    accuracy=np.mean(prediction==y_eval)
    print(f"{accuracy:.3f}")

    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m,n=x.shape
        self.theta=np.zeros(n)
        epsilon=1e-5
        while True:
            h=self.sigmoid(x @ self.theta)
            gradient=x.T @ (h-y) /m
            H=x.T*(h*(1-h)) @ x /m           
            delta = np.linalg.inv(H) @ gradient
            if(delta.dot(delta) **0.5 <=epsilon):
                break
            self.theta=self.theta-delta
        # *** END CODE HERE ***

    def Judge(self,z):
            return z>=0.5
    
    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return self.Judge(self.sigmoid(x @ self.theta))    
        # *** END CODE HERE ***

if __name__ == "__main__":
    train_path = r"C:\\Users\17270\Desktop\cs229-2018-autumn-main\problem-sets\PS1\data\ds1_train.csv"  
    eval_path = r"C:\\Users\17270\Desktop\cs229-2018-autumn-main\problem-sets\PS1\data\ds1_valid.csv"  
    pred_path = r"C:\\Users\17270\Desktop\cs229-2018-autumn-main\problem-sets\PS1\data\p01prediction_logreg.txt"  

    # 运行 main() 函数
    main(train_path, eval_path, pred_path)
