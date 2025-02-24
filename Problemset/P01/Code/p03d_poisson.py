import numpy as np
import util
import matplotlib.pyplot as plt
from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    clf=PoissonRegression(step_size=lr,eps=1e-5)
    print(clf.step_size)

    clf.fit(x_train,y_train)
    x_eval,y_eval=util.load_dataset(eval_path,add_intercept=False)
    print(y_eval)

    y_pred=clf.predict(x_eval)
    plt.figure()
    plt.plot(y_eval, y_pred, 'bx')
    plt.xlabel('True count')
    plt.ylabel('Prediction')
    plt.savefig(r'C:\Users\17270\Desktop\cs229-2018-autumn-main\problem-sets\PS1\P03d.png')
    np.savetxt(pred_path, y_pred)
class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m,n=x.shape
        self.theta=np.zeros(n)
        # print(m,n)
        while(True):
            thetap = np.copy(self.theta)
            self.theta += self.step_size * x.T @ (y - np.exp(x @ (self.theta)))
            if np.linalg.norm(self.theta - thetap, ord=1) < self.eps:
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        return np.exp(x @ self.theta)
        # *** END CODE HERE ***

# lr=1e-10
# train_path = r"C:\Users\17270\Desktop\cs229-2018-autumn-main\problem-sets\PS1\data\ds4_train.csv"  
# eval_path = r"C:\Users\17270\Desktop\cs229-2018-autumn-main\problem-sets\PS1\data\ds4_valid.csv"  
# pred_path = r"C:\Users\17270\Desktop\cs229-2018-autumn-main\problem-sets\PS1\data\.csv"  

# main(lr,train_path, train_path, eval_path)