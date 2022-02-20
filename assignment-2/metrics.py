
def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)
    # TODO: Write here
    count=0
    for i in range(y.size):
        if (y[i]==y_hat[i]):
            count = count + 1
    acc = count/y.size
    return acc
    pass

def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    assert(y_hat.size == y.size)
    count=0
    dem=0
    for i in range(y.size):
        if (y_hat[i]==cls):
            dem = dem + 1
            if (y[i]==cls):
                count = count + 1
    if (dem==0):
        return -1
    prec = count/dem
    return prec
    pass

def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    assert(y_hat.size == y.size)
    count=0
    dem=0
    for i in range(y.size):
        if (y[i]==cls):
            dem = dem + 1
            if (y_hat[i]==cls):
                count = count + 1
    recal = count/dem
    return recal
    pass

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    assert(y_hat.size == y.size)
    loss=0
    for i in range(y.size):
        loss = loss + ((y_hat[i]-y[i])**2)
    loss = loss / y.size
    loss = loss**(1/2)
    return loss
    pass

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    assert(y_hat.size == y.size)
    loss=0
    for i in range(y.size):
        loss = loss + abs(y_hat[i]-y[i])
    loss = loss / y.size
    return loss
    pass
