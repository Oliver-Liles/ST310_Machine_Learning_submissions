**Best Model Descriptions**
  
I used the following packages for this task: 
  
`car, tidyverse, gridExtra, ggplot2, boot, gam, glmnet, MASS, e1071, and class.`
  
These are the models, tuning parameters and how I chose them:

1. Big data: I used a mixed selection linear regression model which I then used k-fold cross validation on to compare to other models (some were also mixed selection, and some were others like backwards selection and GAMs).
2. High dimensional data: I used a lasso model as it had a better MSE (when trained on training data and applied to test data after splitting the data) than ridge and a linear regression model.
3. Classification data: I used LDA as it had the best classification rate of the model I used (when using a training & test data split) and then edited it to be trained on all the data and with ‘prior=c(0.4, 0.6)’ for the classification split as the instructions said the test data has “about 68% positive cases”.

