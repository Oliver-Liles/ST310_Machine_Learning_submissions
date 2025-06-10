setwd("~/ST310/PredChallenge")

# Change to your number
candidate_num <- 50537

# Change file_path to wherever you saved the data
file_path <- "~/ST310/PredChallenge/data/"

# -----------------------
# Bigdata challenge

## Load data
bigdata_train <- read.csv(paste0(file_path, "ST310_2024_bigdata_train.csv"))
bigdata_test <- read.csv(paste0(file_path, "ST310_2024_bigdata_test.csv"))
y_train <- bigdata_train[, 1]
x_train <- as.matrix(bigdata_train[, -1]) # leave out y
x_test <- as.matrix(bigdata_test) # doesn't come with y

## Fit models
library(car)
library(tidyverse)
library(gridExtra)
library(ggplot2)
library(boot)
library(gam)

### exploring the data
summary(bigdata_train)
summary(bigdata_test)
dim(bigdata_train)


### correlation in data
corr_matrix <- cor(x_train)
corrplot::corrplot(corr_matrix, method = "circle")


### all predictors
lm_all <- lm(y ~ ., data = bigdata_train) # simple baseline
summary(lm_all) # adj. R-squared is 0.485
vif(lm_all)

#### k-fold cross-validation, k = 10
set.seed(1)
glm_all <- glm(y ~ ., data = bigdata_train)
cv_error_all <- cv.glm(bigdata_train, glm_all, K = 10)$delta[1]
cv_error_all # 27.71613

### One by one selection using as many significant predictors as possible
lm_max_pred <- lm(y ~ x1 + x2 + x3 + x20 + x27 + x6 * x5 + x8 + x10 * x9 + 
                    x12 * x11 + x21 + x25 * x26 + x17 + x14 + x22 + x24, data = bigdata_train)
summary(lm_max_pred) # adj. R-squared is 0.4972
vif(lm_max_pred) # x11, x6, x5, x10, x9, x12, x25, x26 all have significant (>10) multicolinearity

#### k-fold cross-validation, k = 10
set.seed(1)
glm_max_pred <- glm(y ~ x1 + x2 + x3 + x20 + x27 + x6 * x5 + x8 + x10 * x9 + 
                      x12 * x11 + x21 + x25 * x26 + x17 + x14 + x22 + x24, data = bigdata_train)
cv_error_maxp <- cv.glm(bigdata_train, glm_max_pred, K = 10)$delta[1]
cv_error_maxp # lower than model with all predictors


### Mixed selection using R-squared
lm_mix_1 <- lm(y ~ x22 + x21 + x4 + x19 + x20 + x17 * x18 + x16, data = bigdata_train)
summary(lm_mix_1) # adj. R-squared is 0.1133
vif(lm_mix_1) # x17, x18 all have significant (>10) multicolinearity

lm_mix_2 <- lm(y ~ x22 + x21 + x4 + x1 + x19 + x20 + x17 + x16, data = bigdata_train)
summary(lm_mix_2) # adj. R-squared is 0.06375
vif(lm_mix_2) # insignificant multicolinearity

lm_mix_3 <- lm(y ~ x22 + x21 + x4 + x1 + x19 + x20 + x17, data = bigdata_train)
summary(lm_mix_3) # adj. R-squared is 0.04702
vif(lm_mix_3) # insignificant multicolinearity

lm_mix_4 <- lm(y ~ x22 + x21 + x4 + x1 + x19 + x20 + x18, data = bigdata_train)
summary(lm_mix_4) # adj. R-squared is 0.04548
vif(lm_mix_4) # insignificant multicolinearity

#### k-fold cross-validation, k = 10
set.seed(1)
cv_error <- rep(0, 4)
glm_mix_1 <- glm(y ~ x22 + x21 + x4 + x19 + x20 + x17 * x18 + x16, data = bigdata_train)
cv_error[1] <- cv.glm(bigdata_train, glm_mix_1, K = 10)$delta[1]
glm_mix_2 <- glm(y ~ x22 + x21 + x4 + x1 + x19 + x20 + x17 + x16, data = bigdata_train)
cv_error[2] <- cv.glm(bigdata_train, glm_mix_2, K = 10)$delta[1]
glm_mix_3 <- glm(y ~ x22 + x21 + x4 + x1 + x19 + x20 + x17, data = bigdata_train)
cv_error[3] <- cv.glm(bigdata_train, glm_mix_3, K = 10)$delta[1]
glm_mix_4 <- glm(y ~ x22 + x21 + x4 + x1 + x19 + x20 + x18, data = bigdata_train)
cv_error[4] <- cv.glm(bigdata_train, glm_mix_4, K = 10)$delta[1]
cv_error # lm_mix_1 is the best for avoiding over fitting


### Backwards selection
lm_back_sel <- lm(y ~ . - x14 - x15 - x7 - x24 - x16 - x19 - x18 - x4 + x5:x6 + 
                    x9:x10 + x11:x12 + x25:x26, data = bigdata_train)
summary(lm_back_sel) # adj. R-squared is 0.4981
vif(lm_back_sel) # x5, x6, x9, x10, x11, x12, x25, x26 have significantly multicolinearity

#### k-fold cross-validation, k = 10
set.seed(1)
glm_back_sel <- glm(y ~ . - x14 - x15 - x7 - x24 - x16 - x19 - x18 - x4 + x5:x6 + 
                 x9:x10 + x11:x12 + x25:x26, data = bigdata_train)
cv_error_back <- cv.glm(bigdata_train, glm_back_sel, K = 10)$delta[1]
cv_error_back # lower than model with all predictors

# GAM backwards selection
gam_back_sel <- gam(y ~ . - x24 - x16 - x10 - x9 + x5:x6 + x7:x8 + x11:x12 + 
                      x13:x14 + x17:x18 + x19:x20 + x21:x22, data = bigdata_train)
summary.Gam(gam_back_sel)
vif(gam_back_sel) # x5, x6, x7, x8, x11, x12, x13, x14, x17, x18, x19, x20, x21, x22 
                  # had significant multicolinearity
gam_pred <- predict.Gam(gam_back_sel, newdata=bigdata_train)
mean((gam_pred - y_train)^2) # MSE is 26.58105

###### k-fold cross-validation, k = 10
set.seed(1)
glm_gam_back_sel <- glm(y ~ . - x14 - x15 - x7 - x24 - x16 - x19 - x18 - x4 + x5:x6 + 
                      x9:x10 + x11:x12 + x25:x26, data = bigdata_train)
cv_error_back <- cv.glm(bigdata_train, glm_gam_back_sel, K = 10)$delta[1]
cv_error_back # lower than model with all predictors


### deciding between models 
anova(lm_all, lm_max_pred, lm_mix_1, lm_back_sel, gam_back_sel, test = "F") 
#### lm_mix_1 and lm_back_sel, gam_back_sel are valid options
#### to avoid an overfit model, I choose lm_mix 1 as the best model


## Choose your best model
bigdata_best <- lm_mix_1

## Save best predictions
### Note: depending on your code and model choice you may
### need to change the code that saves predictions
### Be sure to check it is saved in the right format!
bigdata_test_y <- data.frame(y = predict(bigdata_best, newdata = bigdata_test))
names(bigdata_test_y) <- "y"

## Check for correct number of predictions
nrow(bigdata_test_y) == nrow(bigdata_test)

## Check that they have the same format
cbind(head(bigdata_test_y$y), head(bigdata_train$y))

## Save results for upload to Moodle
out_file <- paste0(file_path, "bigdata_", candidate_num, ".csv")
write.csv(bigdata_test_y, out_file, row.names = FALSE, quote = FALSE)



# -----------------------
# High-dim. challenge

## Load data
highdim_train <- read.csv(paste0(file_path, "ST310_2024_highdim_train.csv"))
highdim_test <- read.csv(paste0(file_path, "ST310_2024_highdim_test.csv"))
y_train <- highdim_train[, 1]
x_train <- as.matrix(highdim_train[, -1])
x_test <- as.matrix(highdim_test)

## Fit models

### exploring the data
summary(highdim_train[1:40])
summary(highdim_test[1:40])
highdim_train[is.na(highdim_train) | highdim_train=="Inf"] <- NA
sum(is.na(highdim_train))
dim(highdim_train)


### all predictors
lm_all <- lm(y ~ ., data = highdim_train, na.action=na.omit) # simple baseline
summary(lm_all)


### shrinkage
library(glmnet)

y_vec <- y_train
x_mat <- model.matrix(y ~ ., data = highdim_train)[, -1]
length(y_vec)
dim(x_mat)

grid <- 10^seq(10, -2, length = 100)

#### ridge
glm_ridge <- glmnet(x_mat, y_vec, alpha = 0, lambda = grid)
dim(coef(glm_ridge))
summary(glm_ridge)

glm_ridge$lambda[50]
coef(glm_ridge)[, 50]
sqrt(sum(coef(glm_ridge)[-1, 50]^2))

predict(glm_ridge , s = 50, type = "coefficients")[1:20, ]

set.seed(1)
train <- sample(1:nrow(x_mat), nrow(x_mat)/2)
test <- (-train)
y_test <- y_vec[test]

ridge_train <- glmnet(x_mat[train, ], y_vec[train], alpha = 0, lambda = grid , thresh = 1e-12)

mean((mean(y_vec[train]) - y_test)^2) # MSE of model with just an intercept

ridge_pred_ls <- predict(ridge_train , s = 0, newx = x_mat[test , ], exact = T, x = x_mat[train, ], y = y_vec[train]) # check this
mean((ridge_pred_ls - y_test)^2) # MSE of using least squares(lambda = 0)

ridge_pred <- predict(ridge_train , s = 100, newx = x_mat[test , ])
mean((ridge_pred - y_test)^2) # MSE using lambda = s

set.seed(1)
cv_out <- cv.glmnet(x_mat[train, ], y_vec[train], alpha = 0)
plot(cv_out)
bestlam <- cv_out$lambda.min
bestlam

ridge_pred <- predict(ridge_train, s = bestlam, newx = x_mat[test, ])
mean((ridge_pred - y_test)^2) # MSE using bestlam

out <- glmnet(x_mat, y_vec, alpha = 0)
ridge_model <- predict(out, type = "coefficients", s = bestlam)
ridge_model

#### lasso
lasso_train <- glmnet(x_mat[train, ], y_vec[train], alpha = 1, lambda = grid)
plot(lasso_train)

set.seed(1)
cv_out <- cv.glmnet(x_mat[train, ], y_vec[train], alpha = 1)
plot(cv_out)
bestlam <- cv_out$lambda.min
bestlam

lasso_pred_train <- predict(lasso_train, s = bestlam, newx = x_mat[test, ])
mean((lasso_pred_train - y_test)^2) # MSE

lasso_model <- glmnet(x_mat, y_vec, alpha = 1, lambda = bestlam)
lasso_coef <- predict(lasso_model , type = "coefficients", s = bestlam)
lasso_coef

lasso_pred <- predict(lasso_model, s = bestlam, newx = x_mat[test, ])
mean((lasso_pred - y_test)^2) # MSE



### deciding between models

####  lasso has the best test MSE and removes (sets coefficient to zero for) the 
####  less significant predictors which may be helpful for this high dimensional
####  data

## Choose your best model
highdim_best <- lasso_model

## Save best predictions
highdim_test <- model.matrix(~ ., data = highdim_test[, -1])
dim(highdim_test)

highdim_test_y <- data.frame(y = predict(highdim_best, s=bestlam, newx = highdim_test))
names(highdim_test_y) <- "y"

## Check for correct number of predictions
nrow(highdim_test_y) == nrow(highdim_test)

## Check that they have the same format
cbind(head(highdim_test_y$y), head(highdim_train$y))

## Save results for upload to Moodle
out_file <- paste0(file_path, "highdim_", candidate_num, ".csv")
write.csv(highdim_test_y, out_file, row.names = FALSE, quote = FALSE)


# -----------------------
# Classify challenge

## Load data
classify_train <- read.csv(paste0(file_path, "ST310_2024_classify_train.csv"))
classify_train$y <- as.factor(classify_train$y)
classify_test <- read.csv(paste0(file_path, "ST310_2024_classify_test.csv"))
y_train <- classify_train[, 1]
x_train <- as.matrix(classify_train[, -1])
x_test <- as.matrix(classify_test)

## Fit models

### exploring data
summary(classify_train)
summary(classify_test)
dim(classify_train)


### correlation in data
corr_matrix <- cor(x_train)
corrplot::corrplot(corr_matrix, method = "circle") # x13, x10 have some correlation


### all predictors
glm_all <- glm(y ~ ., data = classify_train, family = "binomial")
summary(glm_all)
glm_probs_all <- predict(glm_all, type = "response")
glm_pred <- rep("0", 512)
glm_pred[glm_probs_all > .5] = "1"
table(glm_pred, y_train)
mean(glm_pred == y_train) # 0.843 on all data, 0.843 for test/train split
vif(glm_all) # no significant multicolinearity


### logistic regression
glm_lr_back <- glm(y ~ . - x13 - x14 - x9 - x12 - x7, data = classify_train, 
                   family = "binomial") # backward selection for predictors
summary(glm_lr_back)
glm_probs_b <- predict(glm_lr_back, type = "response")
glm_pred_b <- rep("0", 512)
glm_pred_b[glm_probs_b > .5] = "1"
table(glm_pred_b, y_train)
mean(glm_pred_b == y_train) # 0.830 on all data, 0.835 for test/train split

glm_lr_mix_s <- glm(y ~ x1 + x3 + x6 + x8 + x2 + x4 + x5 + x13 + x11, data = classify_train,
                  family = "binomial") # mixed selection for predictors by success
summary(glm_lr_mix_s)
glm_probs_ms <- predict(glm_lr_mix_s, type = "response")
glm_pred_ms <- rep("0", 512)
glm_pred_ms[glm_probs_ms > .5] = "1"
table(glm_pred_ms, y_train)
mean(glm_pred_ms == y_train) # 0.828 on all data, 0.835 for test/train split

glm_lr_mix_a <- glm(y ~ x1 + x8 + x6 + x3 + x2 + x5 + x4 + x10 + x11 + x7 + x12, data = classify_train,
                  family = "binomial") # mixed selection for predictors by AIC
summary(glm_lr_mix_a)
glm_probs_ma <- predict(glm_lr_mix_a, type = "response")
glm_pred_ma <- rep("0", 512)
glm_pred_ma[glm_probs_ma > .5] = "1"
table(glm_pred_ma, y_train)
mean(glm_pred_ma == y_train) # 0.845 on all data, 0.859 for test/train split
vif(glm_lr_mix_a) # no significant multicolinearity

set.seed(1)
train <- sample(1:512, 384) # creating a training/ test sample split (75:25)
test <- -train

glm_train <- glm(y ~ x1 + x8 + x6 + x3 + x2 + x5 + x4 + x10 + x11 + x7 + x12, data = classify_train,
                 family = "binomial", subset = train)
glm_probs_train <- predict(glm_train , classify_train[test, ] ,type = "response")
glm_pred_t <- rep("0", 128)
glm_pred_t[glm_probs_train > .5] <- "1"
table(glm_pred_t, y_train[test])
mean(glm_pred_t == y_train[test])


### LDA
library(MASS)
lda_fit <- lda(y ~ x1 + x8 + x6 + x3 + x2 + x5 + x4 + x10 + x11 + x7 + x12,
               data = classify_train, subset = train)
lda_fit
plot(lda_fit)
lda_pred <- predict(lda_fit, classify_train[test, ])
lda_class <- lda_pred$class
table(lda_class, y_train[test])
mean(lda_class == y_train[test]) # 0.867 test value


### QDA
qda_fit <- qda(y ~ x1 + x8 + x6 + x3 + x2 + x5 + x4 + x10 + x11 + x7 + x12,
                   data = classify_train, subset = train)
qda_fit
qda_class <- predict(qda_fit, classify_train[test, ])$class
table(qda_class, y_train[test])
mean(qda_class == y_train[test]) # 0.820 test value


### naive Bayes
library(e1071)
nb_fit <- naiveBayes(y ~ x1 + x8 + x6 + x3 + x2 + x5 + x4 + x10 + x11 + x7 + x12,
                         data = classify_train, subset = train)
nb_fit
nb_class <- predict(nb_fit, classify_train[test, ])
table(nb_class, y_train[test])
mean(nb_class == y_train[test]) # 0.765 test value


### K-Nearest Neighbours (KNN)
library(class)
attach(classify_train)
train.X <- cbind(x1, x2, x3, x4, x5, x6, x7, x8, x10, x11, x12)[train, ]
test.X <- cbind(x1, x2, x3, x4, x5, x6, x7, x8, x10, x11, x12)[test, ]
train.Y<- y_train[train]

set.seed(1)
knn_pred <- knn(train.X, test.X, train.Y, k = 22)
table(knn_pred , y_train[test])
mean(knn_pred == y_train[test]) # k=22 gives best test value of 0.796


### GAMs
library(gam)

#### all predictors
gam_all <- gam(y ~ ., data = classify_train, family = "binomial")
summary(gam_all)
gam_probs_all <- predict(gam_all, type = "response")
gam_pred <- rep("0", 512)
gam_pred[gam_probs_all > .5] = "1"
table(gam_pred, y_train)
mean(gam_pred == y_train) # 0.843 on all data, 0.843 on test data

#### back selection
gam_back <- gam(y ~ . - x13 - x14 - x12 - x9 - x2 - x10 - x7 - x11, 
               family = binomial, data = classify_train)
summary(gam_back)
gam_probs_back <- predict(gam_back, type = "response")
gam_pred_b <- rep("0", 512)
gam_pred_b[gam_probs_back > .5] = "1"
table(gam_pred_b, y_train)
mean(gam_pred_b == y_train) # 0.804 on all data, 0.812 on test data

### mixed selection
gam_mix <- gam(y ~ x1 + x8 + x6 + x3 + x2 + x5 + x4 + x10 + x11 + x7 + x12, 
                family = binomial, data = classify_train)
summary(gam_mix)
gam_probs_mix <- predict(gam_mix, type = "response")
gam_pred_m <- rep("0", 512)
gam_pred_m[gam_probs_mix > .5] = "1"
table(gam_pred_m, y_train)
mean(gam_pred_m == y_train) # 0.845 on all data, 0.859 on test data

gam_train <- gam(y ~ x1 + x8 + x6 + x3 + x2 + x5 + x4 + x10 + x11 + x7 + x12, data = classify_train, family = "binomial", subset = train)
gam_probs_train <- predict(gam_train, classify_train[test, ], type = "response")
gam_pred_t <- rep("0", 128)
gam_pred_t[gam_probs_train > .5] <- "1"
table(gam_pred_t, y_train[test])
mean(gam_pred_t == y_train[test])



### deciding between model

#### the model with the best test result is lda_fit but now, i'll update it for 
#### the test data which has 68% positive cases and train it on all the data
#### [assumption: positive means '1' in the instructions]

lda_test_model <- lda(y ~ x1 + x8 + x6 + x3 + x2 + x5 + x4 + x10 + x11 + x7 + x12,
               data = classify_train, prior=c(0.4, 0.6))
lda_test_model

## Choose your best model
classify_best <- lda_test_model

## Save best predictions
probs_test_y <- predict(classify_best, newdata = classify_test, type = "response")
classify_test_y <- data.frame(
  y = as.factor(probs_test_y$class)
)
names(classify_test_y) <- "y"
summary(classify_test_y) # 69% of predictions are positive (=='1')


## Check for correct number of predictions
nrow(classify_test_y) == nrow(classify_test)

## Check that predictions are classifications with same format
## Note: you would not want to do this for a numeric outcome
sort(unique(classify_test_y$y)) == sort(unique(classify_train$y))

## Save results for upload to Moodle
out_file <- paste0(file_path, "classify_", candidate_num, ".csv")
write.csv(classify_test_y, out_file, row.names = FALSE, quote = FALSE)
