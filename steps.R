library(caret)
library(randomForest)
set.seed(1234)

if (!dir.exists("data")) dir.create("data")

train_file <- "data/pml-training.csv"
test_file  <- "data/pml-testing.csv"

if (!file.exists(train_file)) {
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
                      train_file)
}
if (!file.exists(test_file)) {
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
                      test_file)
}

training <- read.csv(train_file, na.strings = c("NA", "", "#DIV/0!"))
testing  <- read.csv(test_file,  na.strings = c("NA", "", "#DIV/0!"))

dim(training); dim(testing)
table(training$classe)
################################################

# 1. Drop the first 7 bookkeeping columns
#    (row id, user_name, 3 timestamps, new_window, num_window)
training <- training[, -(1:7)]
testing  <- testing[,  -(1:7)]

# 2. Drop columns that are mostly NA (the summary rows' aggregate stats)
na_prop <- colMeans(is.na(training))
keep    <- na_prop < 0.95
training <- training[, keep]
testing  <- testing[,  keep]

# 3. Drop any near-zero-variance predictors that remain
nzv <- nearZeroVar(training)
if (length(nzv) > 0) {
        training <- training[, -nzv]
        testing  <- testing[,  -nzv]
}

training$classe <- factor(training$classe)
dim(training); dim(testing)
#################################################

# ------------------------------------------------------------------------------
# 3. Data Splitting and Model Training
# ------------------------------------------------------------------------------
set.seed(1234)
inTrain   <- createDataPartition(training$classe, p = 0.7, list = FALSE)
train_set <- training[inTrain, ]
valid_set <- training[-inTrain, ]

ctrl <- trainControl(method = "cv", number = 5)

set.seed(1234)
fit_rf <- train(classe ~ ., data = train_set, method = "rf",
                trControl = ctrl, ntree = 100)

cat("\n--- Random Forest: Cross-Validation Results ---\n")
print(fit_rf)
cat("Best mtry:", fit_rf$bestTune$mtry, "\n")
cat("CV Accuracy:", round(max(fit_rf$results$Accuracy), 4), "\n")

###
# ------------------------------------------------------------------------------
# 4. Out-of-Sample Validation
# ------------------------------------------------------------------------------
pred_valid <- predict(fit_rf, newdata = valid_set)
cm         <- confusionMatrix(pred_valid, valid_set$classe)

cat("\n--- Validation Set Performance ---\n")
print(cm$table)
cat("Accuracy:", round(cm$overall["Accuracy"], 4), "\n")
cat("Out-of-Sample Error:", round(1 - cm$overall["Accuracy"], 4), "\n")

plot(varImp(fit_rf), top = 15)


######
# ------------------------------------------------------------------------------
# 5. Prediction on the 20 Test Cases
# ------------------------------------------------------------------------------
pred_test <- predict(fit_rf, newdata = testing)

cat("\n--- Predictions for 20 Test Cases ---\n")
print(data.frame(problem_id = testing$problem_id, prediction = pred_test))
