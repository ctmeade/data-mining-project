#Librarys
library(data.table)
library(caret)

#Import Data
headers <- c("timesPregnant", "plasmaGlucose", "diastolicPressure", "tricepThickness", "serumInsulin", "bmi", "pedigreeFunction", "age", "diabetes")

data <- read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"),
                 header = FALSE,
                 col.names = headers)
data$diabetes <- as.factor(ifelse(data$diabetes == 0, "notDiabetic", "Diabetic"))
#Exploratory Analysis


###Dealing with missing values
#Data set not encoded with NA's
#But does have 0's where biologocially impossible
#Varibles include plasamaGlucose, diastolicPressure, tricepThickness, serumInsulin, bmi
#Only 768 observation, so can't drop observations with missing valuessa

#Take mean of nonmissing values in each column
means <- sapply(data[,1:8], function(x) mean(x[x>0]))

#Replace missing values with mean
processedVariables <- c("plasamaGlucose", "diastolicPressure", "tricepThickness", "serumInsulin", "bmi")



#Feature Selection
correlationMatrix <- cor(data[,1:8])
highCorrelation <- findCorrelation(correlationMatrix, cutoff=0.4)
print(highCorrelation)

#Construct 10fold CV method
tenFoldCV <- trainControl(method = "repeatedcv",
                          number = 10,
                          repeats = 10,
                          classProbs=TRUE,
                          summaryFunction=twoClassSummary)

#Standardization
x.data <- data[,1:8]
y.data <- as.factor(data[,9])

#NB b/c of assumption of independence
controlModel <- train(x.data,
                      y.data,
                      method = "rf", 
                      preProcess = c("center","scale"), 
                      trControl= tenFoldCV)

plot(controlModel, type=c("g", "o"))

svm <- train(x.data,
             y.data,
             method = "svmLinear",
             preProcess = c("center", "scale"),
             trControl = tenFoldCV)

svm

svm.tune <- train(x.data,
                  y.data,
                  method = "svmRadial",   # Radial kernel
                  tuneLength = 2,					# 8 values of the cost function
                  preProc = c("center","scale"),  # Center and scale data
                  metric="ROC",
                  trControl=tenFoldCV)