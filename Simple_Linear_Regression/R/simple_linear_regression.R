#Simple Linear regression

#importing dataset
dataset = read.csv("Salary_Data.csv")
# dataset = dataset[, 2:3]

#Splitting the dataset into training dataset and test dataset
library(caTools)
set.seed(123)
split = dataset.split(dataset$Salary, SplitRatio = 2/3)
training_dataset = subset(dataset, split == TRUE)
test_dataset = subset(dataset, split == FALSE)


#Feature Scaling
# training_dataset[,2:3] = scale(training_dataset[,2:3])
# test_dataset[,2:3] = scale(test_dataset[,2:3])