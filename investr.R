library(readr)
library(investr)
CH_TO <- read_csv("C:/Users/foresight_User/Desktop/公司文件/4.CIT季賽/測試資料/CH_TO.csv")



drops <- c("Context Name")
CH_TO <-CH_TO[ , !(names(CH_TO) %in% drops)]

colnames(CH_TO)<-make.names(colnames(CH_TO))

x <-colnames(CH_TO)
x <-x[ !x == 'Y']
y <- c("Y")
x <- make.names(x)
y <- make.names(y)

formula <- as.formula(paste(y, "~", paste(x, collapse = " + ")))

beetle.glm <- glm(formula, data = CH_TO)

predictor_names <- names(beetle.glm$coefficients)[-1]
# 進行反推預測，並將結果存儲在列表中
predictor_names[1]
invest(beetle.glm, y0 = 0.5, x0.name = predictor_names,newdata = CH_TO)

fixed_values <- list(x2 = 0, x3 = 0)

as.data.frame(fixed_values)

# 查看結果
names(results) <- predictor_names
print(results)