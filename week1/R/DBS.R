d = read.csv("C:/Users/11453/PycharmProjects/riskassessment/NTU/week1/DBS.csv")

model = lm(d$DBS ~ d$SGD,data=d)

pred = predict(model,newdata = d)

print(pred)

err = d$DBS - pred

print(err)

rmse = mean(err^2)^0.5

print(rmse)
