import splitfolders
input_folder = "C:/Users/11453/PycharmProjects/riskassessment/data/COVID-19_Radiography_Dataset"
output = "C:/Users/11453/PycharmProjects/riskassessment/data/COVID-19_test_train"
splitfolders.ratio(input_folder, output=output, seed=42, ratio=(.7, .1, .2))
