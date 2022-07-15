import splitfolders
input_folder = "C:/Users/11453/PycharmProjects/riskassessment/data/Dataset_BUSI_with_GT"
output = "C:/Users/11453/PycharmProjects/riskassessment/data/Dataset_BUSI_with_split"
splitfolders.ratio(input_folder, output=output, seed=42, ratio=(.7, .1, .2))
