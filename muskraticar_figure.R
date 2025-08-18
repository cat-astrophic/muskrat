# Summary stats figure

library(modelsummary)

direc <- 'D:/muskrat/'

reg <- read.csv(paste0(direc, '/data/reg.csv'))

reg <- reg[,c(10, 25, 13:24)]

colnames(reg)[1] <- 'TSLA (USD)'
colnames(reg)[2] <- 'Market Trend (USD)'
colnames(reg)[3] <- 'Negative (Sentiment)'
colnames(reg)[4] <- 'Neutral (Sentiment)'
colnames(reg)[5] <- 'Positive (Sentiment)'
colnames(reg)[6] <- 'Anger (Emotion)'
colnames(reg)[7] <- 'Disgust (Emotion)'
colnames(reg)[8] <- 'Negative (Emotion)'
colnames(reg)[9] <- 'Joy (Emotion)'
colnames(reg)[10] <- 'Positive (Emotion)'
colnames(reg)[11] <- 'Fear (Emotion)'
colnames(reg)[12] <- 'Sadness (Emotion)'
colnames(reg)[13] <- 'Trust (Emotion)'
colnames(reg)[14] <- 'Surprise (Emotion)'

datasummary_skim(reg, fmt = '%.3f')

