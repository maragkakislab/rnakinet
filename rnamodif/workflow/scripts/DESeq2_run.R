library('DESeq2', quietly=T)

args <- commandArgs(trailingOnly = TRUE)

countDataPath <- args[1]
metadataPath <- args[2]

countData <- read.csv(countDataPath, header=TRUE, sep='\t')
metaData <- read.csv(metadataPath, header=TRUE, sep='\t')

dds <- DESeqDataSetFromMatrix(countData=countData, 
                              colData=metaData, 
                              design=~condition, tidy = TRUE)
dds <- DESeq(dds)
res <- results(dds)

write.table(res, args[3], sep='\t', row.names=TRUE, col.names=TRUE, dec=".")

