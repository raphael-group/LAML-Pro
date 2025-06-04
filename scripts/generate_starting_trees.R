#!/usr/bin/env Rscript

# ------------------------------------------------------------
#  random_addition.R : generate starting trees using neighbor 
#  joining and random stepwise addition with parsimony.
# ------------------------------------------------------------

suppressPackageStartupMessages({
  library(optparse)
  library(phangorn)
  library(ape)
})

opt <- parse_args(OptionParser(option_list = list(
  make_option(c("-i", "--input"),  type = "character",
              help = "CSV file with character matrix (required)"),
  make_option(c("-n", "--nrep"),   type = "integer",  default = 10,
              help = "number of random‑addition replicates  [default %default]"),
  make_option(c("-o", "--output"), type = "character", default = "initial",
              help = "output Newick file                    [default %default]"),
  make_option(c("-s", "--seed"),   type = "integer",  default = 42,
              help = "RNG seed                                [default %default]")
)))

make_binary_and_root <- function(tr) {
    if (!is.binary(tr)) tr <- multi2di(tr, random = TRUE)
    if (is.null(tr$edge.length)) tr <- compute.brlen(tr, method = "equal") 
    midpoint(tr)
}

if (is.null(opt$input)) stop("Input CSV required.  Use -i / --input <file>", call. = FALSE)

set.seed(opt$seed)

# load data into a phyDat object
raw <- read.csv(opt$input, stringsAsFactors = FALSE, check.names = FALSE)
rownames(raw) <- raw[[1]]; raw[[1]] <- NULL
mat <- as.matrix(raw)
mode(mat) <- "character" 
st  <- sort(unique(na.omit(c(mat))))
dat <- phyDat(mat, type = "USER", levels = st)

# construct random stepwise addition trees using parsimony
ras <- replicate(opt$nrep, random.addition(dat) |> make_binary_and_root(), simplify = FALSE)

# construct single nj tree
dm      <- dist.hamming(dat, exclude = "pairwise")
nj_tree <- NJ(dm) |> make_binary_and_root()

# write random addition trees
tree_count <- 0
for (i in 1:opt$nrep) {
  tree_count <- tree_count + 1
  output_file <- sprintf("%s_stepwise_addition_%02d.nwk", opt$output, i)
  write.tree(ras[[i]], file = output_file)
}

# write NJ tree separately
output_file <- sprintf("%s_neighbor_joining.nwk", opt$output)
write.tree(nj_tree, file = output_file)

cat(sprintf("Wrote %d random addition trees and 1 neighbor joining tree to %s_*\n", 
            opt$nrep, opt$output))