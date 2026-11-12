# Writes cell256.png, the 256 x 256 greyscale crop used by the fourier2d demo.
# Re-run by hand from this folder:  Rscript make-image.R
# Source: the full-size micrograph, downloaded from the Wikimedia Commons URL below
# (not checked in). Kept here so the crop can be regenerated.
# Image: transmission electron micrograph of a mesenchymal stem cell,
# Robert M. Hunt, CC BY 3.0, via Wikimedia Commons
# https://commons.wikimedia.org/wiki/File:MSC_high_magnification.jpg

orig_file <- file.path(tempdir(), "msc-original.jpg")
if (!file.exists(orig_file)) {
  download.file("https://upload.wikimedia.org/wikipedia/commons/5/52/MSC_high_magnification.jpg",
                orig_file, mode = "wb")
}
big <- jpeg::readJPEG(orig_file)            # 2400 rows x 1803 columns
if (length(dim(big)) == 3) big <- big[, , 1]

# square crop around the cell, 1536 x 1536 pixels, then 6 x 6 block averages -> 256 x 256
f <- 6; n <- 256
crop <- big[350 + seq_len(f * n), 150 + seq_len(f * n)]
img <- matrix(colMeans(matrix(aperm(array(crop, c(f, n, f, n)), c(1, 3, 2, 4)), f * f)), n, n)
img <- (img - min(img)) / diff(range(img))   # stretch contrast to 0..1
png::writePNG(img, "cell256.png")
cat("wrote cell256.png\n")
